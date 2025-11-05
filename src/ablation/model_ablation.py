import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Relative Position Bias
# -------------------------
class RelativePositionBias(nn.Module):
    """
    Learned relative position bias like T5:
    we create a table of size (2*max_distance+1, num_heads)
    and add bias b_{i-j} for attention score between i and j.
    """
    def __init__(self, num_heads, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias_table = nn.Parameter(torch.zeros(2 * max_distance + 1, num_heads))
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self, qlen, klen, device=None):
        """
        Return (num_heads, qlen, klen) bias tensor.
        """
        # compute relative position matrix (qlen x klen)
        range_q = torch.arange(qlen, device=device)
        range_k = torch.arange(klen, device=device)
        # shape (qlen, klen)
        distance = range_k.unsqueeze(0) - range_q.unsqueeze(1)
        # clip to [-max_distance, max_distance]
        distance_clipped = torch.clamp(distance, -self.max_distance, self.max_distance) + self.max_distance
        # index into table -> (qlen, klen, num_heads)
        bias = self.relative_bias_table[distance_clipped.view(-1)].view(qlen, klen, -1)
        # transpose to (num_heads, qlen, klen)
        bias = bias.permute(2, 0, 1).contiguous()
        return bias  # (num_heads, qlen, klen)


# -------------------------
# Scaled Dot-Product Attention (supports mask and rel-bias)
# -------------------------
def scaled_dot_product_attention(q, k, v, attn_mask=None, rel_bias=None, dropout=None):
    # q,k,v: (batch, heads, qlen, head_dim) and (batch, heads, klen, head_dim)
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (batch, heads, qlen, klen)
    if rel_bias is not None:
        # rel_bias: (heads, qlen, klen) -> broadcast to batch
        scores = scores + rel_bias.unsqueeze(0)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)  # (batch, heads, qlen, head_dim)
    return out, attn

# -------------------------
# Multi-head modules
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x):
        # x: (batch, seq, d_model) -> (batch, heads, seq, head_dim)
        b, seq, d = x.size()
        x = x.view(b, seq, self.num_heads, self.head_dim).permute(0,2,1,3)
        return x

    def forward(self, query, key, value, attn_mask=None, rel_bias=None):
        # query/key/value: (batch, seq, d_model)
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))
        out, attn = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, rel_bias=rel_bias, dropout=self.dropout)
        # out: (batch, heads, qlen, head_dim) -> (batch, qlen, d_model)
        out = out.permute(0,2,1,3).contiguous().view(query.size(0), query.size(1), self.d_model)
        out = self.out_proj(out)
        return out, attn

# -------------------------
# FFN and Layer with ablative options
# -------------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, use_gelu=True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() if use_gelu else nn.ReLU()
        # Only add dropout if probability > 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, rel_pos=None, use_gelu=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Pass ablative options to FFN
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, use_gelu)
        self.norm2 = nn.LayerNorm(d_model)
        # Only add dropout if probability > 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rel_pos = rel_pos

    def forward(self, x, src_mask=None):
        # x: (batch, seq, d_model)
        qlen = klen = x.size(1)
        rel_bias = None
        if self.rel_pos is not None:
            rel_bias = self.rel_pos(qlen, klen, device=x.device)
        attn_out, attn = self.self_attn(x, x, x, attn_mask=src_mask, rel_bias=rel_bias)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, rel_pos=None, use_gelu=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # Pass ablative options to FFN
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, use_gelu)
        self.norm3 = nn.LayerNorm(d_model)
        # Only add dropout if probability > 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rel_pos = rel_pos

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # masked self-attn
        qlen = klen = x.size(1)
        rel_bias_self = None
        if self.rel_pos is not None:
            rel_bias_self = self.rel_pos(qlen, klen, device=x.device)
        self_attn_out, self_attn = self.self_attn(x, x, x, attn_mask=tgt_mask, rel_bias=rel_bias_self)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # cross attention (query from decoder, key/value from encoder)
        qlen = x.size(1); klen = enc_out.size(1)
        rel_bias_cross = None
        cross_attn_out, cross_attn = self.cross_attn(x, enc_out, enc_out, attn_mask=memory_mask, rel_bias=rel_bias_cross)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        return x, self_attn, cross_attn

# -------------------------
# Full Encoder-Decoder with ablative options
# -------------------------
class TransformerEncDec(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_layers=6, num_heads=8, d_ff=1024,
                 max_len=256, dropout=0.1, rel_pos_max_distance=128, use_relpos=True, use_gelu=True):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_scale = math.sqrt(d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_enc, std=0.02)
        
        # Conditionally create relative position bias
        if use_relpos:
            self.rel_pos = RelativePositionBias(num_heads, max_distance=rel_pos_max_distance)
        else:
            self.rel_pos = None
            
        # Pass ablative options to layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, 
                                                          rel_pos=self.rel_pos, use_gelu=use_gelu)
                                             for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, 
                                                          rel_pos=self.rel_pos, use_gelu=use_gelu)
                                             for _ in range(num_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src_ids: (batch, src_len)
        tgt_ids: (batch, tgt_len)
        masks are boolean tensors where 1 indicates valid and 0 padded (we'll convert to attention masks).
        """
        b, src_len = src_ids.size()
        b, tgt_len = tgt_ids.size()
        device = src_ids.device

        # embeddings
        src = self.src_embed(src_ids) * self.pos_scale + self.pos_enc[:, :src_len, :].to(device)
        tgt = self.tgt_embed(tgt_ids) * self.pos_scale + self.pos_enc[:, :tgt_len, :].to(device)

        # convert pad masks to attention masks
        if src_mask is not None:
            src_attn_mask = src_mask.unsqueeze(1).unsqueeze(1)  # (batch,1,1,src_len)
        else:
            src_attn_mask = None
        if memory_mask is not None:
            memory_attn_mask = memory_mask.unsqueeze(1).unsqueeze(1)
        else:
            memory_attn_mask = src_attn_mask

        # encode
        enc_out = src
        enc_attns = []
        for layer in self.encoder_layers:
            enc_out, attn = layer(enc_out, src_mask=src_attn_mask)
            enc_attns.append(attn)
        enc_out = self.encoder_norm(enc_out)

        # prepare decoder causal mask for target
        # causal mask: allow i to attend to <= i
        if tgt_mask is not None:
            # tgt_mask: (batch, tgt_len)
            batch_masks = []
            for bidx in range(tgt_mask.size(0)):
                valid = tgt_mask[bidx].to(device)  # (tgt_len,)
                causal_mat = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device))
                # only allow positions j where valid[j]=1
                valid_cols = valid.unsqueeze(0).expand(tgt_len, tgt_len)
                combined = causal_mat & valid_cols
                batch_masks.append(combined)
            tgt_attn_mask = torch.stack(batch_masks, dim=0).unsqueeze(1)  # (batch,1,tgt_len,tgt_len)
        else:
            tgt_attn_mask = torch.tril(torch.ones((1,1,tgt_len,tgt_len), dtype=torch.bool, device=device)).bool()

        # decoder layers
        dec_out = tgt
        dec_attns = []
        cross_attns = []
        for layer in self.decoder_layers:
            dec_out, self_attn, cross_attn = layer(dec_out, enc_out, tgt_mask=tgt_attn_mask, memory_mask=memory_attn_mask)
            dec_attns.append(self_attn)
            cross_attns.append(cross_attn)
        dec_out = self.decoder_norm(dec_out)

        logits = self.output_proj(dec_out)  # (batch, tgt_len, tgt_vocab)
        return logits, enc_attns, dec_attns, cross_attns