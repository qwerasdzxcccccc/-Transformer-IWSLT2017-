# src/sample_mt.py
import torch
import argparse
from src.model_relpos import TransformerEncDec
from src.data_iwslt import prepare_iwslt_tokenizer

@torch.no_grad()
def translate_sentence(model, sp, sentence, device='cuda', max_len=80):
    model.eval()
    src_ids = sp.encode(sentence, out_type=int)
    src_tensor = torch.tensor([[1] + src_ids + [2]], dtype=torch.long, device=device)
    src_mask = torch.ones_like(src_tensor, dtype=torch.bool, device=device)
    prev = torch.tensor([[1]], dtype=torch.long, device=device)
    generated = prev
    for _ in range(max_len):
        logits, *_ = model(src_tensor, generated, src_mask=src_mask, tgt_mask=None, memory_mask=src_mask)
        next_logits = logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == 2:
            break
    hyp_ids = generated[0].cpu().numpy().tolist()[1:]
    if 2 in hyp_ids:
        hyp_ids = hyp_ids[:hyp_ids.index(2)]
    return sp.decode(hyp_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sentence', type=str, default="How are you today?")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt, map_location=args.device)
    params = ckpt['args']
    sp = prepare_iwslt_tokenizer(save_dir='data/spm', vocab_size=params['vocab_size'])
    model = TransformerEncDec(sp.get_piece_size(), sp.get_piece_size(),
                              d_model=params['d_model'], num_layers=params['num_layers'],
                              num_heads=params['num_heads'], d_ff=params['d_ff'],
                              max_len=params['max_len']).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    print("Input:", args.sentence)
    print("Output:", translate_sentence(model, sp, args.sentence, device=args.device))
