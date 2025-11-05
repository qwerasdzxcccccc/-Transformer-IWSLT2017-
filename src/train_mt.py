# src/train_mt.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.data_iwslt import get_dataloaders
from src.model_relpos import TransformerEncDec, RelativePositionBias
import sacrebleu
import numpy as np

def make_masks(src_mask, tgt_mask):
    # src_mask: (batch, src_len) bool
    # tgt_mask: (batch, tgt_len) bool
    return src_mask, tgt_mask

def train_one_epoch(model, loader, optimizer, device, criterion, epoch, grad_clip=None):
    model.train()
    total_loss = 0.0
    it = 0
    for src, tgt, src_mask, tgt_mask in tqdm(loader, desc=f"Train epoch {epoch}"):
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        optimizer.zero_grad()
        # inputs for teacher forcing: decoder input is tgt[:,:-1], predict tgt[:,1:]
        decoder_in = tgt[:, :-1]
        decoder_out = tgt[:, 1:]
        logits, *_ = model(src, decoder_in, src_mask=src_mask, tgt_mask=tgt_mask[:, :-1], memory_mask=src_mask)
        # logits: (batch, tgt_len-1, vocab)
        vocab_size = logits.size(-1)
        loss = criterion(logits.view(-1, vocab_size), decoder_out.contiguous().view(-1))
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        it += 1
    return total_loss / max(1, it)

def evaluate(model, loader, device, sp, max_gen_len=80):
    model.eval()
    total_loss = 0.0
    it = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    refs = []
    hyps = []
    with torch.no_grad():
        for src, tgt, src_mask, tgt_mask in tqdm(loader, desc="Validate"):
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            decoder_in = tgt[:, :-1]
            decoder_out = tgt[:, 1:]
            logits, *_ = model(src, decoder_in, src_mask=src_mask, tgt_mask=tgt_mask[:, :-1], memory_mask=src_mask)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), decoder_out.contiguous().view(-1))
            total_loss += loss.item()
            it += 1

            # generate greedy for BLEU (on small validation subset for speed)
            # use greedy decoding (not beam) for demonstration
            batch_size = src.size(0)
            # greedy decode
            prev = torch.tensor([[1]] * batch_size, device=device)  # BOS id=1
            generated = prev
            for _ in range(max_gen_len):
                logits_gen, *_ = model(src, generated, src_mask=src_mask, tgt_mask=None, memory_mask=src_mask)
                next_logits = logits_gen[:, -1, :]  # (batch, vocab)
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tokens], dim=1)
            # convert tokens to text
            for i in range(batch_size):
                # skip initial BOS
                hyp_ids = generated[i].cpu().numpy().tolist()
                # cut at EOS (2)
                if 2 in hyp_ids:
                    idx = hyp_ids.index(2)
                    hyp_ids = hyp_ids[1:idx]
                else:
                    hyp_ids = hyp_ids[1:]
                ref_ids = tgt[i].cpu().numpy().tolist()
                if 2 in ref_ids:
                    idx = ref_ids.index(2)
                    ref_ids = ref_ids[1:idx]
                else:
                    ref_ids = ref_ids[1:]
                hyp = sp.decode(hyp_ids)
                ref = sp.decode(ref_ids)
                hyps.append(hyp)
                refs.append([ref])
    # compute BLEU via sacrebleu
    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))  # sacrebleu expects refs list-of-lists
    return total_loss / max(1, it), bleu.score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=16000)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', type=str, default='results/run_experiments/run1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, valid_loader, sp, vocab_size_actual = get_dataloaders(batch_size=args.batch_size,
                                                                        max_len=args.max_len,
                                                                        vocab_size=args.vocab_size)
    print("Actual vocab size from sp:", vocab_size_actual)
    model = TransformerEncDec(src_vocab_size=vocab_size_actual, tgt_vocab_size=vocab_size_actual,
                              d_model=args.d_model, num_layers=args.num_layers,
                              num_heads=args.num_heads, d_ff=args.d_ff,
                              max_len=args.max_len, rel_pos_max_distance=128).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_losses = []
    valid_losses = []
    bleus = []

    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, grad_clip=args.grad_clip)
        val_loss, bleu = evaluate(model, valid_loader, device, sp)
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        bleus.append(bleu)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, BLEU={bleu:.2f}")
        # save ckpt
        ckpt = {
            'model_state': model.state_dict(),
            'sp_model': None,  # sentencepiece files exist on disk
            'args': vars(args)
        }
        torch.save(ckpt, os.path.join(args.output_dir, f'ckpt_epoch{epoch}.pt'))

        # plot loss curve
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, label='train')
        plt.plot(range(1, len(valid_losses)+1), valid_losses, label='valid')
        plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
        plt.title('loss curve')
        plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
        plt.close()

        # save metrics
        np.save(os.path.join(args.output_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(args.output_dir, 'valid_losses.npy'), np.array(valid_losses))
        np.save(os.path.join(args.output_dir, 'bleus.npy'), np.array(bleus))

if __name__ == "__main__":
    main()
