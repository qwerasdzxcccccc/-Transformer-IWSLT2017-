# src/eval_bleu.py
import torch
import argparse
from datasets import load_dataset
from tqdm import tqdm
import sacrebleu
from src.model_relpos import TransformerEncDec
from src.data_iwslt import prepare_iwslt_tokenizer, collate_fn
import numpy as np

@torch.no_grad()
def generate_greedy(model, src_tensor, src_mask, sp, max_len=80, device='cpu'):
    model.eval()
    batch_size = src_tensor.size(0)
    prev = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # BOS
    generated = prev
    for _ in range(max_len):
        logits, *_ = model(src_tensor, generated, src_mask=src_mask, tgt_mask=None, memory_mask=src_mask)
        next_logits = logits[:, -1, :]  # last token
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    # decode
    hyps = []
    for i in range(batch_size):
        hyp_ids = generated[i].cpu().numpy().tolist()
        if 2 in hyp_ids:
            idx = hyp_ids.index(2)
            hyp_ids = hyp_ids[1:idx]
        else:
            hyp_ids = hyp_ids[1:]
        hyps.append(sp.decode(hyp_ids))
    return hyps

def evaluate_bleu(ckpt_path, split='validation', src_lang='en', tgt_lang='de', max_len=80, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('args', {})
    sp = prepare_iwslt_tokenizer(save_dir='data/spm', vocab_size=args.get('vocab_size',16000))
    model = TransformerEncDec(src_vocab_size=sp.get_piece_size(), tgt_vocab_size=sp.get_piece_size(),
                              d_model=args.get('d_model',256), num_layers=args.get('num_layers',4),
                              num_heads=args.get('num_heads',8), d_ff=args.get('d_ff',1024),
                              max_len=args.get('max_len',128)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ds = load_dataset("IWSLT/iwslt2017", f"iwslt2017-{src_lang}-{tgt_lang}")
    split_ds = ds[split]
    refs = []
    hyps = []
    for ex in tqdm(split_ds.select(range(500))):  # sample 500 pairs for speed
        src_text = ex['translation'][src_lang]
        ref_text = ex['translation'][tgt_lang]
        src_ids = sp.encode(src_text, out_type=int)
        src_tensor = torch.tensor([ [1] + src_ids + [2] ], dtype=torch.long, device=device)
        src_mask = torch.ones_like(src_tensor, dtype=torch.bool, device=device)
        hyp_text = generate_greedy(model, src_tensor, src_mask, sp, max_len=max_len, device=device)[0]
        refs.append([ref_text])
        hyps.append(hyp_text)
    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    print(f"BLEU ({split}): {bleu.score:.2f}")
    return bleu.score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    evaluate_bleu(args.ckpt, split=args.split, device=args.device)
