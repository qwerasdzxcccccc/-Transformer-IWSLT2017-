# src/data_iwslt.py
import os
import sentencepiece as spm
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def prepare_iwslt_tokenizer(save_dir='data/spm', vocab_size=16000, src_lang='en', tgt_lang='de', sample_size=200000):
    """
    Load IWSLT2017 (en-de), collect source+target text to train a joint sentencepiece model.
    Produces model at save_dir/spm.model and spm.vocab.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_prefix = os.path.join(save_dir, 'spm')
    # If already trained, skip
    if os.path.exists(model_prefix + '.model'):
        sp = spm.SentencePieceProcessor()
        sp.load(model_prefix + '.model')
        return sp

    print("Loading IWSLT2017 dataset to train SentencePiece tokenizer...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    ds = load_dataset("iwslt2017", f"iwslt2017-{src_lang}-{tgt_lang}", trust_remote_code=True)
    # collect train split many examples
    texts = []
    for ex in ds['train']:
        src_text = ex['translation'][src_lang]
        tgt_text = ex['translation'][tgt_lang]
        texts.append(src_text)
        texts.append(tgt_text)
        if len(texts) >= sample_size:
            break
    tmp_file = os.path.join(save_dir, 'spm_corpus.txt')
    with open(tmp_file, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t.replace('\n',' ') + '\n')
    # train sentencepiece
    spm.SentencePieceTrainer.Train(f"--input={tmp_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=0.9995 --model_type=bpe")
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + '.model')
    print("Trained sentencepiece:", model_prefix + '.model')
    return sp

class TranslationDataset(Dataset):
    def __init__(self, split_ds, sp, src_lang='en', tgt_lang='de', max_len=128):
        self.examples = []
        self.sp = sp
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        for ex in split_ds:
            src = ex['translation'][src_lang]
            tgt = ex['translation'][tgt_lang]
            src_ids = sp.encode(src, out_type=int)[:max_len-1]
            tgt_ids = sp.encode(tgt, out_type=int)[:max_len-1]
            # add BOS/EOS (we use 1 as BOS, 2 as EOS if using default sp ids; but we will use sp.get_piece_size() for vocab size)
            self.examples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, sp, pad_id=0, bos_id=1, eos_id=2, max_len=128):
    # batch: list of (src_ids, tgt_ids)
    bs = len(batch)
    srcs = []
    tgts = []
    for s, t in batch:
        srcs.append([bos_id] + s + [eos_id])
        tgts.append([bos_id] + t + [eos_id])
    # pad
    src_max = min(max(len(x) for x in srcs), max_len)
    tgt_max = min(max(len(x) for x in tgts), max_len)
    src_tensor = torch.full((bs, src_max), pad_id, dtype=torch.long)
    tgt_tensor = torch.full((bs, tgt_max), pad_id, dtype=torch.long)
    src_mask = torch.zeros((bs, src_max), dtype=torch.bool)
    tgt_mask = torch.zeros((bs, tgt_max), dtype=torch.bool)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        ls = min(len(s), src_max)
        lt = min(len(t), tgt_max)
        src_tensor[i, :ls] = torch.tensor(s[:ls], dtype=torch.long)
        tgt_tensor[i, :lt] = torch.tensor(t[:lt], dtype=torch.long)
        src_mask[i, :ls] = 1
        tgt_mask[i, :lt] = 1
    return src_tensor, tgt_tensor, src_mask, tgt_mask

def get_dataloaders(batch_size=64, max_len=128, vocab_size=16000, src_lang='en', tgt_lang='de', num_workers=2, sample_size=200000):
    sp = prepare_iwslt_tokenizer(save_dir='data/spm', vocab_size=vocab_size,
                                 src_lang=src_lang, tgt_lang=tgt_lang, sample_size=sample_size)
    ds = load_dataset("iwslt2017", f"iwslt2017-{src_lang}-{tgt_lang}", trust_remote_code=True)
    train_ds = TranslationDataset(ds['train'], sp, src_lang=src_lang, tgt_lang=tgt_lang, max_len=max_len)
    valid_ds = TranslationDataset(ds['validation'], sp, src_lang=src_lang, tgt_lang=tgt_lang, max_len=max_len)
    pad_id = 0
    bos_id = 1
    eos_id = 2
    def collate(batch):
        return collate_fn(batch, sp, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    vocab_size_actual = sp.get_piece_size()
    return train_loader, valid_loader, sp, vocab_size_actual
