import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from sklearn.model_selection import train_test_split


def load_sequences_from_csv(path: Path) -> List[List[str]]:
    seqs: List[List[str]] = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'code_tokens' not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'code_tokens' column")
        for row in reader:
            raw = row['code_tokens']
            if not raw:
                continue
            try:
                v = json.loads(raw)
                if isinstance(v, list):
                    toks = [str(x) for x in v if str(x).strip()]
                else:
                    toks = [t for t in str(v).split() if t]
            except Exception:
                toks = [t for t in raw.split() if t]
            if toks:
                seqs.append(toks)
    return seqs


@dataclass
class Config:
    n: int
    k: float

class Ngram:
    def __init__(self, cfg: Config):
        assert cfg.n >= 2
        self.cfg = cfg
        self.counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.ctx_counts: Counter = Counter()
        self.vocab: set[str] = set(['<BOS>', '<EOS>'])

    def fit(self, seqs: Iterable[List[str]]):
        n = self.cfg.n
        for s in seqs:
            s2 = ['<BOS>']*(n-1) + s + ['<EOS>']
            for i in range(len(s2) - n + 1):
                ctx = tuple(s2[i:i+n-1])
                tok = s2[i+n-1]
                self.counts[ctx][tok] += 1
                self.ctx_counts[ctx] += 1
                self.vocab.add(tok)

    def _dist(self, context: Sequence[str]) -> Dict[str, float]:
        n = self.cfg.n
        k = self.cfg.k
        V = len(self.vocab)
        ctx = list(context)[-(n-1):]
        if len(ctx) < n-1:
            ctx = ['<BOS>']*(n-1-len(ctx)) + ctx
        ctx_t = tuple(ctx)
        base = self.counts.get(ctx_t, Counter())
        denom = self.ctx_counts.get(ctx_t, 0) + k * V
        if denom <= 0:
            denom = k * V if V > 0 else 1.0
        return {w: (base.get(w, 0) + k) / denom for w in self.vocab}

    def predict_topk(self, context: Sequence[str], k_top: int = 10):
        dist = self._dist(context)
        return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k_top]

    def sample_next(self, context: Sequence[str]) -> str:
        dist = self._dist(context)
        items = list(dist.items())
        tokens, probs = zip(*items)
        s = sum(probs)
        r = random.random() * s
        c = 0.0
        for t, p in items:
            c += p
            if r <= c:
                return t
        return items[-1][0]

    def sample_completion(self, context: Sequence[str], max_len: int = 50) -> List[str]:
        out: List[str] = []
        ctx = list(context)
        for _ in range(max_len):
            t = self.sample_next(ctx)
            if t == '<EOS>':
                break
            out.append(t)
            ctx.append(t)
            if t == '}':
                break
        return out

    def logprob(self, seq: List[str]) -> float:
        n = self.cfg.n
        s = ['<BOS>']*(n-1) + seq + ['<EOS>']
        lp = 0.0
        for i in range(len(s) - n + 1):
            ctx = s[i:i+n-1]
            nxt = s[i+n-1]
            p = self._dist(ctx).get(nxt, 1e-12)
            if p <= 0:
                p = 1e-12
            lp += math.log(p)
        return lp

    def perplexity(self, seqs: Iterable[List[str]]) -> float:
        T = 0
        L = 0.0
        for seq in seqs:
            T += len(seq) + 1  # include EOS
            L += self.logprob(seq)
        return float('inf') if T == 0 else (math.exp(-L / T))


def select_by_val_pp(train_full: List[List[str]], n_list: List[int], k: float, val_ratio: float = 0.1) -> Tuple[Ngram, List[Tuple[int, float]]]:
    """
    Select the best N-gram model by validation perplexity.
    """
    train, val = train_test_split(train_full, test_size=val_ratio)
    best = None
    best_val = float('inf')
    records: List[Tuple[int, float]] = []
    for n in n_list:
        m = Ngram(Config(n=n, k=k))
        m.fit(train)
        vpp = m.perplexity(val)
        records.append((n, vpp))
        if vpp < best_val:
            best_val = vpp
            best = m
    assert best is not None
    return best, records


def build_contexts_from_test(test: List[List[str]], n: int, target: int) -> List[List[str]]:
    """
    Build contexts from test sequences for prediction sampling.
    """
    ctxs: List[List[str]] = []
    for seq in test:
        if len(seq) < 2:
            continue
        idxs = list(range(1, len(seq)))
        random.shuffle(idxs)
        for i in idxs[:5]:
            ctxs.append(seq[max(0, i-(n-1)): i])
            if len(ctxs) >= target:
                return ctxs
    while len(ctxs) < target:
        ctxs.append([])
    return ctxs


def write_predictions_jsonl(model: Ngram, test: List[List[str]], num_samples: int, topk: int, path: Path):
    n = model.cfg.n
    ctxs = build_contexts_from_test(test, n, num_samples)
    with path.open('w', encoding='utf-8') as f:
        for ctx in ctxs:
            top = model.predict_topk(ctx, k_top=topk)
            comp = model.sample_completion(ctx, max_len=50)
            rec = {
                'context': ctx,
                'topk': [t for t, _ in top],
                'probs': [float(p) for _, p in top],
                'sampled_completion': comp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description='N‑gram (train/test CSVs) for AI4SE Lab‑01')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--n_list', type=int, nargs='+', default=[3,5,7])
    ap.add_argument('--k', type=float, default=0.1)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--num_samples', type=int, default=1000)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    train_full = load_sequences_from_csv(Path(args.train_csv))
    test = load_sequences_from_csv(Path(args.test_csv))
    print(f"[INFO] TRAIN: {len(train_full)} sequences, TEST: {len(test)} sequences")

    best, records = select_by_val_pp(train_full, args.n_list, args.k, args.val_ratio)
    print(f"[INFO] Selected N={best.cfg.n} with val_perplexity={records[args.n_list.index(best.cfg.n)][1]:.6f}")

    # Test perplexity for the selected model
    test_pp = best.perplexity(test)

    # write outputs
    with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'val_perplexity'])
        for n, vpp in records:
            w.writerow([n, f'{vpp:.6f}'])
        w.writerow(['TEST_PERPLEXITY', f'{test_pp:.6f}'])
    print(f"[INFO] Test Perplexity of the best model: {test_pp:.6f}")

    # best_model.json
    (out / 'best_model.json').write_text(
        json.dumps({'N': best.cfg.n, 'k': best.cfg.k}, indent=2), encoding='utf-8')

    # predictions.jsonl (≥1000 from TEST)
    write_predictions_jsonl(best, test, args.num_samples, args.topk, out / 'predictions.jsonl')

    print('Done.')

if __name__ == '__main__':
    main()