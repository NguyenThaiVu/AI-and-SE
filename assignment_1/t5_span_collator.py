"""
This is just a helper data collator implementing T5-style span corruption.
"""

import math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

@dataclass
class DataCollatorForT5SpanCorruption:
    """
    T5-style span corruption for continued pretraining.

    - Replaces ~noise_density of tokens with sentinel tokens (<extra_id_k>) in inputs.
    - Targets are concatenation of the removed spans, each preceded by its sentinel.
    - Pads/trim inputs to `input_length` and targets to `target_length`.
    """
    tokenizer: Any
    noise_density: float = 0.15
    mean_span_length: float = 3.0
    input_length: int = 512
    target_length: int = 128
    pad_to_multiple_of: Optional[int] = None
    seed: int = 42

    def __post_init__(self):
        random.seed(self.seed)
        self.pad_id = self.tokenizer.pad_token_id
        assert self.pad_id is not None, "Tokenizer must have pad_token_id"
        self.sentinel_id = lambda k: self.tokenizer.convert_tokens_to_ids(f"<extra_id_{k}>")

    def _random_segmentation(self, num_items, num_segments):
        """Randomly partition num_items into num_segments positive lengths."""
        if num_segments <= 1:
            return [num_items]
        cuts = sorted(random.sample(range(1, num_items), num_segments - 1))
        segs, last = [], 0
        for c in cuts + [num_items]:
            segs.append(c - last)
            last = c
        return segs

    def _create_noise_mask(self, length):
        """Boolean mask choosing which tokens are replaced by sentinels (noise)."""
        num_noise = max(1, min(length - 1, int(round(self.noise_density * length))))
        num_spans = max(1, int(round(num_noise / self.mean_span_length)))
        # Create alternating non-noise and noise spans
        num_nonnoise = length - num_noise
        nonnoise_lens = self._random_segmentation(num_nonnoise, num_spans + 1)
        noise_lens    = self._random_segmentation(num_noise,    num_spans)
        # Interleave [non, noise, non, noise, ..., non]
        spans = []
        for a, b in zip(nonnoise_lens, noise_lens + [0]):
            spans += [0] * a + [1] * b
        # In case rounding caused mismatch
        spans = spans[:length] + [0] * max(0, length - len(spans))
        return torch.tensor(spans, dtype=torch.bool)

    def _strip_pad(self, ids: torch.Tensor):
        # remove trailing pads so span logic works on real tokens only
        if ids.ndim == 1:
            t = ids
            last_nonpad = (t != self.pad_id).nonzero(as_tuple=False)
            if last_nonpad.numel() == 0:
                return t[:0]
            return t[: last_nonpad[-1].item() + 1]
        raise ValueError("Expected 1D ids")

    def _insert_sentinels(self, tokens: torch.Tensor, noise_mask: torch.Tensor):
        """
        Build encoder input by replacing each contiguous noise span with a sentinel token.
        Also build decoder target as: <sentinel_0> noisy-span-0 <sentinel_1> noisy-span-1 ...
        """
        # Identify span boundaries
        # prepend a False and append a False to catch edges
        mask = noise_mask
        starts = (~mask[:-1] & mask[1:]).nonzero(as_tuple=False).flatten() + 1
        if mask[0]:
            starts = torch.cat([torch.tensor([0], device=mask.device), starts])
        ends = (mask[:-1] & ~mask[1:]).nonzero(as_tuple=False).flatten() + 1
        if mask[-1]:
            ends = torch.cat([ends, torch.tensor([mask.numel()], device=mask.device)])

        # Encoder input: keep non-noise tokens; insert sentinel at each noise start
        enc = []
        last = 0
        span_idx = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            enc.extend(tokens[last:s][~mask[last:s]].tolist())
            enc.append(self.sentinel_id(span_idx))
            last = e
            span_idx += 1
        
        # tail non-noise
        enc.extend(tokens[last:][~mask[last:]].tolist())
        if not enc:
            # avoid empty input
            enc = [self.sentinel_id(0)]

        enc = torch.tensor(enc, dtype=torch.long)

        # Decoder target: for each noise span, prepend sentinel_k then the noisy tokens
        dec = []
        span_idx = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            dec.append(self.sentinel_id(span_idx))
            dec.extend(tokens[s:e].tolist())
            span_idx += 1
        
        # End with EOS if desired
        dec = torch.tensor(dec, dtype=torch.long)
        return enc, dec


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        out_input_ids, out_attention_mask, out_labels = [], [], []

        for ids in batch_input_ids:
            ids = self._strip_pad(ids)
            # truncate if too long
            if ids.numel() > self.input_length - 1:
                ids = ids[: self.input_length - 1]

            length = ids.numel()
            length = max(length, 2)  # Ensure at least 2 tokens to mask
            if ids.numel() < length:
                ids = torch.cat([ids, torch.full((length - ids.numel(),), self.pad_id, dtype=torch.long)])

            # Create noise mask over the *non-pad* prefix
            nonpad_len = (ids != self.pad_id).sum().item()
            noise_mask = self._create_noise_mask(nonpad_len)
            # Extend mask to full length (pads are non-noise)
            if noise_mask.numel() < ids.numel():
                noise_mask = torch.cat([noise_mask, torch.zeros(ids.numel() - noise_mask.numel(), dtype=torch.bool)])

            enc, dec = self._insert_sentinels(ids, noise_mask)

            # pad/truncate to fixed lengths
            def pad_to(x: torch.Tensor, L: int, pad_token: int):
                if x.numel() >= L:
                    return x[:L]
                pad = torch.full((L - x.numel(),), pad_token, dtype=torch.long)
                return torch.cat([x, pad])

            enc = pad_to(enc, self.input_length, self.pad_id)
            attn = (enc != self.pad_id).long()
            dec = pad_to(dec, self.target_length, self.pad_id)

            out_input_ids.append(enc)
            out_attention_mask.append(attn)
            out_labels.append(dec)

        batch = {
            "input_ids": torch.stack(out_input_ids, dim=0),
            "attention_mask": torch.stack(out_attention_mask, dim=0),
            # labels: pad positions should be -100 so they’re ignored by loss
            "labels": torch.stack(out_labels, dim=0),
        }
        batch["labels"][batch["labels"] == self.pad_id] = -100
        return batch
