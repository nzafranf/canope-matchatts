import torch
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
from difflib import SequenceMatcher

from data.augmenter import RuleBasedAugmentor
from data.tokenizer import PhonemeTokenizer
from datasets import load_dataset, Dataset as HFDataset
from typing import Optional
import random
import numpy as np


_REQUIRED_COLS = ["id", "text", "phoneme"]

_MAX_TEXT_CHARS = 512
_EXEMPT_ID_PREFIX = "NUS"


def _length_filter_batch(ids, texts, max_text_chars: int, exempt_id_prefix: str):
    keep = []
    for i, t in zip(ids, texts):
        if exempt_id_prefix and str(i).startswith(exempt_id_prefix):
            keep.append(True)
        else:
            keep.append(len(str(t)) < max_text_chars)
    return keep


def _load_and_clean(
    mode: str,
    dataset_path: Optional[str],
    max_text_chars: int = _MAX_TEXT_CHARS,
    exempt_id_prefix: str = _EXEMPT_ID_PREFIX,
) -> HFDataset:
    if mode == "huggingface":
        raw = load_dataset("avalonai/english-singlish-g2p")["train"]
    else:
        raw = load_dataset("csv", data_files=dataset_path)["train"]

    missing_required = [c for c in _REQUIRED_COLS if c not in raw.column_names]
    if missing_required:
        raise ValueError(
            "Dataset tidak memiliki kolom wajib: " + ", ".join(missing_required)
        )
    existing_required_cols = [c for c in _REQUIRED_COLS if c in raw.column_names]

    if existing_required_cols:
        raw = raw.filter(
            lambda ex: all(ex[c] is not None for c in existing_required_cols),
            desc="Membuang baris dengan kolom wajib null",
        )
    content_cols = [c for c in ("text", "phoneme") if c in raw.column_names]
    if content_cols:
        raw = raw.filter(
            lambda ex: all(str(ex[c]).strip() != "" for c in content_cols),
            desc="Membuang teks/fonem kosong",
        )

    for col in ("unnormalized_text", "negation", "phoneme_negation"):
        if col not in raw.column_names:
            raw = raw.add_column(col, ["-"] * len(raw))
    sentinel_cols = ["unnormalized_text", "negation", "phoneme_negation"]
    if sentinel_cols:
        def _fill_sentinel(example):
            for col in sentinel_cols:
                v = example[col]
                if v is None or str(v).strip() == "":
                    example[col] = "-"
            return example
        raw = raw.map(_fill_sentinel, desc="Normalisasi sentinel '-'")

    raw = raw.filter(
        lambda ex: len(str(ex["text"]).split()) > 1,
        desc="Membuang kalimat <=1 kata",
    )

    if max_text_chars is not None and "id" in raw.column_names and "text" in raw.column_names:
        raw = raw.filter(
            lambda batch: _length_filter_batch(
                batch["id"], batch["text"], max_text_chars, exempt_id_prefix
            ),
            batched=True,
            desc=(
                f"Membuang teks >={max_text_chars} char "
                f"(kecuali id berawalan '{exempt_id_prefix}')"
            ),
        )

    return raw


class BatchSampler(Sampler):
    def __init__(self, dataset: HFDataset, batch_size: int = 64, n_singlish_per_batch: int = 2,
                 n_negation_per_batch: int = 2, seed: int = 42, drop_last: bool = True,
                 require_phoneme_negation: bool = False, bucket_factor: int = 0):
        self.n_singlish = n_singlish_per_batch
        self.n_negation = n_negation_per_batch
        self.n_regular = batch_size - n_singlish_per_batch - n_negation_per_batch
        assert self.n_regular >= 0, "Kuota singlish + negation melebihi batch size"

        self.seed = int(seed)
        self.epoch = 0
        self._n = len(dataset)

        positions = np.arange(self._n)
        unnorm = np.array([str(v).strip() for v in dataset["unnormalized_text"]], dtype=object)
        neg = np.array([str(v).strip() for v in dataset["negation"]], dtype=object)
        phoneme_neg = np.array(
            [str(v).strip() for v in dataset["phoneme_negation"]], dtype=object
        )
        singlish_mask = unnorm != "-"
        if require_phoneme_negation:
            negation_mask = (neg != "-") & (phoneme_neg != "-") & ~singlish_mask
        else:
            negation_mask = (neg != "-") & ~singlish_mask

        self.singlish_idx = positions[singlish_mask].tolist()
        self.negation_idx = positions[negation_mask].tolist()
        special_mask = singlish_mask | negation_mask
        self.regular_idx = positions[~special_mask].tolist()

        assert len(self.singlish_idx) >= self.n_singlish
        assert len(self.negation_idx) >= self.n_negation
        assert len(self.regular_idx) >= self.n_regular

        self.n_batches = self._n // batch_size if drop_last else -(-self._n // batch_size)

        self.bucket_factor = max(0, int(bucket_factor))
        if self.bucket_factor > 1:
            def blen(arr):
                return np.array([0 if v == "-" else len(v.encode("utf-8"))
                                 for v in arr], dtype=np.int32)
            main = np.array([len(str(v).encode("utf-8")) for v in dataset["text"]],
                            dtype=np.int32)
            self._sort_len = np.maximum(np.maximum(main, blen(unnorm)), blen(neg))
        else:
            self._sort_len = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @staticmethod
    def _make_cycle(pool: list, rng: random.Random):
        pool = list(pool)
        while True:
            shuffled = pool[:]
            rng.shuffle(shuffled)
            for i in shuffled:
                yield i

    def _draw_unique(self, cycle_gen, k: int, exclude: set) -> list:
        picked = []
        seen = set()
        while len(picked) < k:
            i = next(cycle_gen)
            if i not in exclude and i not in seen:
                picked.append(i)
                seen.add(i)
        return picked

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        regular_cycle = self._make_cycle(self.regular_idx, rng)
        singlish_cycle = self._make_cycle(self.singlish_idx, rng)
        negation_cycle = self._make_cycle(self.negation_idx, rng)
        if self.bucket_factor > 1:
            yield from self._iter_bucketed(
                rng, regular_cycle, singlish_cycle, negation_cycle)
            return
        for _ in range(self.n_batches):
            regular = self._draw_unique(regular_cycle, self.n_regular, exclude=set())
            singlish = self._draw_unique(singlish_cycle, self.n_singlish, exclude=set(regular))
            negation = self._draw_unique(
                negation_cycle, self.n_negation, exclude=set(regular) | set(singlish)
            )
            negation_twins = [i + self._n for i in negation]

            yield regular + singlish + negation + negation_twins

    def _iter_bucketed(self, rng, regular_cycle, singlish_cycle, negation_cycle):
        k = self.bucket_factor
        sl = self._sort_len
        done = 0
        while done < self.n_batches:
            g = min(k, self.n_batches - done)
            regular = self._draw_unique(regular_cycle, g * self.n_regular, set())
            singlish = self._draw_unique(singlish_cycle, g * self.n_singlish,
                                         set(regular))
            negation = self._draw_unique(negation_cycle, g * self.n_negation,
                                         set(regular) | set(singlish))
            regular.sort(key=lambda i: sl[i])
            singlish.sort(key=lambda i: sl[i])
            negation.sort(key=lambda i: sl[i])
            order = list(range(g))
            rng.shuffle(order)
            for j in order:
                r = regular[j * self.n_regular:(j + 1) * self.n_regular]
                s = singlish[j * self.n_singlish:(j + 1) * self.n_singlish]
                n = negation[j * self.n_negation:(j + 1) * self.n_negation]
                yield r + s + n + [i + self._n for i in n]
            done += g

    def __len__(self):
        return self.n_batches


def collate_fn(batch, pad_value=0, max_seq_len: int = 4096):
    natural_max = max(seq.shape[0] for sample in batch for seq in sample["x"])
    max_len = max(1, min(natural_max, max_seq_len))

    batch_x = []
    batch_mask = []

    for sample in batch:
        padded = []
        masks = []
        for seq, l in zip(sample["x"], sample["x_lengths"]):
            l_capped = min(l, max_seq_len)
            if l_capped < 1:
                raise ValueError(f"Sekuens kosong pada sample id={sample.get('id')!r}")
            seq = seq[:l_capped]
            seq = F.pad(seq, (0, max_len - len(seq)), value=pad_value)
            padded.append(seq)

            m = torch.ones(l_capped, dtype=torch.bool)
            m = F.pad(m, (0, max_len - len(m)), value=False)
            masks.append(m)

        batch_x.append(torch.stack(padded))
        batch_mask.append(torch.stack(masks))

    return {
        "id": [b["id"] for b in batch],
        "x": torch.stack(batch_x),
        "texts": [
            [b["x_canon_text"], b["x_canon_text"]] + b["x_aug_1"] + b["x_aug_2"]
            for b in batch
        ],
        "x_lengths": [[min(l, max_seq_len) for l in b["x_lengths"]] for b in batch],
        "mask": torch.stack(batch_mask),
    }


class AugmentDataset(Dataset):
    def __init__(
        self,
        lexicon_path: str,
        dataset_path: Optional[str] = None,
        mode: str = "huggingface",
        hf_dataset: Optional[HFDataset] = None,
        tokenizer: Optional[PhonemeTokenizer] = None,
        max_text_chars: Optional[int] = _MAX_TEXT_CHARS,
        exempt_id_prefix: str = _EXEMPT_ID_PREFIX,
        canon_mode: str = "phoneme",
        n_aug_1: int = 3,
        n_aug_2: int = 2,
        max_corruption: float = 0.0,
        max_corruption_tries: int = 4,
    ):
        assert canon_mode in ("phoneme", "text"), (
            f"canon_mode harus 'phoneme' atau 'text', dapat {canon_mode!r}"
        )
        self.canon_mode = canon_mode
        self.n_aug_1 = n_aug_1
        self.n_aug_2 = n_aug_2
        self.max_corruption = float(max_corruption)
        self.max_corruption_tries = int(max_corruption_tries)
        if hf_dataset is not None:
            self.dataset = hf_dataset
        else:
            self.dataset = _load_and_clean(
                mode, dataset_path, max_text_chars=max_text_chars, exempt_id_prefix=exempt_id_prefix
            )

        if "phoneme_negation" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "phoneme_negation", ["-"] * len(self.dataset)
            )

        self.augmenter = RuleBasedAugmentor(lexicon_path=lexicon_path)

        if tokenizer is None:
            self.phoneme_tokenizer = PhonemeTokenizer.from_corpus(self.dataset["phoneme"])
        else:
            self.phoneme_tokenizer = tokenizer

        self._n = len(self.dataset)

    def _bounded(self, text: str, fn):
        if self.max_corruption <= 0:
            return fn(text)
        for _ in range(self.max_corruption_tries):
            out = fn(text)
            if not out.strip():
                continue
            if 1.0 - SequenceMatcher(None, text, out).ratio() <= self.max_corruption:
                return out
        out = self.augmenter.augment_easy(text)
        return out if out.strip() else text

    def __len__(self):
        return self._n

    def __getitem__(self, idx, n_aug_1=None, n_aug_2=None, canon_mode=None):
        n_aug_1 = self.n_aug_1 if n_aug_1 is None else n_aug_1
        n_aug_2 = self.n_aug_2 if n_aug_2 is None else n_aug_2
        canon_mode = self.canon_mode if canon_mode is None else canon_mode
        return self._getitem_impl(idx, n_aug_1, n_aug_2, canon_mode)

    def _getitem_impl(self, idx, n_aug_1=3, n_aug_2=2, canon_mode="phoneme"):
        assert canon_mode in ("phoneme", "text"), (
            "Unknown canon repr. return mode. It has to be either phoneme or text."
        )

        if idx < self._n:
            data = self.dataset[idx]
            text = data["text"]
            phoneme = data["phoneme"]
            unnormalized_text = data["unnormalized_text"]
            data_id = data["id"]
        else:
            base_idx = idx - self._n
            data = self.dataset[base_idx]
            assert data["negation"] != "-", (
                f"Baris {base_idx} tidak berlabel negasi, tidak valid diakses lewat idx negasi ({idx})"
            )
            text = data["negation"]
            phoneme = data["phoneme_negation"]
            if canon_mode == "phoneme":
                assert str(phoneme).strip() not in ("", "-"), (
                    f"phoneme_negation kosong/'-' untuk baris {base_idx}, "
                    f"cek preprocessing dulu"
                )
            unnormalized_text = "-"
            data_id = f"{data['id']}-NEG"

        x_aug_1, x_aug_2 = [], []

        for _ in range(n_aug_1):
            x_aug_1.append(self._bounded(text, self.augmenter.augment_easy))

        if n_aug_2 > 0:
            remaining = n_aug_2
            if unnormalized_text != "-":
                x_aug_2.append(unnormalized_text)
                remaining -= 1
            for _ in range(remaining):
                x_aug_2.append(
                    self._bounded(text, self.augmenter.augment_hard_surface))

        x_graph_canon = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        if canon_mode == "phoneme":
            x_anchor = self.phoneme_tokenizer.encode_single(phoneme)
        else:
            x_anchor = x_graph_canon.clone()

        x_v = x_aug_1 + x_aug_2
        x_v = [torch.tensor(list(t.encode("utf-8")), dtype=torch.long) for t in x_v]
        x_all = [x_anchor, x_graph_canon] + x_v
        x_lengths = [len(x_i) for x_i in x_all]

        return {
            "id": data_id,
            "x_canon": phoneme,
            "x_canon_text": text,
            "x_aug_1": x_aug_1,
            "x_aug_2": x_aug_2,
            "x": x_all,
            "x_lengths": x_lengths,
            "canon_mode": canon_mode,
        }
