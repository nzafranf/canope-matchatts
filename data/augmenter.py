import random
import re
from typing import Literal, Optional
from collections import defaultdict
import json

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# Alias sangat pendek hanya aman bila memang merupakan acronym multi-kata atau
# bentuk numerik yang maknanya eksplisit. Lexicon sumber tetap lengkap; pagar
# ini hanya menentukan alias mana yang boleh dipakai sebagai positive view.
_TRUSTED_SHORT_ALIASES = {
    ("before", "b4"),
    ("twelve", "12"),
    ("twenty", "20"),
    ("twice", "2x"),
}

# Disimpan bersama run_config/checkpoint. Naikkan versi ini setiap kali aturan
# pemilihan alias berubah agar run lama tidak diam-diam dilanjutkan dengan
# distribusi positive-view yang berbeda.
ABBREVIATION_POLICY_VERSION = "injective-abbrev-v1"


def _normalize_lexeme(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _compact_alias(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize_lexeme(text))

_QWERTY_ADJACENT: dict[str, str] = {
    # row 1
    "q": "was",
    "w": "qeasd",
    "e": "wrsd f",
    "r": "etdfg",
    "t": "ryfgh",
    "y": "tughj",
    "u": "yihjk",
    "i": "uojkl",
    "o": "ipkl",
    "p": "ol;'[",
    # row 2
    "a": "qwszx",
    "s": "qweadzxc",
    "d": "wersxcv f",
    "f": "ertdcvb g",
    "g": "rtyfvbn h",
    "h": "tyugnbm j",
    "j": "yuihmn k",
    "k": "uiojml",
    "l": "iopk;.,",
    # row 3
    "z": "asx",
    "x": "zasdc",
    "c": "xsdvf",
    "v": "cdfgb",
    "b": "vfghn",
    "n": "bghj m",
    "m": "njkl, ",
    # digits → adjacent digits
    "1": "2q",
    "2": "13qw",
    "3": "24we",
    "4": "35er",
    "5": "46rt",
    "6": "57ty",
    "7": "68yu",
    "8": "79ui",
    "9": "80io",
    "0": "9op",
}

# Bersihkan spasi yang sengaja dipakai sebagai separator di atas
_QWERTY_ADJACENT = {k: v.replace(" ", "") for k, v in _QWERTY_ADJACENT.items()}

def qwerty_neighbor(ch: str, rng: Optional[random.Random] = None) -> str:
    rng = rng if rng is not None else random
    neighbors = _QWERTY_ADJACENT.get(ch.lower(), "")
    if not neighbors:
        return ch

    replacement = rng.choice(neighbors)
    return replacement.upper() if ch.isupper() else replacement

class AbbrevAugmenter:
    def __init__(
        self,
        lexicon_path: Optional[str],
        mode: Literal["first", "random"] = "random",
        seed: Optional[int] = None,
    ):
        assert mode in ("first", "random")
        self.mode = mode
        self._rng = random.Random(seed)

        self.uni_lexicon: dict[str, list[str]] = {}
        self.multi_lexicon: dict[tuple[str, ...], list[str]] = {}
        self.max_n = 5
        self._safe_alias_owner: dict[str, str] = {}
        self._safety_stats: dict[str, int] = {}

        self._load(lexicon_path)

    @staticmethod
    def _short_alias_is_safe(canonical: str, alias: str) -> bool:
        compact = _compact_alias(alias)
        if len(compact) >= 3:
            return True
        if (canonical, compact) in _TRUSTED_SHORT_ALIASES:
            return True

        # Acronym multi-kata yang persis mengikuti initial setiap kata:
        # direct message -> dm, don't know -> dk, content warning -> cw.
        words = _WORD_RE.findall(canonical)
        initials = "".join(word[0].casefold() for word in words)
        return len(words) >= 2 and compact == initials

    def _load(self, lexicon_path: Optional[str]) -> None:
        if not lexicon_path:
            self._safety_stats = {
                "entries_total": 0, "entries_active": 0,
                "aliases_total": 0, "aliases_kept": 0,
                "aliases_unique_kept": 0,
            }
            return

        with open(lexicon_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        canonical_vocabulary = {
            _normalize_lexeme(entry.get("word", ""))
            for entry in entries if _normalize_lexeme(entry.get("word", ""))
        }

        # Dua-pass wajib. Tanpa reverse index global, alias seperti "count"
        # tampak valid saat membaca "countries", padahal dipakai puluhan kata
        # lain. Semua perbandingan case-insensitive dan whitespace-normalized.
        reverse_owners: dict[str, set[str]] = defaultdict(set)
        aliases_total = 0
        for entry in entries:
            canonical = _normalize_lexeme(entry.get("word", ""))
            if not canonical:
                continue
            for raw_alias in entry.get("abbreviations", []):
                alias = _normalize_lexeme(raw_alias)
                if alias:
                    reverse_owners[alias].add(canonical)
                    aliases_total += 1

        uni_tmp = defaultdict(list)
        multi_tmp = defaultdict(list)
        rejected = defaultdict(int)
        safe_alias_owner = {}

        for entry in entries:
            word = str(entry.get("word", ""))
            canonical = _normalize_lexeme(word)
            tokens = tuple(w.casefold() for w in word.split())
            if not tokens:
                continue

            if len(tokens) == 1:
                bucket = uni_tmp[tokens[0]]
            else:
                bucket = multi_tmp[tokens]

            for raw_alias in entry.get("abbreviations", []):
                alias = _normalize_lexeme(raw_alias)
                if not alias:
                    rejected["empty"] += 1
                    continue
                if len(reverse_owners[alias]) != 1:
                    rejected["ambiguous_reverse"] += 1
                    continue
                if alias == canonical:
                    rejected["no_op"] += 1
                    continue
                if alias in canonical_vocabulary:
                    rejected["canonical_word_collision"] += 1
                    continue
                if not self._short_alias_is_safe(canonical, alias):
                    rejected["unsafe_short"] += 1
                    continue

                raw_alias = str(raw_alias)
                if raw_alias not in bucket:
                    bucket.append(raw_alias)
                safe_alias_owner[alias] = canonical

        # Buang canonical entry yang kehilangan seluruh alias aman, tetapi tidak
        # pernah mengubah JSON sumber. Dengan demikian kelengkapan lexicon tetap
        # tersedia untuk audit/kurasi berikutnya.
        self.uni_lexicon = {key: value for key, value in uni_tmp.items() if value}
        self.multi_lexicon = {key: value for key, value in multi_tmp.items() if value}
        self.max_n = max((len(key) for key in self.multi_lexicon), default=1)
        self._safe_alias_owner = safe_alias_owner
        self._safety_stats = {
            "entries_total": len(entries),
            "entries_active": len(self.uni_lexicon) + len(self.multi_lexicon),
            "aliases_total": aliases_total,
            "aliases_kept": sum(map(len, self.uni_lexicon.values()))
                            + sum(map(len, self.multi_lexicon.values())),
            "aliases_unique_kept": len(safe_alias_owner),
            "rejected_ambiguous_reverse": rejected["ambiguous_reverse"],
            "rejected_canonical_word_collision": rejected["canonical_word_collision"],
            "rejected_no_op": rejected["no_op"],
            "rejected_unsafe_short": rejected["unsafe_short"],
            "rejected_empty": rejected["empty"],
        }

    def safety_report(self) -> dict[str, int]:
        """Ringkasan filter untuk logging dan audit reproducibility."""
        return dict(self._safety_stats)

    def _pick(self, abbrevs: list[str], original_span_text: str) -> str:
        is_all_upper = original_span_text.isupper() and any(
            c.isalpha() for c in original_span_text
        )

        if is_all_upper:
            upper_candidates = [a for a in abbrevs if a.isupper()]
            if upper_candidates:
                return self._choose(upper_candidates)
            lower_candidates = [a for a in abbrevs if a.islower()]
            base = self._choose(lower_candidates) if lower_candidates else self._choose(abbrevs)
            return base.upper()

        lower_candidates = [a for a in abbrevs if a.islower()]
        if lower_candidates:
            return self._choose(lower_candidates)
        return self._choose(abbrevs)

    def _choose(self, candidates: list[str]) -> str:
        if self.mode == "random":
            return self._rng.choice(candidates)
        return candidates[0]

    @staticmethod
    def _tokenize(text: str):
        return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]

    @staticmethod
    def _gap_is_clean(gap_text: str) -> bool:
        return gap_text.strip() == ""

    def _roll(self, aug_p: float) -> bool:
        if aug_p >= 1.0:
            return True
        if aug_p <= 0.0:
            return False
        return self._rng.random() <= aug_p

    def augment(self, text: str, aug_p: float = 1.0, aug_max: Optional[int] = None) -> str:
        tokens = self._tokenize(text)
        n_tokens = len(tokens)

        out = []
        cursor = 0
        i = 0
        n_replaced = 0

        while i < n_tokens:
            consumed = 0

            can_replace_more = aug_max is None or n_replaced < aug_max

            if can_replace_more:
                max_possible_n = min(self.max_n, n_tokens - i)
                for n in range(max_possible_n, 1, -1):
                    span = tokens[i:i + n]

                    gaps_clean = all(
                        self._gap_is_clean(text[span[k][2]:span[k + 1][1]])
                        for k in range(n - 1)
                    )
                    if not gaps_clean:
                        continue

                    key = tuple(t[0].lower() for t in span)
                    abbrevs = self.multi_lexicon.get(key)
                    if abbrevs is None:
                        continue

                    if self._roll(aug_p):
                        start_char, end_char = span[0][1], span[-1][2]
                        original_span_text = text[start_char:end_char]

                        out.append(text[cursor:start_char])
                        out.append(self._pick(abbrevs, original_span_text))
                        cursor = end_char
                        consumed = n
                        n_replaced += 1

                    break

            if consumed == 0 and can_replace_more:
                word_text, start_char, end_char = tokens[i]
                abbrevs = self.uni_lexicon.get(word_text.lower())
                if abbrevs is not None and self._roll(aug_p):
                    out.append(text[cursor:start_char])
                    out.append(self._pick(abbrevs, word_text))
                    cursor = end_char
                    consumed = 1
                    n_replaced += 1

            if consumed == 0:
                consumed = 1

            i += consumed

        out.append(text[cursor:])
        return "".join(out)

class RuleBasedAugmentor:
    policy_version = ABBREVIATION_POLICY_VERSION

    def __init__(self, lexicon_path: Optional[str] = None, seed: Optional[int] = None):
        self.abbrev_augmenter = AbbrevAugmenter(lexicon_path, mode="random", seed=seed)
        self._rng = random.Random(seed)

    def safety_report(self) -> dict[str, int]:
        return self.abbrev_augmenter.safety_report()

    @staticmethod
    def keyboard_typo(text: str, p: float = 0.10, rng: Optional[random.Random] = None) -> str:
        rng = rng if rng is not None else random
        result = []
        for ch in text:
            if ch.isalnum() and rng.random() < p:
                result.append(qwerty_neighbor(ch, rng=rng))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def apostrophe_drop(text: str) -> str:
        return re.sub(r"['\u2019]", "", text)

    @staticmethod
    def case_variation(text: str, p: float = 0.10, rng: Optional[random.Random] = None) -> str:
        rng = rng if rng is not None else random
        words = text.split()
        new_words = []

        for w in words:
            if rng.random() < p:
                choice = rng.choice(["upper", "lower", "random"])
                if choice == "upper":
                    w = w.upper()
                elif choice == "lower":
                    w = w.lower()
                else:
                    w = "".join(ch.upper() if rng.random() > 0.7 else ch.lower() for ch in w)

                new_words.append(w)
            else:
                new_words.append(w)

        return " ".join(new_words)

    @staticmethod
    def char_swap(text: str, p: float = 0.05, rng: Optional[random.Random] = None) -> str:
        rng = rng if rng is not None else random
        chars = list(text)
        for i in range(len(chars) - 1):
            if chars[i].isalpha() and chars[i + 1].isalpha() and rng.random() < p:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]

        return "".join(chars)

    @staticmethod
    def char_drop(text: str, p: float = 0.05, rng: Optional[random.Random] = None) -> str:
        rng = rng if rng is not None else random
        words = text.split()
        new_words = []

        for w in words:
            if len(w) > 3 and rng.random() < p:
                idx = rng.randint(1, len(w) - 2)
                w = w[:idx] + w[idx + 1:]
            new_words.append(w)

        return " ".join(new_words)

    @staticmethod
    def char_repeat(text: str, p: float = 0.05, rng: Optional[random.Random] = None) -> str:
        rng = rng if rng is not None else random
        result = []
        for ch in text:
            result.append(ch)
            if ch.isalpha() and rng.random() < p:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def letter_space(
        text: str,
        target_words: Optional[list[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        rng = rng if rng is not None else random
        words = text.split()
        if not words:
            return text

        if target_words is not None:
            spaced = [
                " ".join(list(w)) if w.lower() in target_words else w
                for w in words
            ]
        else:
            short_indices = [i for i, w in enumerate(words) if 2 <= len(w) <= 5 and w.isalpha()]
            if short_indices:
                idx = rng.choice(short_indices)
                words[idx] = " ".join(list(words[idx]))
            spaced = words
        return " ".join(spaced)

    @staticmethod
    def word_concat(text: str, p: float = 0.2, rng: Optional[random.Random] = None) -> str:
        rng = rng if rng is not None else random
        words = text.split()
        if len(words) < 2:
            return text

        result = []
        i = 0
        while i < len(words):
            if i < len(words) - 1 and rng.random() < p:
                result.append(words[i] + words[i + 1])
                i += 2
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)

    @staticmethod
    def abbreviate(
        text: str,
        abbrev_augmenter: AbbrevAugmenter,
        aug_p: float = 1.0,
        aug_max: Optional[int] = None,
    ) -> str:
        return abbrev_augmenter.augment(text, aug_p=aug_p, aug_max=aug_max)

    def augment_easy(self, text: str) -> str:
        """
        Stage 1 easy – kombinasi acak dari rule-based transforms ringan.
        Setiap transform dipilih secara independen.
        """
        text = self.case_variation(text, rng=self._rng)

        if self._rng.random() < 0.7:
            text = self.apostrophe_drop(text)
        if self._rng.random() < 0.3:
            text = self.char_swap(text, p=0.1, rng=self._rng)
        if self._rng.random() < 0.2:
            text = self.char_drop(text, p=0.08, rng=self._rng)
        if self._rng.random() < 0.2:
            text = self.char_repeat(text, p=0.08, rng=self._rng)
        if self._rng.random() < 0.25:
            text = self.keyboard_typo(text, p=0.1, rng=self._rng)
        return text

    def augment_medium(self, text: str) -> str:
            if self._rng.random() < 0.35:
                text = self.abbreviate(text, self.abbrev_augmenter, aug_p=0.4, aug_max=2)
            text = self.case_variation(text, rng=self._rng)
            if self._rng.random() < 0.85:
                text = self.apostrophe_drop(text)
            if self._rng.random() < 0.25:
                text = self.letter_space(text, rng=self._rng)
            if self._rng.random() < 0.275:
                text = self.keyboard_typo(text, p=0.11, rng=self._rng)
            if self._rng.random() < 0.3:
                text = self.word_concat(text, p=0.175, rng=self._rng)
            return text

    def augment_hard_surface(self, text: str) -> str:
        if self._rng.random() < 0.7:
            text = self.abbreviate(text, self.abbrev_augmenter, aug_p=0.8, aug_max=3)
        text = self.case_variation(text, rng=self._rng)
        text = self.apostrophe_drop(text)
        if self._rng.random() < 0.5:
            text = self.letter_space(text, rng=self._rng)
        if self._rng.random() < 0.3:
            text = self.keyboard_typo(text, p=0.12, rng=self._rng)
        if self._rng.random() < 0.3:
            text = self.word_concat(text, p=0.25, rng=self._rng)
        return text
