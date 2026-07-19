from __future__ import annotations
import hashlib
from typing import Iterable, List
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
MASK_IDX = len(AMINO_ACIDS)
def clean_sequence(seq: str) -> str:
    return "".join([c for c in str(seq).upper() if c.isalpha()])
def sequence_hash(seq: str) -> str:
    return hashlib.md5(str(seq).encode("utf-8")).hexdigest()
def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError(f"Cannot compare sequences with different lengths: {len(a)} vs {len(b)}")
    return sum(x != y for x, y in zip(a, b))
def mutation_line_simple(seq: str, wt: str, chain: str = "A") -> str:
    muts = [f"{w}{chain}{i}{m}" for i, (w, m) in enumerate(zip(wt, seq), start=1) if w != m]
    return ",".join(muts) + (";" if muts else "")
def seqs_to_tokens(seqs: Iterable[str]) -> List[List[int]]:
    return [[AA_TO_IDX[a] for a in clean_sequence(s)] for s in seqs]
