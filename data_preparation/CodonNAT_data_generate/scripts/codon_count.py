#!/usr/bin/env python3
"""Count codon usage frequencies for all CDS in unique.tsv.

Usage:
  python3 codon_count.py --input unique.tsv --output codon.csv

Output CSV: aa, codon(RNA), count, frequency(%) (normalized per amino acid)
"""

import argparse
from collections import defaultdict

BASES = "TCAG"
AA_TABLE = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_AA = {}
_i = 0
for _b1 in BASES:
    for _b2 in BASES:
        for _b3 in BASES:
            CODON_AA[_b1 + _b2 + _b3] = AA_TABLE[_i]
            _i += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="unique.tsv (with a cds_sequence column)")
    ap.add_argument("--output", required=True, help="Output codon.csv")
    args = ap.parse_args()

    counts = defaultdict(int)
    with open(args.input, encoding="utf-8") as f:
        f.readline()  # Header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            seq = line.split("\t", 1)[0].strip().upper()
            for i in range(0, len(seq) // 3 * 3, 3):
                counts[seq[i : i + 3]] += 1

    aa_groups = defaultdict(dict)
    for codon, cnt in counts.items():
        aa = CODON_AA.get(codon, "X")  # Unknown codons are grouped as X (consistent with codon_freq/*.csv)
        rna = codon.replace("T", "U")
        aa_groups[aa][rna] = cnt

    # Standard amino acid order (same as count_codon_freq.py: ACDEFGHIKLMNPQRSTVWY*)
    aa_order = "ACDEFGHIKLMNPQRSTVWY*"
    sorted_aas = sorted(aa_groups.keys(), key=lambda a: aa_order.index(a) if a in aa_order else 99)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        f.write("aa,codon,count,frequency(%)\n")
        for aa in sorted_aas:
            codons = aa_groups[aa]
            total = sum(codons.values())
            # Sort within each group by count descending (same as count_codon_freq.py)
            for rna, cnt in sorted(codons.items(), key=lambda x: -x[1]):
                freq = cnt / total * 100 if total else 0.0
                f.write(f"{aa},{rna},{cnt},{freq:.2f}\n")

    print(f"[codon_count] Done: {len(counts)} codons -> {args.output}")


if __name__ == "__main__":
    main()
