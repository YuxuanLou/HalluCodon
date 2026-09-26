#!/usr/bin/env python3
"""Stage 4 step 1: match the abundance table to CDS.
Input: protein_abundance_sorted.csv (target_id,Intensity, descending), filtered_cds.csv (ID,cds,prot)
Logic:
  1) Drop Intensity=0 entries from the abundance table
  2) Strip the .X suffix from CDS IDs (Pt1G42640.1 -> Pt1G42640) and match against abundance-table IDs
  3) Keep only matched CDS
  4) Of the matched data: top 1/3 -> label=1, bottom 1/3 -> label=0, middle third discarded
Output: Pinus_cds_label_matched.csv: ID,cds_sequence_dna,protein_sequence,label
"""
import csv
import sys


def main():
    ab_file = sys.argv[1]
    cds_file = sys.argv[2]
    out_file = sys.argv[3]

    # 1. Read the abundance table, drop Intensity=0
    abundance = []  # (target_id, intensity)
    with open(ab_file) as f:
        r = csv.DictReader(f)
        for row in r:
            intensity = float(row['Intensity'])
            if intensity > 0:
                abundance.append((row['target_id'].strip(), intensity))
    # The abundance table is already sorted descending (Intensity from high to low)
    print(f"abundance entries (>0): {len(abundance)}")

    # Build the abundance ID set
    ab_ids = {tid for tid, _ in abundance}
    ab_rank = {tid: i for i, (tid, _) in enumerate(abundance)}

    # 2. Read the filtered CDS and match by suffix-stripped ID
    matched = []  # (gene_id, intensity, cds, prot)
    unmatched = 0
    with open(cds_file) as f:
        r = csv.DictReader(f)
        for row in r:
            cds_id = row['ID']
            gene_id = cds_id.split('.')[0]
            if gene_id in ab_ids:
                matched.append((gene_id, ab_rank[gene_id], row['cds_sequence_dna'], row['protein_sequence']))
            else:
                unmatched += 1
    print(f"CDS total: {sum(1 for _ in open(cds_file))-1}")
    print(f"matched: {len(matched)}, unmatched: {unmatched}")

    # 3. Sort by abundance rank (highest abundance first)
    matched.sort(key=lambda x: x[1])

    # 4. Top 1/3 label=1, bottom 1/3 label=0, middle discarded
    n = len(matched)
    third = n // 3
    high = matched[:third]
    low = matched[n - third:]
    print(f"n={n}, third={third}, high={len(high)}, low={len(low)}")

    with open(out_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'cds_sequence_dna', 'protein_sequence', 'label'])
        for gene_id, rank, cds, prot in high:
            w.writerow([gene_id, cds, prot, 1])
        for gene_id, rank, cds, prot in low:
            w.writerow([gene_id, cds, prot, 0])
    print(f"saved: {out_file}")


if __name__ == '__main__':
    main()
