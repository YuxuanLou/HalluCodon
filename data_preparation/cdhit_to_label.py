#!/usr/bin/env python3
"""Stage 4 step 3: generate the final CodonEXP training CSV from cd-hit output.
Input: Pinus_cds_label_matched.csv (ID,cds_sequence_dna,protein_sequence,label)
      matched_prot_cdhit.fasta.clstr (cd-hit clustering results)
Logic: keep the label of each cluster representative (marked with *); discard the other sequences in the same cluster.
Output: Pinus_cds_label.csv: ID,cds_sequence,protein_sequence,label (DNA/T format)
"""
import csv
import re
import sys


def parse_clstr(clstr_file):
    """Parse the clstr file and return the seqN index of each representative sequence (marked with *)."""
    reps = []
    with open(clstr_file) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>Cluster'):
                continue
            if line.rstrip().endswith('*'):
                # e.g. 0	1021aa, >seq3181|PtXG09320|l... *
                m = re.search(r'>seq(\d+)\|', line)
                if m:
                    reps.append(int(m.group(1)))
    return reps


def main():
    matched_csv = sys.argv[1]
    clstr_file = sys.argv[2]
    out_csv = sys.argv[3]

    reps = parse_clstr(clstr_file)
    print(f"representatives from cd-hit: {len(reps)}")

    # Read full records back from the matched table, indexed by seqN
    by_seq = {}
    with open(matched_csv) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            by_seq[i] = row
    print(f"matched rows: {len(by_seq)}")

    # Assemble output
    out_rows = []
    for seqn in reps:
        rec = by_seq[seqn]
        out_rows.append([rec['ID'], rec['cds_sequence_dna'], rec['protein_sequence'], rec['label']])

    # Sort by ID for stability
    out_rows.sort(key=lambda x: x[0])

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'cds_sequence', 'protein_sequence', 'label'])
        for r in out_rows:
            w.writerow(r)
    print(f"saved: {out_csv}, rows: {len(out_rows)}")

    from collections import Counter
    print("label counts:", dict(Counter(r[3] for r in out_rows)))


if __name__ == '__main__':
    main()
