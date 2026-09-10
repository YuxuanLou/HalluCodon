import pandas as pd
import numpy as np
import argparse
import math
from collections import defaultdict


def load_codon_table(codon_count_file):
    """Build the codon table from a codon frequency file"""
    codon_table = {}
    aa_max_count = defaultdict(float)

    # Read the codon frequency file
    df = pd.read_csv(codon_count_file)

    # Preprocess: find the most frequent codon count for each amino acid
    for _, row in df.iterrows():
        aa = row['aa']
        codon = row['codon']
        count = row['count']
        if count > aa_max_count[aa]:
            aa_max_count[aa] = count

    # Build codon table: {codon: (amino acid, relative adaptiveness)}
    for _, row in df.iterrows():
        aa = row['aa']
        codon = row['codon']
        count = row['count']
        if aa_max_count[aa] > 0:
            w_ij = count / aa_max_count[aa]
        else:
            w_ij = 0.0
        codon_table[codon] = (aa, w_ij)

    return codon_table


def calculate_csi(cds_sequence, codon_table):
    """Calculate the CSI value of a single CDS sequence"""
    total_ln_w = 0.0
    valid_codons = 0

    for i in range(0, len(cds_sequence), 3):
        codon = cds_sequence[i:i + 3].upper()
        _, w_ij = codon_table[codon]
        if w_ij > 0:
            total_ln_w += math.log(w_ij)
            valid_codons += 1

    if valid_codons > 0:
        csi = math.exp(total_ln_w / valid_codons)
    else:
        csi = 0.0

    return csi


def main():
    parser = argparse.ArgumentParser(
        description='Calculate CSI for CDS sequences and output top 10%.')
    parser.add_argument('--input', required=True,
                        help='Input CSV file with cds_sequence and protein_sequence columns.')
    parser.add_argument('--codon_freq',
                        required=True,
                        help='CSV file with codon frequency data (amino_acid,codon,count,frequency).')
    parser.add_argument('--output', required=True,
                        help='Output CSV file for top 10% CSI sequences.')
    args = parser.parse_args()

    # Load the codon table
    codon_table = load_codon_table(
        args.codon_freq)

    # Read the input sequence file
    df = pd.read_csv(args.input)

    # Compute CSI for each sequence
    df['csi_value'] = df['cds_sequence'].apply(
        lambda x: calculate_csi(x, codon_table))

    # Sort by CSI descending and take the top 10%
    df_sorted = df.sort_values(by='csi_value',
                               ascending=False)
    top_10_percent = df_sorted.head(
        int(len(df_sorted) * 0.1))

    # Save results
    top_10_percent[
        ['cds_sequence', 'protein_sequence',
         'csi_value']].to_csv(args.output,
                              index=False)
    print(
        f"Top 10% CSI sequences saved to {args.output}")


if __name__ == '__main__':
    main()

