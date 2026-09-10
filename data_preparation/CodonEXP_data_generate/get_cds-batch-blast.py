import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
import os
import subprocess
import tempfile
import re


def setup_blast_db(cds_proteins, temp_dir):
    """Create a BLAST database that encodes length information"""
    db_file = os.path.join(temp_dir,
                           "translated_cds.fasta")
    with open(db_file, "w") as f:
        for i, (cds, protein) in enumerate(
                cds_proteins):
            # Record the protein length in the sequence ID
            protein_length = len(protein)
            f.write(
                f">cds_{i}_{protein_length}\n{protein}\n")

    cmd = ["makeblastdb", "-in", db_file,
           "-dbtype", "prot"]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    return db_file


def run_blast_batch(query_items, db_file, temp_dir, num_threads):
    """Concatenate all queries into one multi-sequence FASTA and run a single blastp.

    Returns {row_idx: (cds_idx, pident, align_length)}
    """
    query_file = os.path.join(temp_dir,
                              "queries.fasta")
    with open(query_file, "w") as f:
        for row_idx, query_seq in query_items:
            f.write(f">q{row_idx}\n{query_seq}\n")

    output_file = os.path.join(temp_dir,
                               "blast_result.txt")
    cmd = [
        "blastp",
        "-query", query_file,
        "-db", db_file,
        "-out", output_file,
        "-outfmt",
        "6 qseqid sseqid pident length",
        "-max_target_seqs", "1",
        "-num_threads", str(num_threads)
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)

    results = {}
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r") as f:
            for line in f:
                fields = line.split()
                if len(fields) < 4:
                    continue
                qid, target_id = fields[0], fields[1]
                # A query may have multiple HSPs (multiple lines); the first line is the best hit
                if qid in results:
                    continue
                match = re.search(r"cds_(\d+)_(\d+)",
                                  target_id)
                if match:
                    results[qid] = (
                        int(match.group(1)),
                        float(fields[2]),
                        int(fields[3])
                    )
    return results


def translate_cds(cds_seq):
    """Translate a CDS sequence, truncating at the first stop codon"""
    # Find the position of the first stop codon
    stop_codons = ['TAA', 'TAG', 'TGA']
    stop_pos = len(cds_seq)

    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i:i + 3]
        if codon in stop_codons:
            stop_pos = i + 3  # Include the stop codon
            break

    # Truncate at the first stop codon
    truncated_cds = cds_seq[:stop_pos]

    # Check the length is a multiple of 3
    if len(truncated_cds) % 3 != 0:
        return None, None

    try:
        protein = str(
            Seq(truncated_cds).translate(
                to_stop=True))
        return truncated_cds, protein
    except Exception as e:
        print(f"Translation error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Batch BLAST sequence matching (single blastp run, no race conditions)')
    parser.add_argument('mpb_csv',
                        help='Input CSV file')
    parser.add_argument('fasta_file',
                        help='CDS FASTA file')
    parser.add_argument('output_csv',
                        help='Output CSV file')
    parser.add_argument('--threshold', type=float,
                        default=80.0,
                        help='Similarity threshold (%)')
    parser.add_argument('--coverage', type=float,
                        default=80.0,
                        help='Coverage threshold (%)')
    parser.add_argument('--parallel', type=int,
                        default=4,
                        help='Number of blastp threads')
    parser.add_argument('--temp_dir',
                        help='Temporary directory')
    args = parser.parse_args()

    if args.temp_dir:
        temp_dir = args.temp_dir
    else:
        # Created under ./tmp in the working directory by default, deleted when done (see cleanup at the end of the file)
        local_tmp = os.path.join(os.getcwd(), "tmp")
        os.makedirs(local_tmp, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="blast_", dir=local_tmp)
    os.makedirs(temp_dir, exist_ok=True)

    # Process CDS
    cds_proteins = []
    for record in SeqIO.parse(args.fasta_file,
                              'fasta'):
        truncated_cds, translated = translate_cds(str(record.seq))
        if truncated_cds and translated:
            cds_proteins.append(
                (truncated_cds, translated))
    print(f"Loaded CDS: {len(cds_proteins)} sequences")

    # BLAST database
    db_file = setup_blast_db(cds_proteins,
                             temp_dir)
    print(f"Database created: {db_file}")

    # Process data
    mpb_df = pd.read_csv(args.mpb_csv, sep=',')
    if 'seq' in mpb_df.columns and 'protein_sequence_ori' not in mpb_df.columns:
        mpb_df.rename(columns={
            'seq': 'protein_sequence_ori'},
                      inplace=True)

    # Exact matches first
    translated_set = {}
    for i, (cds, translated) in enumerate(cds_proteins):
        translated_set.setdefault(translated, i)

    results = {}
    blast_queries = []
    for idx, row in mpb_df.iterrows():
        protein_seq = str(row['protein_sequence_ori'])
        first_hit = translated_set.get(protein_seq)
        if first_hit is not None:
            cds, translated = cds_proteins[first_hit]
            row_dict = row.to_dict()
            row_dict.update({
                'cds_sequence': cds.replace('T', 'U'),  # DNA -> RNA
                'protein_sequence': translated,
                'similarity': 100.0,
                'coverage': 100.0
            })
            results[idx] = row_dict
        else:
            blast_queries.append((idx, protein_seq))

    print(f"Exact matches: {len(results)} sequences, sent to BLAST: {len(blast_queries)} sequences")

    if blast_queries:
        hits = run_blast_batch(blast_queries, db_file, temp_dir,
                               args.parallel)
        for idx, protein_seq in blast_queries:
            hit = hits.get(f"q{idx}")
            if hit is None:
                continue
            cds_idx, similarity, align_length = hit
            cds, translated = cds_proteins[cds_idx]
            query_len = len(protein_seq)
            target_len = len(translated)
            coverage = (align_length / max(query_len, target_len)) * 100
            if similarity >= args.threshold and coverage >= args.coverage:
                row_dict = mpb_df.loc[idx].to_dict()
                row_dict.update({
                    'cds_sequence': cds.replace('T', 'U'),  # DNA -> RNA
                    'protein_sequence': translated,
                    'similarity': similarity,
                    'coverage': coverage
                })
                results[idx] = row_dict

    # Save results (in input order)
    if results:
        result_df = pd.DataFrame(
            [results[i] for i in sorted(results)])
        result_df.to_csv(args.output_csv,
                         index=False)
        print(f"Results saved: {len(result_df)} sequences")
    else:
        pd.DataFrame().to_csv(args.output_csv,
                              index=False)
        print("No matching results")

    # Cleanup
    if not args.temp_dir:
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
