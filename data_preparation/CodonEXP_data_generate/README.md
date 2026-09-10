# CodonEXP data generation pipeline

Converts PaxDb protein abundance data into CodonEXP datasets: splits proteins into
high/low expression by abundance, matches them back to species CDS sequences,
removes redundancy with cd-hit, and outputs an expression dataset CSV with codon sequences.

## Dependencies

- `python3` + `pandas` + `biopython`
- `blastp` / `makeblastdb` (NCBI BLAST+, must be on PATH)
- `cd-hit` (must be on PATH)
- `bash` (download and de-redundancy scripts)

## File list

| File | Purpose |
|---|---|
| `get-paxdb-proteins.sh` | Download PaxDb protein sequence FASTA |
| `get-paxdb-dataset.sh` | Download/list PaxDb dataset txt files |
| `get_cds-batch-blast.py` | Protein → CDS matching (batch blastp, no race conditions) |
| `cd-hit-pro.sh` | Remove redundancy at a 0.9 threshold based on protein_sequence |
| `paxdb-codonexp.py` | Pipeline script chaining all of the above steps |

## Usage steps

### 1. Download protein sequences (by species taxonomy id)

```bash
./get-paxdb-proteins.sh 3702 fasta.v11.5.3702.fa
# Downloaded from https://pax-db.org/downloads/5.0/paxdb-protein-sequences-v5.0/fasta.v11.5.{taxid}.fa
```

### 2. Download a dataset (e.g. the integrated abundance file for a tissue)

```bash
./get-paxdb-dataset.sh 3702 list              # List all datasets for the species
./get-paxdb-dataset.sh 3702 FLOWER-integrated 3702-FLOWER-integrated.txt
# Downloaded from https://pax-db.org/downloads/5.0/datasets/{taxid}/{taxid}-{name}.txt
# The dataset name may carry a {taxid}- prefix or a .txt suffix; the script normalizes it automatically
```

### 3. Run the pipeline (single dataset)

```bash
python3 paxdb-codonexp.py \
    --cds Arabidopsis_thaliana.TAIR10.cds.all.fa \
    --dataset 3702-FLOWER-integrated.txt \
    --proteins fasta.v11.5.3702.fa \
    --out 3702-FLOWER-0.9.csv \
    --threshold 90 --coverage 50 --parallel 8
```

| Argument | Description | Default |
|---|---|---|
| `--cds` | CDS FASTA (e.g. Ensembl Plants `*.cds.all.fa`) | required |
| `--dataset` | PaxDb dataset txt | required |
| `--proteins` | PaxDb protein sequence FASTA (output of step 1) | required |
| `--out` | Final output CSV | required |
| `--threshold` | Similarity threshold (%) = blastp pident | 90 |
| `--coverage` | Coverage threshold (%) = alignment length / max(query, subject) length | 50 |
| `--parallel` | Number of blastp threads | 8 |
| `--workdir` | Directory for intermediate files | `tmp/paxdb-codonexp-work/<dataset name>/` |

## Pipeline logic

1. **Parse the dataset**: skip `#` comment lines, read `string_external_id + abundance`
2. **High/low labeling**: sort by abundance descending, **top 1/3 labeled `high` (label=1), bottom 1/3 labeled `low` (label=0), middle 1/3 discarded**
3. **Fetch sequences**: extract sequences from the protein FASTA by ID; entries missing a sequence are dropped and counted
4. **Match CDS**: exact matches first (identical strings → similarity/coverage = 100); remaining queries are concatenated into one multi-sequence FASTA and run through **a single blastp** (first line = best HSP), filtered by the 90/50 dual thresholds
5. **Remove redundancy**: cluster `protein_sequence` with `cd-hit -c 0.9 -n 5`, keeping the representative rows (see `cd-hit-pro.sh` for the implementation)

## Output format

Same as existing CodonEXP CSVs:

```
id, abundance, protein_sequence_ori, exp, label, cds_sequence, protein_sequence, similarity, coverage
```

- `id`: PaxDb string_external_id (e.g. `3702.AT5G16970.1`)
- `exp`/`label`: `high`/1 or `low`/0
- `cds_sequence`: the CDS coding region truncated at the first stop codon
- `protein_sequence`: the translation of that CDS
- `similarity`: blastp pident (local alignment; 100 for exact matches)
- `coverage`: alignment length / max(query length, subject length) × 100

## Notes

- All temporary files (BLAST databases, cd-hit intermediates) are **created under the working directory `./tmp/` and deleted automatically at the end of the run**; the intermediate products `query.csv` / `matched.csv` are kept in the workdir for troubleshooting
- Verified: batch blastp and per-query blastp produce **identical results** (similarity/coverage/cds_sequence all equal) on 34,608 rows across 5 Arabidopsis tissues; the batch version is about 50–100x faster
- Running multiple tissues in parallel is conflict-free (temp directories and output files are isolated per dataset name); with enough cores, `--parallel 16` per dataset is recommended
- Proteins tied with boundary abundance values are split by file order; no tie handling is performed.
