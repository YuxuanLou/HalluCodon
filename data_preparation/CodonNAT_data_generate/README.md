# CDS taxonomy dataset pipeline (assembly_summary → top10_U.csv)

Given a species taxon name (e.g. `Enterobacteriaceae`, `Escherichia`, `Mammalia`),
the pipeline automatically downloads CDS sequences of species in that taxon from NCBI,
filters, translates and deduplicates them, computes codon usage and CSI, and finally
outputs `top10_U.csv` (the top 10% by CSI, with T already converted to U — same
semantics as the original script).

## Directory structure

```
codon_pipeline/
├── 01_download_assembly_summary.sh   # Download the NCBI assembly summary (resumable)
├── 02_build_top10_U.sh               # Main entry point: taxon name -> top10_U.csv
├── scripts/
│   ├── select_species.py             # Filter species by lineage + pick one representative genome per species
│   ├── fetch_and_filter.py           # Download CDS in parallel -> filter/translate/deduplicate by protein
│   ├── codon_count.py                # Codon usage frequency counting
│   ├── calculate_csi.py              # CSI calculation + top N selection
│   └── t2u.py                        # T -> U
├── data/                             # Assembly summary + NCBI taxonomy (downloaded automatically)
├── work/<taxon>/                     # Intermediate files (species.txt / unique.tsv / codon.csv, etc.)
└── output/                           # Final top10_U.csv
```

## Usage

### Step 1: download the assembly summary only (optional; the 02 script calls it automatically as well)

```bash
./01_download_assembly_summary.sh            # Current directory by default
./01_download_assembly_summary.sh data/assembly_summary_genbank.txt
```

### Step 2: generate top10_U.csv in one command

```bash
# Full run for a taxon (e.g. the whole Enterobacteriaceae family; very large, see the tip below)
./02_build_top10_U.sh Enterobacteriaceae

# Small quick test (genus names are matched too; Escherichia has only a few species)
./02_build_top10_U.sh Escherichia

# Common options
./02_build_top10_U.sh Enterobacteriaceae -p 10 -j 8 -o output     # Top 10% (default)
./02_build_top10_U.sh Escherichia -a /path/to/assembly_summary_genbank.txt
```

Options:

| Option | Meaning | Default |
|---|---|---|
| `-p P` | Keep the top P% by CSI (default 10, matching the original script) | 10 |
| `-j N` | Number of parallel download threads | 4 |
| `-a FILE` | Path to an existing assembly_summary_genbank.txt | `data/assembly_summary_genbank.txt` |
| `-o DIR` | Final output directory | `output/` |

## Pipeline logic

1. Ensure `assembly_summary_genbank.txt` exists (downloaded via the 01 script if missing, ~1.6 GB).
2. Ensure the NCBI taxonomy data exists (`nodes.dmp` / `names.dmp`; if missing, `taxdump.tar.gz`
   is downloaded and extracted automatically, ~40 MB).
3. `select_species.py` scans the assembly summary line by line, resolves the full lineage by
   walking up from each species TaxID, and keeps all records whose lineage contains the target
   taxon name. By default one representative genome is kept per species (preferring RefSeq/GCF,
   then complete genomes).
4. `fetch_and_filter.py` downloads CDS files in parallel following the NCBI naming convention
   `FTP path/<last segment>_cds_from_genomic.fna.gz`, filtering on the fly: length must be a
   multiple of 3, translation must contain no internal stop codons, and sequences are globally
   deduplicated by mature protein, producing `unique.tsv` (downloaded files are cached in
   `work/<taxon>/cds_cache/`; genomes already downloaded are skipped on reruns).
5. `codon_count.py` counts codon usage frequencies across all CDS (normalized to percentages per amino acid).
6. `calculate_csi.py` computes the CSI of each CDS (geometric mean of codon relative adaptiveness)
   and keeps the top P% (default 10, matching `--percent 10` of the original `calculate_csi.py`).
7. `t2u.py` replaces T with U in `cds_sequence`, producing the final `top10_U.csv`.

## Outputs and intermediate files

Final output `output/top10_U.csv`, with columns:

```
cds_sequence,protein_sequence,csi_value
```

Intermediate files are in `work/<taxon>/`:

| File | Contents |
|---|---|
| `species.txt` | Selected species + FTP path + lineage |
| `cds_cache/` | Cache of downloaded CDS gz files (deletable to free space) |
| `unique.tsv` | All filtered, deduplicated CDS/protein pairs |
| `codon.csv` | Codon frequencies |
| `cds_top<P>.csv` | Top P% selection (before T-to-U conversion) |
| `fetch_failed.txt` | Records that failed to download (404, etc.) |

## Notes

- Data volume: `Enterobacteriaceae` contains about 6000 species; even with one representative
  genome per species this downloads more than ~10 GB. Start with `Escherichia` or a smaller taxon for testing.
- Matching is by lineage name (case-insensitive), so names such as `Enterobacteriaceae`,
  `Escherichia`, or `Homo sapiens` all work as input.
- The code depends only on the Python 3 standard library (no BioPython / pandas / numpy);
  translation uses the NCBI standard genetic code table (tables 1/11).
- Rerunning the same taxon after an interruption: files in `cds_cache` are reused, nothing is downloaded twice.
