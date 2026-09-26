#!/usr/bin/env python3
"""Download the CDS fasta of each genome, filter/translate/deduplicate by protein, and output unique.tsv.

Filtering rules (same as the original filter.py):
  1. Length must be a multiple of 3
  2. The translation (standard codon table) must contain no internal stop codons
  3. Must be non-empty after stripping the terminal stop codon
  4. Global deduplication by mature protein sequence (MD5), keeping the first CDS encountered

Usage:
  python3 fetch_and_filter.py --list species.txt --out unique.tsv \
      [--jobs 4] [--cache-dir cds_cache]

Each line of species.txt: species name \\t FTP path [\\t lineage...]
The download URL is derived from the FTP path following NCBI naming conventions:
  <ftp>/<last path segment>_cds_from_genomic.fna.gz
Download failures (404, etc.) are recorded in <out>.failed.
"""

import argparse
import gzip
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Standard genetic code (NCBI tables 1/11), expanded in codon order T/C/A/G
BASES = "TCAG"
AA_TABLE = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_AA = {}
_i = 0
for _b1 in BASES:
    for _b2 in BASES:
        for _b3 in BASES:
            CODON_AA[_b1 + _b2 + _b3] = AA_TABLE[_i]
            _i += 1


def translate(cds):
    out = []
    for i in range(0, len(cds) - 2, 3):
        out.append(CODON_AA.get(cds[i : i + 3], "X"))
    return "".join(out)


def build_url(ftp):
    ftp = ftp.rstrip("/")
    acc = ftp.rsplit("/", 1)[-1]
    return ftp + "/" + acc + "_cds_from_genomic.fna.gz", acc


def download(url, dest):
    """Download to dest; skip if it already exists and is non-empty. Returns True/False (404 or any failure -> False)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (codon-pipeline)"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, 1024 * 256)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("downloaded file is empty")
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False  # This genome has no CDS annotation file, skip it
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    return False


def parse_records(path):
    """Read a gzipped fasta record by record, yielding (header, seq)."""
    header = None
    seq_lines = []
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.upper())
        if header is not None:
            yield header, "".join(seq_lines)


def process_one(species, ftp, cache_dir):
    """Download and process one genome, returning (rows, ok, reason). rows is a list of (cds, protein)."""
    url, acc = build_url(ftp)
    if cache_dir:
        gz_path = os.path.join(cache_dir, acc + "_cds_from_genomic.fna.gz")
    else:
        gz_path = os.path.join(
            tempfile_dir(), acc + "_cds_from_genomic.fna.gz"
        )
    try:
        ok = download(url, gz_path)
        if not ok:
            return [], False, url
        rows = []
        n_records = 0
        for header, seq in parse_records(gz_path):
            n_records += 1
            if len(seq) % 3 != 0:
                continue
            prot = translate(seq)
            mature = prot.rstrip("*")
            if not mature or "*" in mature:
                continue
            rows.append((seq, mature))
        if not cache_dir:
            os.remove(gz_path)
        return rows, True, f"{n_records} records -> {len(rows)} kept"
    except Exception as e:
        return [], False, f"{url} ({e})"


def tempfile_dir():
    """Temporary directory used in no-cache mode (auto-cleaned at process exit)."""
    import tempfile

    global _TMP_DIR
    if _TMP_DIR is None:
        _TMP_DIR = tempfile.mkdtemp(prefix="codon_pipeline_")
    return _TMP_DIR


_TMP_DIR = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", required=True, help="Species list TSV (species name, FTP, ...)")
    ap.add_argument("--out", required=True, help="Output TSV (cds_sequence, protein_sequence)")
    ap.add_argument("--jobs", type=int, default=4, help="Number of parallel downloads (default 4)")
    ap.add_argument("--cache-dir", default="", help="Cache directory for CDS gz files (resumable downloads; not kept by default)")
    args = ap.parse_args()

    items = []
    with open(args.list, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2 or not parts[1]:
                continue
            items.append((parts[0], parts[1].strip()))

    print(f"[fetch_and_filter] {len(items)} records to process ({args.jobs} in parallel)", flush=True)

    seen = set()  # Protein MD5s
    total_kept = 0
    ok_count = 0
    fail_count = 0
    fail_lines = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(process_one, sp, ftp, args.cache_dir): (sp, ftp)
            for sp, ftp in items
        }
        with open(args.out, "w", encoding="utf-8") as fout:
            fout.write("cds_sequence\tprotein_sequence\n")
            for done, fut in enumerate(as_completed(futures), 1):
                sp, ftp = futures[fut]
                rows, ok, reason = fut.result()
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    fail_lines.append(f"{sp}\t{ftp}\t{reason}\n")
                for seq, prot in rows:
                    h = hashlib.md5(prot.encode("utf-8")).digest()
                    if h not in seen:
                        seen.add(h)
                        fout.write(f"{seq}\t{prot}\n")
                        total_kept += 1
                if done % 25 == 0 or done == len(items):
                    el = time.time() - t0
                    print(
                        f"[fetch_and_filter] {done}/{len(items)} done, "
                        f"succeeded {ok_count}, failed {fail_count}, unique proteins {total_kept}, "
                        f"elapsed {el:.0f}s",
                        flush=True,
                    )

    if fail_lines:
        fail_path = args.out + ".failed"
        with open(fail_path, "w", encoding="utf-8") as f:
            f.writelines(fail_lines)
        print(f"[fetch_and_filter] Warning: {fail_count} downloads failed, see {fail_path}", flush=True)

    if ok_count == 0:
        print("[fetch_and_filter] Error: all records failed to download", file=sys.stderr)
        sys.exit(1)

    print(
        f"[fetch_and_filter] Done: succeeded {ok_count}, failed {fail_count}, "
        f"unique proteins {total_kept}, elapsed {time.time()-t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
