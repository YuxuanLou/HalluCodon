#!/usr/bin/env python3
"""Select species of a given taxon from assembly_summary_genbank.txt by lineage.

Output TSV (per line): species name \\t FTP path \\t full lineage (tab separated)

Usage:
  python3 select_species.py --assembly-summary FILE --taxonomy-dir DIR \
      --taxon Enterobacteriaceae --out species.txt [--all-genomes]

By default only one representative genome is kept per species (preferring RefSeq(GCF), then complete genomes);
with --all-genomes, all genome records under the taxon are kept.
"""

import argparse
import sys
import time


def load_names(path):
    """names.dmp -> {taxid: scientific name}, preferring "scientific name"."""
    name_of = {}
    seen_class = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            taxid, name, name_class = parts[0], parts[1], parts[3]
            if not taxid or not name:
                continue
            if taxid not in name_of or name_class == "scientific name":
                if taxid not in seen_class or seen_class[taxid] != "scientific name":
                    name_of[taxid] = name
                    seen_class[taxid] = name_class
    return name_of


def load_nodes(path):
    """nodes.dmp -> (parent, rank)."""
    parent = {}
    rank = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            taxid, par, r = parts[0], parts[1], parts[2]
            if taxid and par:
                parent[taxid] = par
                rank[taxid] = r or "no rank"
    return parent, rank


def lineage_names(taxid, parent, names):
    """Walk up from the given taxid to the root, returning scientific names at each level (including itself)."""
    out = []
    seen = set()
    cur = taxid
    while cur and cur not in seen:
        seen.add(cur)
        nm = names.get(cur, f"taxid:{cur}")
        if nm not in out:
            out.append(nm)
        if cur == "1":
            break
        cur = parent.get(cur)
    return out


LEVEL_PRIORITY = {
    "Complete Genome": 0,
    "Chromosome": 1,
    "Scaffold": 2,
    "Contig": 3,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assembly-summary", required=True, help="assembly_summary_genbank.txt")
    ap.add_argument("--taxonomy-dir", required=True, help="Directory containing nodes.dmp and names.dmp")
    ap.add_argument("--taxon", required=True, help="Target taxon name, e.g. Enterobacteriaceae")
    ap.add_argument("--out", required=True, help="Output TSV")
    ap.add_argument("--all-genomes", action="store_true", help="Keep all genomes instead of one per species")
    args = ap.parse_args()

    t0 = time.time()
    target = args.taxon.strip().lower()
    print(f"[select_species] Loading taxonomy ...", flush=True)
    names = load_names(f"{args.taxonomy_dir}/names.dmp")
    parent, rank = load_nodes(f"{args.taxonomy_dir}/nodes.dmp")
    print(f"[select_species] Taxonomy loaded ({time.time()-t0:.1f}s)", flush=True)

    # Scan the assembly summary line by line
    candidates = []          # (idx, species_taxid, species, ftp, lineage)
    lineage_cache = {}
    n_rows = 0
    with open(args.assembly_summary, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n_rows += 1
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 20:
                continue
            species_taxid = cols[6].strip()
            ftp = cols[19].strip()
            if not species_taxid or species_taxid == "0" or ftp in ("", "na"):
                continue
            if species_taxid not in lineage_cache:
                lineage_cache[species_taxid] = lineage_names(
                    species_taxid, parent, names
                )
            lg = lineage_cache[species_taxid]
            if not any(n.lower() == target for n in lg):
                continue
            species = cols[7].strip() or (lg[-1] if lg else species_taxid)
            candidates.append(
                (n_rows, species_taxid, species, ftp, lg, cols)
            )
            if n_rows % 500000 == 0:
                print(f"[select_species] Scanned {n_rows} rows, {len(candidates)} matches", flush=True)

    print(f"[select_species] Scan complete: {n_rows} rows, {len(candidates)} matching records", flush=True)
    if not candidates:
        print(f"[select_species] Error: no species matched taxon [{args.taxon}]", file=sys.stderr)
        sys.exit(1)

    if not args.all_genomes:
        # Keep one record per species: prefer GCF(RefSeq) > complete genome > reference/representative > file order
        best = {}
        for cand in candidates:
            idx, sp_taxid, species, ftp, lg, cols = cand
            acc = cols[0]
            refseq = 0 if acc.startswith("GCF_") else 1
            level = LEVEL_PRIORITY.get(cols[11], 4)
            cat = 0 if cols[4] in ("reference genome", "representative genome") else 1
            key = (refseq, level, cat, idx)
            old = best.get(sp_taxid)
            if old is None or key < old[0]:
                best[sp_taxid] = (key, cand)
        selected = [c[1] for c in best.values()]
        print(f"[select_species] After keeping one representative per species: {len(selected)} species", flush=True)
    else:
        selected = candidates

    with open(args.out, "w", encoding="utf-8") as f:
        for idx, sp_taxid, species, ftp, lg, cols in selected:
            f.write(species + "\t" + ftp + "\t" + "\t".join(lg) + "\n")

    print(f"[select_species] Output: {args.out} ({len(selected)} lines)", flush=True)


if __name__ == "__main__":
    main()
