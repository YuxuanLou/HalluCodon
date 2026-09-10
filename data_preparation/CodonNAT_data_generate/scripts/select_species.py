#!/usr/bin/env python3
"""从 assembly_summary_genbank.txt 中按谱系筛选指定分类下的物种。

输出 TSV（每行）: 物种名 \\t FTP路径 \\t 完整谱系（tab 分隔）

用法:
  python3 select_species.py --assembly-summary FILE --taxonomy-dir DIR \
      --taxon Enterobacteriaceae --out species.txt [--all-genomes]

默认每个物种只保留一个代表基因组（优先 RefSeq(GCF)，再优先完整基因组）；
加 --all-genomes 则保留该分类下的所有基因组记录。
"""

import argparse
import sys
import time


def load_names(path):
    """names.dmp -> {taxid: 学名}，优先 scientific name。"""
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
    """nodes.dmp -> (parent, rank)。"""
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
    """从指定 taxid 向上走到根，返回各级学名（含自身）。"""
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
    ap.add_argument("--taxonomy-dir", required=True, help="含 nodes.dmp 和 names.dmp 的目录")
    ap.add_argument("--taxon", required=True, help="目标分类名，如 Enterobacteriaceae")
    ap.add_argument("--out", required=True, help="输出 TSV")
    ap.add_argument("--all-genomes", action="store_true", help="保留所有基因组而不是每物种一条")
    args = ap.parse_args()

    t0 = time.time()
    target = args.taxon.strip().lower()
    print(f"[select_species] 加载 taxonomy ...", flush=True)
    names = load_names(f"{args.taxonomy_dir}/names.dmp")
    parent, rank = load_nodes(f"{args.taxonomy_dir}/nodes.dmp")
    print(f"[select_species] taxonomy 加载完成（{time.time()-t0:.1f}s）", flush=True)

    # 逐行扫描 assembly summary
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
                print(f"[select_species] 已扫描 {n_rows} 行, 命中 {len(candidates)}", flush=True)

    print(f"[select_species] 扫描完成: {n_rows} 行, 命中 {len(candidates)} 条记录", flush=True)
    if not candidates:
        print(f"[select_species] 错误: 分类 [{args.taxon}] 没有匹配到任何物种", file=sys.stderr)
        sys.exit(1)

    if not args.all_genomes:
        # 每个物种保留一条: 优先 GCF(RefSeq) > 完整基因组 > reference/representative > 文件顺序
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
        print(f"[select_species] 每物种取代表后: {len(selected)} 个物种", flush=True)
    else:
        selected = candidates

    with open(args.out, "w", encoding="utf-8") as f:
        for idx, sp_taxid, species, ftp, lg, cols in selected:
            f.write(species + "\t" + ftp + "\t" + "\t".join(lg) + "\n")

    print(f"[select_species] 输出: {args.out}（{len(selected)} 行）", flush=True)


if __name__ == "__main__":
    main()
