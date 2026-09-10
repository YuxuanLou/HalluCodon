#!/usr/bin/env python3
"""下载每个基因组的 CDS fasta，过滤/翻译/按蛋白去重，输出 unique.tsv。

过滤规则（与原有 filter.py 一致）:
  1. 长度必须为 3 的倍数
  2. 翻译产物（标准密码子表）不能有中间终止密码子
  3. 去掉末尾终止密码子后不能为空
  4. 按成熟蛋白序列全局去重（MD5），保留第一条出现的 CDS

用法:
  python3 fetch_and_filter.py --list species.txt --out unique.tsv \
      [--jobs 4] [--cache-dir cds_cache]

species.txt 每行: 物种名 \\t FTP路径 [\\t 谱系...]
下载地址按 NCBI 命名规则由 FTP 路径推导:
  <ftp>/<最后一段>_cds_from_genomic.fna.gz
下载失败（404 等）会记录到 <out>.failed。
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

# 标准遗传密码（NCBI 表 1/11），按密码子 T/C/A/G 顺序展开
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
    """下载到 dest；已存在且非空则跳过。返回 True/False（404 或失败均为 False）。"""
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
                return False  # 该基因组没有 CDS 注释文件，直接跳过
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    return False


def parse_records(path):
    """逐条读取 gz fasta，yield (header, seq)。"""
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
    """下载并处理一个基因组，返回 (rows, ok, reason)。rows 为 (cds, protein) 列表。"""
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
    """无缓存模式下的临时目录（进程结束自动清理）。"""
    import tempfile

    global _TMP_DIR
    if _TMP_DIR is None:
        _TMP_DIR = tempfile.mkdtemp(prefix="codon_pipeline_")
    return _TMP_DIR


_TMP_DIR = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", required=True, help="物种列表 TSV（物种名, FTP, ...）")
    ap.add_argument("--out", required=True, help="输出 TSV（cds_sequence, protein_sequence）")
    ap.add_argument("--jobs", type=int, default=4, help="并发下载数（默认 4）")
    ap.add_argument("--cache-dir", default="", help="CDS gz 缓存目录（断点续传，默认不保留）")
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

    print(f"[fetch_and_filter] 待处理 {len(items)} 条记录（并发 {args.jobs}）", flush=True)

    seen = set()  # 蛋白 MD5
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
                        f"[fetch_and_filter] {done}/{len(items)} 完成, "
                        f"成功 {ok_count}, 失败 {fail_count}, 唯一蛋白 {total_kept}, "
                        f"用时 {el:.0f}s",
                        flush=True,
                    )

    if fail_lines:
        fail_path = args.out + ".failed"
        with open(fail_path, "w", encoding="utf-8") as f:
            f.writelines(fail_lines)
        print(f"[fetch_and_filter] 警告: {fail_count} 条下载失败，详见 {fail_path}", flush=True)

    if ok_count == 0:
        print("[fetch_and_filter] 错误: 所有记录都下载失败", file=sys.stderr)
        sys.exit(1)

    print(
        f"[fetch_and_filter] 完成: 成功 {ok_count}, 失败 {fail_count}, "
        f"唯一蛋白 {total_kept}, 用时 {time.time()-t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
