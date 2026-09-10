import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
import os
import subprocess
import tempfile
import re


def setup_blast_db(cds_proteins, temp_dir):
    """创建包含长度信息的BLAST数据库"""
    db_file = os.path.join(temp_dir,
                           "translated_cds.fasta")
    with open(db_file, "w") as f:
        for i, (cds, protein) in enumerate(
                cds_proteins):
            # 在序列ID中记录蛋白质长度信息
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
    """所有查询合成一个多序列FASTA,一次blastp跑完。

    返回 {row_idx: (cds_idx, pident, align_length)}
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
                # 同一查询可能有多个HSP(多行),第一行是最佳命中
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
    """翻译CDS序列，截取到第一个终止密码子"""
    # 查找第一个终止密码子的位置
    stop_codons = ['TAA', 'TAG', 'TGA']
    stop_pos = len(cds_seq)

    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i:i + 3]
        if codon in stop_codons:
            stop_pos = i + 3  # 包括终止密码子
            break

    # 截取到第一个终止密码子
    truncated_cds = cds_seq[:stop_pos]

    # 检查是否是3的倍数
    if len(truncated_cds) % 3 != 0:
        return None, None

    try:
        protein = str(
            Seq(truncated_cds).translate(
                to_stop=True))
        return truncated_cds, protein
    except Exception as e:
        print(f"翻译错误: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description='批量BLAST序列匹配(单次blastp,无竞态)')
    parser.add_argument('mpb_csv',
                        help='输入CSV文件')
    parser.add_argument('fasta_file',
                        help='CDS FASTA文件')
    parser.add_argument('output_csv',
                        help='输出CSV文件')
    parser.add_argument('--threshold', type=float,
                        default=80.0,
                        help='相似性阈值(%)')
    parser.add_argument('--coverage', type=float,
                        default=80.0,
                        help='覆盖度阈值(%)')
    parser.add_argument('--parallel', type=int,
                        default=4,
                        help='blastp线程数')
    parser.add_argument('--temp_dir',
                        help='临时目录')
    args = parser.parse_args()

    if args.temp_dir:
        temp_dir = args.temp_dir
    else:
        # 默认在工作目录 ./tmp 下创建,用完即删(见文件末尾清理逻辑)
        local_tmp = os.path.join(os.getcwd(), "tmp")
        os.makedirs(local_tmp, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="blast_", dir=local_tmp)
    os.makedirs(temp_dir, exist_ok=True)

    # CDS处理
    cds_proteins = []
    for record in SeqIO.parse(args.fasta_file,
                              'fasta'):
        truncated_cds, translated = translate_cds(str(record.seq))
        if truncated_cds and translated:
            cds_proteins.append(
                (truncated_cds, translated))
    print(f"加载CDS: {len(cds_proteins)}条")

    # BLAST数据库
    db_file = setup_blast_db(cds_proteins,
                             temp_dir)
    print(f"数据库创建: {db_file}")

    # 数据处理
    mpb_df = pd.read_csv(args.mpb_csv, sep=',')
    if 'seq' in mpb_df.columns and 'protein_sequence_ori' not in mpb_df.columns:
        mpb_df.rename(columns={
            'seq': 'protein_sequence_ori'},
                      inplace=True)

    # 完全匹配优先
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

    print(f"完全匹配: {len(results)}条, 进入BLAST: {len(blast_queries)}条")

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

    # 结果保存(按输入顺序)
    if results:
        result_df = pd.DataFrame(
            [results[i] for i in sorted(results)])
        result_df.to_csv(args.output_csv,
                         index=False)
        print(f"保存结果: {len(result_df)}条")
    else:
        pd.DataFrame().to_csv(args.output_csv,
                              index=False)
        print("无符合结果")

    # 清理
    if not args.temp_dir:
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
