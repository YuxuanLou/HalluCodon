# CodonEXP 数据生成流程

把 PaxDb 蛋白丰度数据转换为 CodonEXP 数据集:按丰度划分高/低表达蛋白,匹配回物种 CDS 序列,cd-hit 去冗余,输出带密码子序列的表达数据集 CSV。

## 依赖

- `python3` + `pandas` + `biopython`
- `blastp` / `makeblastdb`(NCBI BLAST+,需在 PATH 中)
- `cd-hit`(需在 PATH 中)
- `bash`(下载与去冗余脚本)

## 文件清单

| 文件 | 作用 |
|---|---|
| `get-paxdb-proteins.sh` | 下载 PaxDb 蛋白序列 FASTA |
| `get-paxdb-dataset.sh` | 下载/列出 PaxDb 数据集 txt |
| `get_cds-batch-blast.py` | 蛋白 → CDS 匹配(批量 blastp,无竞态) |
| `cd-hit-pro.sh` | 按 protein_sequence 以 0.9 阈值去冗余 |
| `paxdb-codonexp.py` | 流程脚本,串联以下全部步骤 |

## 使用步骤

### 1. 下载蛋白序列(按物种 taxonomy id)

```bash
./get-paxdb-proteins.sh 3702 fasta.v11.5.3702.fa
# 下载自 https://pax-db.org/downloads/5.0/paxdb-protein-sequences-v5.0/fasta.v11.5.{taxid}.fa
```

### 2. 下载数据集(如某组织的整合丰度文件)

```bash
./get-paxdb-dataset.sh 3702 list              # 列出该物种全部数据集
./get-paxdb-dataset.sh 3702 FLOWER-integrated 3702-FLOWER-integrated.txt
# 下载自 https://pax-db.org/downloads/5.0/datasets/{taxid}/{taxid}-{name}.txt
# 数据集名可带 {taxid}- 前缀或 .txt 后缀,脚本会自动归一化
```

### 3. 运行流程(单数据集)

```bash
python3 paxdb-codonexp.py \
    --cds Arabidopsis_thaliana.TAIR10.cds.all.fa \
    --dataset 3702-FLOWER-integrated.txt \
    --proteins fasta.v11.5.3702.fa \
    --out 3702-FLOWER-0.9.csv \
    --threshold 90 --coverage 50 --parallel 8
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--cds` | CDS FASTA(如 Ensembl Plants 的 `*.cds.all.fa`) | 必填 |
| `--dataset` | PaxDb 数据集 txt | 必填 |
| `--proteins` | PaxDb 蛋白序列 FASTA(步骤 1 的产物) | 必填 |
| `--out` | 最终输出 CSV | 必填 |
| `--threshold` | 相似度阈值(%)= blastp pident | 90 |
| `--coverage` | 覆盖度阈值(%)= 比对长度/max(查询,目标)长度 | 50 |
| `--parallel` | blastp 线程数 | 8 |
| `--workdir` | 中间文件目录 | `tmp/paxdb-codonexp-work/<数据集名>/` |

## 流程内部逻辑

1. **解析数据集**:跳过 `#` 注释,读取 `string_external_id + abundance`
2. **高/低标注**:丰度降序,**前 1/3 标记 `high`(label=1),后 1/3 标记 `low`(label=0),中间 1/3 丢弃**
3. **取序列**:从蛋白序列 FASTA 按 ID 提取序列,缺序列的丢弃并计数
4. **匹配 CDS**:完全匹配优先(字符串相等 → similarity/coverage 记 100);其余查询合成一个多序列 FASTA **一次 blastp** 跑完(第一行 = 最佳 HSP),按 90/50 双阈值过滤
5. **去冗余**:`cd-hit -c 0.9 -n 5` 对 `protein_sequence` 聚类,保留代表行(实现见 `cd-hit-pro.sh`)

## 输出格式

与既有 CodonEXP CSV 一致:

```
id, abundance, protein_sequence_ori, exp, label, cds_sequence, protein_sequence, similarity, coverage
```

- `id`:PaxDb string_external_id(如 `3702.AT5G16970.1`)
- `exp`/`label`:`high`/1 或 `low`/0
- `cds_sequence`:截断到第一个终止密码子的 CDS 编码区
- `protein_sequence`:该 CDS 的翻译产物
- `similarity`:blastp pident(局部比对,完全匹配为 100)
- `coverage`:比对长度 / max(查询长度, 目标长度) × 100

## 注意事项

- 所有临时文件(BLAST 库、cd-hit 中间文件)在**工作目录 `./tmp/` 下创建,运行结束自动删除**;中间产物 `query.csv` / `matched.csv` 保留在 workdir 便于排查
- 已验证:批量 blastp 与逐条 blastp 在拟南芥 5 个组织共 34,608 行上结果**逐位一致**(similarity/coverage/cds_sequence 全同),批量版快约 50~100 倍
- 多组织并行跑互不冲突(临时目录、输出文件均按数据集名隔离);机器核数允许时建议每数据集 `--parallel 16`
- 边界同丰度的蛋白按文件顺序划分,不做平局处理
