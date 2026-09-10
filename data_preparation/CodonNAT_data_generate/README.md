# CDS 分类数据集流水线（assembly_summary → top10_U.csv）

输入一个物种分类名（如 `Enterobacteriaceae`、`Escherichia`、`Mammalia`），
自动从 NCBI 下载该分类下物种的 CDS 序列，过滤翻译去重后计算密码子偏好与
CSI，最终输出 `top10_U.csv`（CSI 最高的前 10%，T 已转 U，与原脚本语义一致）。

## 目录结构

```
codon_pipeline/
├── 01_download_assembly_summary.sh   # 下载 NCBI 组装汇总表（支持断点续传）
├── 02_build_top10_U.sh               # 主流程入口：分类名 -> top10_U.csv
├── scripts/
│   ├── select_species.py             # 按谱系筛选物种 + 每物种取代表基因组
│   ├── fetch_and_filter.py           # 并行下载 CDS -> 过滤/翻译/按蛋白去重
│   ├── codon_count.py                # 密码子使用频率统计
│   ├── calculate_csi.py              # CSI 计算 + 取 top N
│   └── t2u.py                        # T -> U
├── data/                             # assembly summary + NCBI taxonomy（自动下载）
├── work/<分类名>/                    # 中间产物（species.txt / unique.tsv / codon.csv 等）
└── output/                           # 最终 top10_U.csv
```

## 用法

### 第一步：只下载 assembly summary（可选，02 脚本也会自动调用）

```bash
./01_download_assembly_summary.sh            # 默认当前目录
./01_download_assembly_summary.sh data/assembly_summary_genbank.txt
```

### 第二步：一键生成 top10_U.csv

```bash
# 完整跑一个分类（如整个肠杆菌科，数据量很大，见下方提示）
./02_build_top10_U.sh Enterobacteriaceae

# 小范围快速测试（属名也会被匹配，Escherichia 只有几个物种）
./02_build_top10_U.sh Escherichia

# 常用选项
./02_build_top10_U.sh Enterobacteriaceae -p 10 -j 8 -o output     # 前 10%（默认）
./02_build_top10_U.sh Escherichia -a /path/to/assembly_summary_genbank.txt
```

选项说明：

| 选项 | 含义 | 默认 |
|---|---|---|
| `-p P` | 取 CSI 最高的前 P%（默认 10，即与原脚本一致） | 10 |
| `-j N` | 并行下载线程数 | 4 |
| `-a FILE` | 指定已有的 assembly_summary_genbank.txt | `data/assembly_summary_genbank.txt` |
| `-o DIR` | 最终输出目录 | `output/` |

## 流程说明

1. 确保 `assembly_summary_genbank.txt` 存在（不存在则用 01 脚本下载，约 1.6 GB）。
2. 确保 NCBI taxonomy 数据存在（`nodes.dmp` / `names.dmp`，不存在则自动下载
   `taxdump.tar.gz` 并解压，约 40 MB）。
3. `select_species.py` 逐行扫描 assembly summary，用物种 TaxID 向上解析完整谱系，
   把谱系中包含目标分类名的记录全部筛出。默认每物种取一个代表基因组
   （优先 RefSeq/GCF，其次完整基因组）。
4. `fetch_and_filter.py` 按 NCBI 命名规则 `FTP路径/<最后一段>_cds_from_genomic.fna.gz`
   并行下载 CDS，边下边过滤：长度须为 3 的倍数、翻译无中间终止密码子、
   按成熟蛋白序列全局去重，输出 `unique.tsv`（下载文件会缓存在
   `work/<分类名>/cds_cache/`，重跑时跳过已下载的基因组）。
5. `codon_count.py` 统计全部 CDS 的密码子使用频率（按氨基酸归一化为百分比）。
6. `calculate_csi.py` 计算每条 CDS 的 CSI（密码子相对适应度的几何平均），
   取前 P%（默认 10，与原 `calculate_csi.py` 的 `--percent 10` 一致）。
7. `t2u.py` 把 `cds_sequence` 的 T 替换为 U，得到最终 `top10_U.csv`。

## 输出与中间文件

最终输出 `output/top10_U.csv`，列为：

```
cds_sequence,protein_sequence,csi_value
```

中间产物在 `work/<分类名>/`：

| 文件 | 内容 |
|---|---|
| `species.txt` | 筛选出的物种 + FTP + 谱系 |
| `cds_cache/` | 下载的 CDS gz 缓存（可删除以释放空间） |
| `unique.tsv` | 过滤去重后的全部 CDS/蛋白对 |
| `codon.csv` | 密码子频率 |
| `cds_top<P>.csv` | 前 P% 的筛选结果（T 转 U 之前） |
| `fetch_failed.txt` | 下载失败（404 等）的记录 |

## 注意事项

- 数据量：整个 `Enterobacteriaceae` 有约 6000 个物种，每物种取一个代表基因组
  也要下载约 10 GB 以上；建议先用 `Escherichia` 或更小的分类做测试。
- 匹配是谱系名匹配（大小写不敏感），所以 `Enterobacteriaceae`、`Escherichia`、
  `Homo sapiens` 这类名称都可以作为输入。
- 代码只依赖 Python 3 标准库（不依赖 BioPython / pandas / numpy），翻译使用
  NCBI 标准遗传密码表（表 1/11）。
- 中断后重跑同一分类：`cds_cache` 中的文件会被复用，不会重复下载。
