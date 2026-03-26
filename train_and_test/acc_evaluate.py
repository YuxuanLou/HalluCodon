import os
import torch
import multimolecule
from multimolecule import RnaTokenizer
import pandas as pd
# 移除了不再需要的 train_test_split
from transformers import AutoTokenizer, AutoConfig, default_data_collator
import numpy as np
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from safetensors.torch import load_file

# 导入已经实现的模型和数据整理器
# 请确保当前目录下有 mrnafm_pro_mlm.py 和 utils.py
from mrnafm_pro_mlm import CustomPlantRNAModelmlm
from utils import CustomDataset

# 设置随机种子以保证结果可重现
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class MLMDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, examples):
        batch = default_data_collator(examples)
        device = batch["cds_input_ids"].device
        inputs = batch["cds_input_ids"].clone()
        labels = inputs.clone()

        # 1. 构建概率矩阵
        probability_matrix = torch.full(
            inputs.shape, self.mlm_probability,
            device=device)

        # 2. 排除特殊字符 (PAD, BOS, EOS)
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for special_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
            if special_id is not None:
                special_tokens_mask = special_tokens_mask | (inputs == special_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(~batch["cds_attention_mask"].bool(), value=0.0)

        # 3. 确定哪些位置需要被预测 (约15%)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 4. 设置标签：不需要预测的地方设为 -100
        labels[~masked_indices] = -100

        # 5. 强制 Mask：将所有被选中的位置 (masked_indices) 全部替换为 mask_token_id
        # 不再保留原词，也不替换为随机词
        inputs[masked_indices] = self.mask_token_id

        batch["cds_input_ids"] = inputs
        batch["labels"] = labels
        batch["masked_indices"] = masked_indices
        # 此时 mask_token_positions 与 masked_indices 完全一致
        batch["mask_token_positions"] = masked_indices

        return batch


# RNA密码子到氨基酸的映射函数
def translate_codon_to_aa(codon):
    """将RNA密码子翻译为对应的氨基酸"""
    # 定义RNA密码子到氨基酸的映射表
    codon_table = {
        # U开头的密码子
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
        'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
        # C开头的密码子
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        # A开头的密码子
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        # G开头的密码子
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    # 检查密码子是否含有模糊核苷酸N
    if 'N' in codon or codon not in codon_table:
        return 'X'

    return codon_table.get(codon, 'X')


# 加载密码子频率文件，获取每个氨基酸的最高频率密码子
def load_codon_frequency(codon_freq_file):
    """加载密码子频率文件，返回每个氨基酸对应的最高频率密码子"""
    df = pd.read_csv(codon_freq_file)

    # 获取每个氨基酸的最高频率密码子
    best_codons = {}
    for aa in df['氨基酸'].unique():
        aa_df = df[df['氨基酸'] == aa]
        best_codon = aa_df.loc[
            aa_df['频率(%)'].idxmax(), '密码子']
        best_codons[aa] = best_codon

    return best_codons


# 在测试集上评估模型
def evaluate_model_on_test(model, test_dataset,
                           data_collator, device,
                           cds_tokenizer,
                           best_codons):
    """在测试集上评估模型性能和比较与BFC的准确率"""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model.eval()

    # 按氨基酸分类的准确率统计
    model_aa_correct = defaultdict(int)
    model_aa_total = defaultdict(int)

    # BFC准确率统计
    bfc_aa_correct = defaultdict(int)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        collate_fn=data_collator,
        shuffle=False
    )

    with torch.no_grad():
        for batch in test_dataloader:
            # 将数据移到设备上
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(**batch)

            # 获取预测结果
            predictions = torch.argmax(outputs["logits"], dim=-1)
            mlm_labels = batch['labels']

            # 只考虑非填充和非特殊标记的位置
            mask = mlm_labels != -100

            # 统计每个氨基酸的准确率
            masked_indices = mask.nonzero(as_tuple=True)  # 获取被mask的位置
            for pos_idx, seq_idx in zip(*masked_indices):
                true_token_id = mlm_labels[pos_idx, seq_idx].item()
                pred_token_id = predictions[pos_idx, seq_idx].item()

                # 转换为密码子
                true_codon = cds_tokenizer.convert_ids_to_tokens([true_token_id])[0]
                pred_codon = cds_tokenizer.convert_ids_to_tokens([pred_token_id])[0]

                # 翻译为氨基酸
                true_aa = translate_codon_to_aa(true_codon)

                if true_aa == 'X' or true_aa == '*':
                    continue

                # 更新模型准确率统计
                model_aa_total[true_aa] += 1
                if true_token_id == pred_token_id:
                    model_aa_correct[true_aa] += 1

                # 更新BFC准确率统计
                if true_aa in best_codons and true_codon == best_codons[true_aa]:
                    bfc_aa_correct[true_aa] += 1

    # 计算每个氨基酸的模型准确率和BFC准确率
    model_aa_accuracy = {
        aa: model_aa_correct[aa] / model_aa_total[aa] if model_aa_total[aa] > 0 else 0
        for aa in model_aa_total}
    bfc_aa_accuracy = {
        aa: bfc_aa_correct[aa] / model_aa_total[aa] if model_aa_total[aa] > 0 else 0
        for aa in model_aa_total}

    # 计算总体准确率
    model_total_correct = sum(model_aa_correct.values())
    bfc_total_correct = sum(bfc_aa_correct.values())
    total_predictions = sum(model_aa_total.values())

    model_overall_accuracy = model_total_correct / total_predictions if total_predictions > 0 else 0
    bfc_overall_accuracy = bfc_total_correct / total_predictions if total_predictions > 0 else 0

    # 添加总体准确率
    model_aa_accuracy['overall'] = model_overall_accuracy
    bfc_aa_accuracy['overall'] = bfc_overall_accuracy

    return model_aa_accuracy, bfc_aa_accuracy


def main():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 添加参数解析器来获取输出目录和数据集路径
    parser = argparse.ArgumentParser(
        description='评估模型准确率与BFC比较')
    parser.add_argument('--model_path', type=str,
                        required=True,
                        help='预训练模型路径')
    # 修改部分：使用 --test 参数直接指定测试集文件
    parser.add_argument('--test',
                        type=str, required=True,
                        help='测试数据集CSV文件的路径')
    parser.add_argument('--codon_freq_file',
                        type=str, required=True,
                        help='密码子频率CSV文件路径')
    parser.add_argument('--output', type=str,
                        required=True,
                        help='输出CSV文件路径')
    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载并处理数据
    print(f"加载测试数据: {args.test}")
    # 直接读取测试数据，不进行 split
    test_data = pd.read_csv(args.test, sep=",")

    # 重置索引 (保险起见)
    test_data = test_data.reset_index(drop=True)
    print(f"测试数据形状: {test_data.shape}")

    # 初始化分词器
    cds_tokenizer = RnaTokenizer.from_pretrained(
        "multimolecule/mrnafm")
    protein_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t33_650M_UR50D")

    # 处理数据的函数
    def process_data(data):
        processed_samples = []
        for _, row in data.iterrows():
            cds_encoding = cds_tokenizer(
                row['cds_sequence'],
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            protein_encoding = protein_tokenizer(
                row['protein_sequence'],
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            processed_sample = {
                'cds_input_ids': cds_encoding[
                    'input_ids'].squeeze(),
                'cds_attention_mask':
                    cds_encoding[
                        'attention_mask'].squeeze(),
                'protein_input_ids':
                    protein_encoding[
                        'input_ids'].squeeze(),
                'protein_attention_mask':
                    protein_encoding[
                        'attention_mask'].squeeze()
            }
            processed_samples.append(
                processed_sample)
        return processed_samples

    # 处理测试数据
    processed_test = process_data(test_data)
    test_dataset = CustomDataset(processed_test)

    # 使用正确的方式加载模型
    print(f"加载模型: {args.model_path}")
    config = AutoConfig.from_pretrained(
        "facebook/esm2_t33_650M_UR50D")
    model = CustomPlantRNAModelmlm(config).to(device)

    model_file_path = os.path.join(args.model_path, 'model.safetensors')
    if not os.path.exists(model_file_path):
        # 尝试不带 model.safetensors 后缀的情况，或者直接是文件路径
        if os.path.isfile(args.model_path):
            model_file_path = args.model_path
        else:
            # 兼容之前的逻辑，假设传入的是目录
            model_file_path = f'{args.model_path}/model.safetensors'

    print(f"正在从 {model_file_path} 加载权重")
    model.load_state_dict(load_file(model_file_path))
    model.eval()

    # 使模型连续化
    def make_model_contiguous(model):
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        for name, buffer in model.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()
        return model

    model = make_model_contiguous(model)

    # 初始化MLM数据整理器
    data_collator = MLMDataCollator(cds_tokenizer)

    # 加载密码子频率文件
    print(
        f"加载密码子频率文件: {args.codon_freq_file}")
    best_codons = load_codon_frequency(
        args.codon_freq_file)
    print("最高频率密码子:", best_codons)

    # 在测试集上评估模型
    print("在测试集上评估模型...")
    model_aa_accuracy, bfc_aa_accuracy = evaluate_model_on_test(
        model, test_dataset, data_collator,
        device, cds_tokenizer, best_codons
    )

    # 创建结果DataFrame并保存为CSV
    results = []
    for aa in sorted(model_aa_accuracy.keys()):
        if aa == 'overall':  # 放到最后处理
            continue
        results.append({
            'Category': aa,
            'BFC': round(bfc_aa_accuracy[aa], 4),
            'Our model': round(
                model_aa_accuracy[aa], 4)
        })

    # 添加总体准确率
    results.append({
        'Category': 'overall',
        'BFC': round(bfc_aa_accuracy['overall'],
                     4),
        'Our model': round(
            model_aa_accuracy['overall'], 4)
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    print(f"结果已保存到: {args.output}")
    print("完成!")


if __name__ == "__main__":
    main()