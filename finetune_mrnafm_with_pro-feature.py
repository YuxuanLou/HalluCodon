import os
import torch
import multimolecule
from multimolecule import RnaTokenizer, RnaFmModel
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from transformers import EarlyStoppingCallback
import numpy as np
import argparse
from collections import defaultdict

# 导入已经实现的模型和数据整理器
from mrnafm_pro_mlm import CustomPlantRNAModelmlm
from utils import CustomDataset, MLMDataCollator, compute_mlm_metrics
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from torch.utils.data import DataLoader, Dataset


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


# 在测试集上评估模型
def evaluate_model_on_test(model, test_dataset, data_collator, device, cds_tokenizer):
    """在测试集上评估模型性能"""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # 按氨基酸分类的准确率统计
    aa_correct = defaultdict(int)
    aa_total = defaultdict(int)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        collate_fn=data_collator,
        shuffle=False
    )
    with torch.no_grad():
        for batch in test_dataloader:  # 直接遍历 DataLoader 生成的批次
            # 将数据移到设备上
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(**batch)
            loss = outputs["loss"]
            total_loss += loss.item()

            # 获取预测结果
            predictions = torch.argmax(outputs["logits"], dim=-1)
            mlm_labels = batch['labels']

            # 只考虑非填充和非特殊标记的位置
            mask = mlm_labels != -100

            # 计算正确预测数
            correct = (predictions[mask] == mlm_labels[mask])
            correct_predictions += correct.sum().item()
            total_predictions += mask.sum().item()

            # 统计每个氨基酸的准确率
            masked_indices = mask.nonzero(as_tuple=True)  # 获取被 mask 的位置
            for pos_idx, seq_idx in zip(*masked_indices):
                true_token_id = mlm_labels[pos_idx, seq_idx].item()
                pred_token_id = predictions[pos_idx, seq_idx].item()

                # 转换为密码子
                true_codon = cds_tokenizer.convert_ids_to_tokens([true_token_id])[0]
                pred_codon = cds_tokenizer.convert_ids_to_tokens([pred_token_id])[0]

                # 翻译为氨基酸
                true_aa = translate_codon_to_aa(true_codon)

                # 更新统计
                aa_total[true_aa] += 1
                if true_token_id == pred_token_id:
                    aa_correct[true_aa] += 1

    # 计算平均损失和准确率
    avg_loss = total_loss / (len(test_dataset) // 2 + (1 if len(test_dataset) % 2 != 0 else 0))
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # 计算每个氨基酸的准确率
    aa_accuracy = {aa: aa_correct[aa] / aa_total[aa] if aa_total[aa] > 0 else 0 for aa in aa_total}

    return {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'aa_accuracy': aa_accuracy
    }


def create_optimizer(model, training_args):
    # 分离参数为两组(仅训练参数)
    pretrained_params = []
    custom_params = []

    # 获取所有可训练参数
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}

    # 分为预训练模型(plantrna + esm2)和自定义层
    for name, param in trainable_params.items():
        if name.startswith('plantrna.'):
            pretrained_params.append(param)
        elif name.startswith('esm2.'):
            pretrained_params.append(param)
        else:
            custom_params.append(param)

    # 打印参数组进行验证
    print(f"预训练模型中可训练参数数量: {len(pretrained_params)}")
    print(f"自定义层中可训练参数数量: {len(custom_params)}")
    print(f"可训练的自定义层参数: {[name for name in trainable_params if not name.startswith(('plantrna.', 'esm2.'))]}")

    # 创建具有不同学习率和权重衰减的参数组
    optimizer_grouped_parameters = [
        {
            "params": pretrained_params,
            "lr": 1e-4,  # 预训练模型较低的学习率
            "weight_decay": 0.01,  # 预训练模型最小的权重衰减
        },
        {
            "params": custom_params,
            "lr": 1e-4,  # 自定义层较高的学习率
            "weight_decay": 0.01,  # 自定义层较多的权重衰减
        },
    ]

    # 创建带参数组的优化器
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 添加参数解析器来获取输出目录和数据集路径
    parser = argparse.ArgumentParser(description='训练MLM模型')
    parser.add_argument('--output_dir', type=str, default='./output', help='存储所有输出文件和模型的目录')
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集CSV文件的路径')
    parser.add_argument('--model_name', type=str, default='finetune-model', help='保存的模型名称')
    args = parser.parse_args()

    # 如果输出目录不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 在输出目录中创建日志目录
    logs_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # 加载并处理数据
    print("加载数据...")
    df = pd.read_csv(args.dataset_path, sep=",")

    # 将数据集分为训练集、验证集和测试集
    # 首先分出测试集
    train_val_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
    # 然后分出验证集
    train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)

    # 重置索引
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    print(f"训练数据形状: {train_data.shape}")
    print(f"验证数据形状: {val_data.shape}")
    print(f"测试数据形状: {test_data.shape}")

    # 初始化分词器
    cds_tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
    protein_tokenizer = AutoTokenizer.from_pretrained("")

    # 创建模型配置
    config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")

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
                'cds_input_ids': cds_encoding['input_ids'].squeeze(),
                'cds_attention_mask': cds_encoding['attention_mask'].squeeze(),
                'protein_input_ids': protein_encoding['input_ids'].squeeze(),
                'protein_attention_mask': protein_encoding['attention_mask'].squeeze()
            }
            processed_samples.append(processed_sample)
        return processed_samples

    # 处理训练、验证和测试数据
    processed_train = process_data(train_data)
    processed_val = process_data(val_data)
    processed_test = process_data(test_data)

    # 创建数据集
    train_dataset = CustomDataset(processed_train)
    val_dataset = CustomDataset(processed_val)
    test_dataset = CustomDataset(processed_test)

    # 创建模型
    model = CustomPlantRNAModelmlm(config).to(device)

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

    # 定义输出路径
    model_output_dir = os.path.join(args.output_dir, 'model-results')

    # 训练参数
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        save_total_limit=1,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=50,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mask_accuracy",
        greater_is_better=True
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        compute_metrics=compute_mlm_metrics,
        optimizers=(create_optimizer(model, training_args), None)
    )

    # 开始训练
    print("开始MLM训练...")
    trainer.train()

    # 获取数据集文件名（不含路径和扩展名）
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    
    # 保存模型
    model_save_path = os.path.join(args.output_dir, f"{dataset_name}-{args.model_name}")
    trainer.save_model(model_save_path)
    print(f"MLM模型已保存到 {model_save_path}")

    # 在测试集上评估模型
    print("在测试集上评估模型...")
    test_results = evaluate_model_on_test(model, test_dataset, data_collator, device, cds_tokenizer)

    # 打印测试结果
    print(f"测试损失: {test_results['test_loss']:.4f}")
    print(f"测试准确率: {test_results['test_accuracy']:.4f}")
    print("各氨基酸对应密码子的准确率:")

    # 按照氨基酸字母顺序排序输出结果
    for aa in sorted(test_results['aa_accuracy'].keys()):
        print(f"{aa}: {test_results['aa_accuracy'][aa]:.4f}")

    # 将结果保存到文件
    results_file = os.path.join(args.output_dir, f"{dataset_name}_test_results.txt")
    with open(results_file, "w") as f:
        f.write(f"测试损失: {test_results['test_loss']:.4f}\n")
        f.write(f"测试准确率: {test_results['test_accuracy']:.4f}\n")
        f.write("各氨基酸对应密码子的准确率:\n")
        for aa in sorted(test_results['aa_accuracy'].keys()):
            f.write(f"{aa}: {test_results['aa_accuracy'][aa]:.4f}\n")

    print(f"测试结果已保存到 {results_file}")
    print("完成!")


if __name__ == "__main__":
    main()

