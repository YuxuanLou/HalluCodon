#!/usr/bin/env python3
"""CodonEXP 单次训练（不五折）: 测试集当验证集, 每组织保留 1 个权重。
用法: python CodonEXP_train_single_tissue.py --dataset_path test-FLOWER-0.9.csv --output_dir ../results/FLOWER
"""
import os
import torch
import multimolecule
from multimolecule import RnaTokenizer, RnaFmModel
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from transformers import EarlyStoppingCallback, TrainerCallback
import numpy as np
import argparse

from models.CodonEXP import CustomPlantRNAModel
from utils import compute_metrics, CustomDataset, default_data_collator


def create_optimizer(model, training_args):
    """不冻结 ESM2: plantrna + esm2 都在 pretrained_params 组 (lr 1e-5)。"""
    pretrained_params = []
    custom_params = []
    pretrained_params.extend(model.plantrna.parameters())
    pretrained_params.extend(model.esm2.parameters())
    custom_params = []
    for name, param in model.named_parameters():
        if not name.startswith('plantrna.') and not name.startswith('esm2.'):
            custom_params.append(param)
    optimizer_grouped_parameters = [
        {
            "params": pretrained_params,
            "lr": 1e-5,
            "weight_decay": 0.0,
        },
        {
            "params": custom_params,
            "lr": 0.001,
            "weight_decay": 0.01,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer


def process_data(data, cds_tokenizer, protein_tokenizer):
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
            'protein_attention_mask': protein_encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(row['label'], dtype=torch.float)
        }
        processed_samples.append(processed_sample)
    return processed_samples


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logs_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    print("Loading data...")
    df = pd.read_csv(args.dataset_path, sep=",")

    # 数据划分: 80% 训练 / 20% 测试(=验证), 不分折
    train_data, val_data = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label'])
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    print(f"Train: {train_data.shape[0]}, Val(=test): {val_data.shape[0]}")
    print(f"  Train high={sum(train_data['label']==1)}, low={sum(train_data['label']==0)}")
    print(f"  Val   high={sum(val_data['label']==1)}, low={sum(val_data['label']==0)}")

    cds_tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")

    processed_train = process_data(train_data, cds_tokenizer, protein_tokenizer)
    processed_val = process_data(val_data, cds_tokenizer, protein_tokenizer)
    train_dataset = CustomDataset(processed_train)
    val_dataset = CustomDataset(processed_val)

    model = CustomPlantRNAModel(config).to(device)

    def make_model_contiguous(model):
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        for name, buffer in model.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()
        return model

    model = make_model_contiguous(model)

    fold_output_dir = os.path.join(args.output_dir, 'model-results')
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        evaluation_strategy="epoch",
        save_strategy='no',          # 不写中间 checkpoint (避免 10G optimizer.pt)
        save_total_limit=1,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=20,
        weight_decay=0,
        logging_dir=logs_dir,
        logging_steps=100,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # 在验证集 f1 创新高时覆盖保存最优权重 (纯模型, 只留 1 份)
    model_save_path = os.path.join(args.output_dir, "classification-model")
    best_f1 = -1.0

    class SaveBestCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            nonlocal best_f1
            if metrics is not None and 'eval_f1' in metrics and metrics['eval_f1'] > best_f1:
                best_f1 = metrics['eval_f1']
                trainer.save_model(model_save_path)
                print(f"  [save] epoch {state.epoch} f1={best_f1:.4f} 创新高, 权重已保存")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20), SaveBestCallback()],
        optimizers=(create_optimizer(model, training_args), None)
    )

    print(f"Starting training (single, no CV)...")
    trainer.train()

    if best_f1 < 0:
        raise RuntimeError("训练完成但从未触发最优权重保存 (eval_f1 始终未评估?), 请检查")

    # 融合参数
    fusion_params = model.get_learned_parameters()
    print(f"\nFusion Parameters: alpha(RNA)={fusion_params['alpha']:.6f}, beta(Protein)={fusion_params['beta']:.6f}")

    # 验证集(=测试集)评估
    print("Evaluating on val(=test) set...")
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nModel saved to {model_save_path}")


if __name__ == '__main__':
    main()
