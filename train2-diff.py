import os
import torch
import multimolecule
from multimolecule import RnaTokenizer, RnaFmModel
import pandas as pd
from sklearn.model_selection import \
    train_test_split, KFold
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score, \
    matthews_corrcoef, roc_auc_score, \
    confusion_matrix
from transformers import EarlyStoppingCallback
import numpy as np
from scipy.special import softmax
import argparse

from model2 import CustomPlantRNAModel
from utils import compute_metrics, CustomDataset, default_data_collator
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Create model for this fold

def create_optimizer(model, training_args):
    # Separate parameters into two groups
    pretrained_params = []
    custom_params = []

    # Add parameters from pre-trained models to first group
    pretrained_params.extend(
        model.plantrna.parameters())
    pretrained_params.extend(
        model.esm2.parameters())

    # Get all parameter names for custom layers
    custom_param_names = []

    for name, param in model.named_parameters():
        if not name.startswith(
                'plantrna.') and not name.startswith(
                'esm2.'):
            custom_param_names.append(name)
            custom_params.append(param)

    # Print parameter groups for verification
    print(
        f"Number of parameters in pre-trained models: {len(pretrained_params)}")
    print(
        f"Number of parameters in custom layers: {len(custom_params)}")
    print(
        f"Custom layer parameters: {custom_param_names}")

    # Create parameter groups with different learning rates and weight decay
    optimizer_grouped_parameters = [
        {
            "params": pretrained_params,
            "lr": 1e-5,
            # Lower learning rate for pre-trained models
            "weight_decay": 0.0,
            # Minimal weight decay for pre-trained models
        },
        {
            "params": custom_params,
            "lr": 0.001,
            # Higher learning rate for custom layers
            "weight_decay": 0.01,
            # More weight decay for custom layers
        },
    ]

    # Create the optimizer with the parameter groups
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters)
    return optimizer


def main():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    # Add argument parser for output directory
    parser = argparse.ArgumentParser(
        description='Train and evaluate binary classification model with 5-fold cross-validation')
    parser.add_argument('--output_dir', type=str,
                        default='./output',
                        help='Directory to store all output files and models')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create logs directory inside output directory
    logs_dir = os.path.join(args.output_dir,
                            'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Load and process data
    print("Loading data...")
    df = pd.read_csv('./xiaomai/wheat-data-checked.tsv',sep=",")

    # Ensure 'label' column exists
    if 'label' not in df.columns:
        if 'Label' in df.columns:  # Try case-insensitive matching
            df['label'] = df['Label']
        else:
            raise ValueError(
                "Required column 'label' not found in dataset")

    # Add protein_sequence if not present
    if 'protein_sequence' not in df.columns and 'translated_protein' in df.columns:
        df['protein_sequence'] = df[
            'translated_protein']

    # Split data into train and test sets
    train_data, test_data = train_test_split(df,test_size=0.2,random_state=42,stratify=df['label'])

    # Reset indices
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    # 筛选短序列样本（长度<=1024）
    test_data['cds_length'] = test_data['cds_sequence'].apply(len)
    short_test_data = test_data[
        test_data['cds_length'] <= 1024].copy()
    print(f"\n短序列测试集样本数: {len(short_test_data)} (占总测试集 {len(test_data)} 的 {len(short_test_data) / len(test_data):.2%})")
    # Initialize tokenizers
    cds_tokenizer = RnaTokenizer.from_pretrained(
        "multimolecule/mrnafm")
    protein_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t33_650M_UR50D")

    # Create K-Fold splitter
    kf = KFold(n_splits=5, shuffle=True,random_state=100)

    # Create configuration for model
    config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
    #config.num_labels = 1  # Binary classification

    # Store models and their predictions
    models = []
    test_predictions = []
    short_test_predictions = []  # 存储短序列预测结果
    fusion_parameters_per_fold = []  # To store alpha, beta for each fold

    # Process the test data once for all folds
    processed_test = []
    for _, row in test_data.iterrows():
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
            'protein_attention_mask':protein_encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(row['label'],dtype=torch.float)}

        processed_test.append(processed_sample)
    test_dataset = CustomDataset(processed_test)
    # 处理短序列测试集
    processed_short_test = []
    for _, row in short_test_data.iterrows():
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
            'cds_attention_mask': cds_encoding[
                'attention_mask'].squeeze(),
            'protein_input_ids': protein_encoding[
                'input_ids'].squeeze(),
            'protein_attention_mask':
                protein_encoding[
                    'attention_mask'].squeeze(),
            'labels': torch.tensor(row['label'],
                                   dtype=torch.float)
        }

        processed_short_test.append(
            processed_sample)

    short_test_dataset = CustomDataset(
        processed_short_test)
    # Training each fold
    for fold, (train_idx, val_idx) in enumerate(
            kf.split(train_data)):
        print(
            f"\n===== Training Fold {fold + 1}/5 =====")

        fold_train_data = train_data.iloc[
            train_idx]
        fold_val_data = train_data.iloc[val_idx]

        # Process train set for this fold
        processed_train = []
        for _, row in fold_train_data.iterrows():
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
                        'attention_mask'].squeeze(),
                'labels': torch.tensor(
                    row['label'],
                    dtype=torch.float)
            }

            processed_train.append(
                processed_sample)

        # Process validation set for this fold
        processed_val = []
        for _, row in fold_val_data.iterrows():
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
                        'attention_mask'].squeeze(),
                'labels': torch.tensor(
                    row['label'],
                    dtype=torch.float)
            }

            processed_val.append(processed_sample)

        # Create datasets for this fold
        train_dataset = CustomDataset(
            processed_train)
        val_dataset = CustomDataset(processed_val)


        model = CustomPlantRNAModel(config).to(device)

        # Make model contiguous
        def make_model_contiguous(model):
            for name, param in model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            for name, buffer in model.named_buffers():
                if not buffer.is_contiguous():
                    buffer.data = buffer.data.contiguous()
            return model

        model = make_model_contiguous(model)

        # Define paths for this fold
        fold_output_dir = os.path.join(
            args.output_dir,
            f'results-fold-{fold + 1}')

        # Training arguments
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            evaluation_strategy="epoch",
            save_strategy='epoch',
            save_total_limit=1,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=20,
            # Reduced from 50 for faster execution
            weight_decay=0,
            logging_dir=logs_dir,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True

        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
            optimizers = (create_optimizer(model, training_args),None)
            # Early stopping
        )

        # Start training
        print(
            f"Starting training for fold {fold + 1}...")
        trainer.train()

        # Save model
        model_save_path = os.path.join(
            args.output_dir,
            f"classification-model-fold-{fold + 1}")
        trainer.save_model(model_save_path)

        # Get and print fusion parameters (alpha, beta)
        fusion_params = model.get_learned_parameters()
        print(
            f"\nFold {fold + 1} Fusion Parameters:")
        print(f"RNA权重: {fusion_params['alpha']:.6f}")
        print(f"蛋白质权重: {fusion_params['beta']:.6f}")


        # Store fusion parameters for this fold
        fusion_params['fold'] = fold + 1
        fusion_parameters_per_fold.append(
            fusion_params)

        # Store the trained model
        models.append(model)

        # Get predictions on test set
        print(
            f"Generating predictions for fold {fold + 1}...")
        predictions = trainer.predict(
            test_dataset)
        pred_probs = 1 / (1 + np.exp(-predictions.predictions.flatten()))  # sigmoid
        test_predictions.append(pred_probs)
        # 短序列测试集预测
        short_pred = trainer.predict(
            short_test_dataset)
        short_pred_probs = 1 / (1 + np.exp(
            -short_pred.predictions.flatten()))
        short_test_predictions.append(
            short_pred_probs)

        fold_metrics = predictions.metrics
        fold_metrics_short = short_pred.metrics

        print(f"Fold {fold + 1} Results:")
        for metric, value in fold_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"Short-Fold {fold + 1} Results:")
        for metric, value in fold_metrics_short.items():
            print(f"  {metric}: {value:.4f}")

    # Save fusion parameters for all folds to a CSV file
    fusion_params_df = pd.DataFrame(
        fusion_parameters_per_fold)
    fusion_params_path = os.path.join(
        args.output_dir, "fusion_parameters.csv")
    fusion_params_df.to_csv(fusion_params_path,
                            index=False)
    print(
        f"\nFusion parameters saved to {fusion_params_path}")

    # Calculate average fusion parameters
    avg_alpha = np.mean(
        [params['alpha'] for params in
         fusion_parameters_per_fold])
    avg_beta = np.mean(
        [params['beta'] for params in
         fusion_parameters_per_fold])


    print(
        f"\nAverage Fusion Parameters Across All Folds:")
    print(f"  Alpha: {avg_alpha:.6f}")
    print(f"  Beta: {avg_beta:.6f}")


    # Ensemble voting (average probabilities)
    test_predictions = np.array(test_predictions)
    ensemble_probs = np.mean(test_predictions,
                             axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    true_labels = np.array([sample['labels'].item() for sample in processed_test])

    # Calculate final metrics
    accuracy = accuracy_score(true_labels,
                              ensemble_preds)
    precision = precision_score(true_labels,
                                ensemble_preds,
                                zero_division=0)
    recall = recall_score(true_labels,
                          ensemble_preds,
                          zero_division=0)
    f1 = f1_score(true_labels, ensemble_preds,
                  zero_division=0)
    mcc = matthews_corrcoef(true_labels,
                            ensemble_preds)
    auc = roc_auc_score(true_labels,
                        ensemble_probs)

    print(
        "\n===== Ensemble Model Results on Test Set =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {auc:.4f}")

    # Print confusion matrix
    cm = confusion_matrix(true_labels,
                          ensemble_preds)
    print(f"Confusion Matrix:\n{cm}")
    metrics_summary = {}
    # 处理短序列测试结果
    if len(short_test_predictions) > 0:
        short_test_predictions = np.array(
            short_test_predictions)
        short_ensemble_probs = np.mean(
            short_test_predictions, axis=0)
        short_ensemble_preds = (
                    short_ensemble_probs >= 0.5).astype(
            int)
        short_true_labels = np.array(
            [sample['labels'].item() for sample in
             processed_short_test])

        # 计算短序列指标
        short_accuracy = accuracy_score(
            short_true_labels,
            short_ensemble_preds)
        short_precision = precision_score(
            short_true_labels,
            short_ensemble_preds, zero_division=0)
        short_recall = recall_score(
            short_true_labels,
            short_ensemble_preds, zero_division=0)
        short_f1 = f1_score(short_true_labels,
                            short_ensemble_preds,
                            zero_division=0)
        short_mcc = matthews_corrcoef(
            short_true_labels,
            short_ensemble_preds)
        short_auc = roc_auc_score(
            short_true_labels,
            short_ensemble_probs)

        print(
            "\n===== 短序列测试集结果 (CDS长度<=1022) =====")
        print(
            f"样本数量: {len(short_true_labels)}")
        print(f"Accuracy: {short_accuracy:.4f}")
        print(f"Precision: {short_precision:.4f}")
        print(f"Recall: {short_recall:.4f}")
        print(f"F1 Score: {short_f1:.4f}")
        print(f"MCC: {short_mcc:.4f}")
        print(f"AUC: {short_auc:.4f}")
        cm_short = confusion_matrix(short_true_labels,
                              short_ensemble_preds)
        print(f"Confusion Matrix:\n{cm_short}")
        # 保存短序列结果
        short_results_df = pd.DataFrame({
            'id': short_test_data['id'].values,
            'cds_length': short_test_data[
                'cds_length'].values,
            'true_label': short_true_labels,
            'ensemble_probability': short_ensemble_probs,
            'ensemble_prediction': short_ensemble_preds
        })

        short_results_path = os.path.join(
            args.output_dir,
            "short_ensemble_results.csv")
        short_results_df.to_csv(
            short_results_path, index=False)
        print(
            f"\n短序列测试结果已保存至: {short_results_path}")

        # 更新指标汇总
        metrics_summary.update({
            'short_accuracy': short_accuracy,
            'short_precision': short_precision,
            'short_recall': short_recall,
            'short_f1': short_f1,
            'short_mcc': short_mcc,
            'short_auc': short_auc
        })
    # Save test results
    results_df = pd.DataFrame({
        'id': test_data['id'].values,
        'true_label': true_labels,
        'ensemble_probability': ensemble_probs,
        'ensemble_prediction': ensemble_preds
    })

    # Save results to the specified output directory
    results_file_path = os.path.join(
        args.output_dir,
        "ensemble_classification_results.csv")
    results_df.to_csv(results_file_path,
                      index=False)

    # Save metrics summary
    metrics_summary.update({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auc': auc,
        'avg_alpha': avg_alpha,
        'avg_beta': avg_beta
    })


    metrics_df = pd.DataFrame([metrics_summary])
    metrics_file_path = os.path.join(
        args.output_dir,
        "final_metrics_summary.csv")
    metrics_df.to_csv(metrics_file_path,
                      index=False)

    print(
        f"All results saved to {args.output_dir}")
    print("Complete!")


if __name__ == "__main__":
    main()
