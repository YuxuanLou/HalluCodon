import os
import torch
import multimolecule
from multimolecule import RnaTokenizer
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, default_data_collator
import numpy as np
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from models.CodonNAT import CustomPlantRNAModelmlm
from utils import CustomDataset

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

        probability_matrix = torch.full(inputs.shape, self.mlm_probability,device=device)
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for special_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
            if special_id is not None:
                special_tokens_mask = special_tokens_mask | (inputs == special_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(~batch["cds_attention_mask"].bool(), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.mask_token_id

        batch["cds_input_ids"] = inputs
        batch["labels"] = labels
        batch["masked_indices"] = masked_indices
        batch["mask_token_positions"] = masked_indices

        return batch

def translate_codon_to_aa(codon):
    codon_table = {
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
        'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    if 'N' in codon or codon not in codon_table:
        return 'X'

    return codon_table.get(codon, 'X')

def load_codon_frequency(codon_freq_file):
    df = pd.read_csv(codon_freq_file)

    best_codons = {}
    for aa in df['aa'].unique():
        aa_df = df[df['aa'] == aa]
        best_codon = aa_df.loc[
            aa_df['frequency(%)'].idxmax(), 'codon']
        best_codons[aa] = best_codon

    return best_codons

def evaluate_model_on_test(model, test_dataset,
                           data_collator, device,
                           cds_tokenizer,
                           best_codons):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model.eval()
    model_aa_correct = defaultdict(int)
    model_aa_total = defaultdict(int)
    bfc_aa_correct = defaultdict(int)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        collate_fn=data_collator,
        shuffle=False
    )

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs["logits"], dim=-1)
            mlm_labels = batch['labels']
            mask = mlm_labels != -100
            masked_indices = mask.nonzero(as_tuple=True) 
            for pos_idx, seq_idx in zip(*masked_indices):
                true_token_id = mlm_labels[pos_idx, seq_idx].item()
                pred_token_id = predictions[pos_idx, seq_idx].item()
                true_codon = cds_tokenizer.convert_ids_to_tokens([true_token_id])[0]
                pred_codon = cds_tokenizer.convert_ids_to_tokens([pred_token_id])[0]
                true_aa = translate_codon_to_aa(true_codon)

                if true_aa == 'X' or true_aa == '*':
                    continue

                model_aa_total[true_aa] += 1
                if true_token_id == pred_token_id:
                    model_aa_correct[true_aa] += 1
                if true_aa in best_codons and true_codon == best_codons[true_aa]:
                    bfc_aa_correct[true_aa] += 1


    model_aa_accuracy = {
        aa: model_aa_correct[aa] / model_aa_total[aa] if model_aa_total[aa] > 0 else 0
        for aa in model_aa_total}
    bfc_aa_accuracy = {
        aa: bfc_aa_correct[aa] / model_aa_total[aa] if model_aa_total[aa] > 0 else 0
        for aa in model_aa_total}

    model_total_correct = sum(model_aa_correct.values())
    bfc_total_correct = sum(bfc_aa_correct.values())
    total_predictions = sum(model_aa_total.values())

    model_overall_accuracy = model_total_correct / total_predictions if total_predictions > 0 else 0
    bfc_overall_accuracy = bfc_total_correct / total_predictions if total_predictions > 0 else 0

    model_aa_accuracy['overall'] = model_overall_accuracy
    bfc_aa_accuracy['overall'] = bfc_overall_accuracy

    return model_aa_accuracy, bfc_aa_accuracy


def main():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--codon_freq_file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_data = pd.read_csv(args.test, sep=",")

    test_data = test_data.reset_index(drop=True)

    cds_tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

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

    processed_test = process_data(test_data)
    test_dataset = CustomDataset(processed_test)

    config = AutoConfig.from_pretrained(
        "facebook/esm2_t33_650M_UR50D")
    model = CustomPlantRNAModelmlm(config).to(device)

    model_file_path = os.path.join(args.model_path, 'model.safetensors')
    model_file_path = f'{args.model_path}/model.safetensors'
    model.load_state_dict(load_file(model_file_path))
    model.eval()

    def make_model_contiguous(model):
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        for name, buffer in model.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()
        return model

    model = make_model_contiguous(model)
    data_collator = MLMDataCollator(cds_tokenizer)
    best_codons = load_codon_frequency(args.codon_freq_file)

    model_aa_accuracy, bfc_aa_accuracy = evaluate_model_on_test(
        model, test_dataset, data_collator,
        device, cds_tokenizer, best_codons
    )

    results = []
    for aa in sorted(model_aa_accuracy.keys()):
        if aa == 'overall': 
            continue
        results.append({
            'Category': aa,
            'BFC': round(bfc_aa_accuracy[aa], 4),
            'Our model': round(model_aa_accuracy[aa], 4)
        })


    results.append({
        'Category': 'overall',
        'BFC': round(bfc_aa_accuracy['overall'],4),
        'Our model': round(model_aa_accuracy['overall'], 4)
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
