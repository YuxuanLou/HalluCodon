import math
import argparse
import re
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import multimolecule
from multimolecule import RnaTokenizer
import numpy as np
import random
import pandas as pd
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoConfig, default_data_collator
from safetensors.torch import load_file
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
from models.CodonEXP_for_inference import CustomPlantRNAModel
from models.CodonNAT_for_inference import CustomPlantRNAModelmlm
from utils import CustomDataset


class MLMDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15, seed=42):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.seed = seed
        self.mask_cache = {}
        self.random_cache = {}
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __call__(self, examples):
        batch = default_data_collator(examples)
        device = batch["cds_input_ids"].device
        inputs = batch["cds_input_ids"].clone()
        labels = inputs.clone()
        batch_size, seq_length = inputs.shape
        if seq_length not in self.mask_cache:
            self._create_mask_pattern(seq_length,device)
        special_tokens_mask = torch.zeros((batch_size, seq_length),dtype=torch.bool, device=device)
        for special_id in [self.pad_token_id,self.bos_token_id,self.eos_token_id]:
            if special_id is not None:
                special_tokens_mask = special_tokens_mask | (inputs == special_id)
        attention_mask = batch["cds_attention_mask"].bool()
        masked_indices = self.mask_cache[seq_length].clone().to(device).unsqueeze(0).expand(batch_size, -1)
        masked_indices = masked_indices & (~special_tokens_mask) & attention_mask
        labels[~masked_indices] = -100
        indices_mask = self.random_cache[seq_length]["mask"].clone().to(device).unsqueeze(0).expand(batch_size,-1) & masked_indices
        inputs[indices_mask] = self.mask_token_id
        indices_random = self.random_cache[seq_length]["random"].clone().to(device).unsqueeze(0).expand(batch_size,-1) & masked_indices & ~indices_mask
        random_words = self.random_cache[seq_length]["words"].clone().to(device).unsqueeze(0).expand(batch_size, -1)
        inputs[indices_random] = random_words[indices_random]
        batch["cds_input_ids"] = inputs
        batch["labels"] = labels
        return batch

    def _create_mask_pattern(self, seq_length,device):
        torch.manual_seed(self.seed)
        probability_matrix = torch.full((seq_length,), self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix,generator=self.generator).bool()
        mask_prob_matrix = torch.full((seq_length,), 0.8)
        indices_mask = torch.bernoulli(mask_prob_matrix,generator=self.generator).bool() & masked_indices
        random_prob_matrix = torch.full((seq_length,), 0.5)
        indices_random = torch.bernoulli(random_prob_matrix,generator=self.generator).bool() & masked_indices & ~indices_mask
        random_words = torch.randint(self.vocab_size, (seq_length,),generator=self.generator)
        self.mask_cache[seq_length] = masked_indices.cpu()
        self.random_cache[seq_length] = {"mask": indices_mask.cpu(),"random": indices_random.cpu(),"words": random_words.cpu()}


# Codon tables
codon_table = {'AUA': 'I', 'AUC': 'I', 'AUU': 'I',
               'AUG': 'M', 'ACA': 'T', 'ACC': 'T',
               'ACG': 'T', 'ACU': 'T', 'AAC': 'N',
               'AAU': 'N', 'AAA': 'K', 'AAG': 'K',
               'AGC': 'S', 'AGU': 'S', 'AGA': 'R',
               'AGG': 'R', 'CUA': 'L', 'CUC': 'L',
               'CUG': 'L', 'CUU': 'L', 'CCA': 'P',
               'CCC': 'P', 'CCG': 'P', 'CCU': 'P',
               'CAC': 'H', 'CAU': 'H', 'CAA': 'Q',
               'CAG': 'Q', 'CGA': 'R', 'CGC': 'R',
               'CGG': 'R', 'CGU': 'R', 'GUA': 'V',
               'GUC': 'V', 'GUG': 'V', 'GUU': 'V',
               'GCA': 'A', 'GCC': 'A', 'GCG': 'A',
               'GCU': 'A', 'GAC': 'D', 'GAU': 'D',
               'GAA': 'E', 'GAG': 'E', 'GGA': 'G',
               'GGC': 'G', 'GGG': 'G', 'GGU': 'G',
               'UCA': 'S', 'UCC': 'S', 'UCG': 'S',
               'UCU': 'S', 'UUC': 'F', 'UUU': 'F',
               'UUA': 'L', 'UUG': 'L', 'UAC': 'Y',
               'UAU': 'Y', 'UAA': '*', 'UAG': '*',
               'UGC': 'C', 'UGU': 'C', 'UGA': '*',
               'UGG': 'W', }
synonymous_codons = {aa: [c for c, a in codon_table.items() if a == aa] for aa in set(codon_table.values())}
def parse_fasta(fasta_file):
    sequences = {}
    current_id = None
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                current_id = line[1:].strip()
                sequences[current_id] = []
            elif line.strip() and current_id:
                sequences[current_id].append(
                    line.strip())
    return {k: ''.join(v) for k, v in sequences.items()}


def parse_codon_frequency_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Codon frequency file not found at path {csv_path}")
        return None

    column_map = {df.columns[0]: 'amino_acid','aa': 'amino_acid', 'codon': 'codon','frequency(%)': 'frequency_percent','count': 'count'}
    df.rename(columns=column_map, inplace=True)
    required_cols = ['amino_acid', 'codon']
    if not all(col in df.columns for col in required_cols):
        print("Error: The CSV file must contain 'aa' and 'codon' columns.")
        return None
    if 'frequency_percent' in df.columns:
        df['frequency'] = df['frequency_percent'] / 100.0
    elif 'count' in df.columns:
        total_counts = df.groupby('amino_acid')['count'].transform('sum')
        df['frequency'] = df['count'] / total_counts
    else:
        print("Error: The CSV file must contain either a 'frequency(%)' or a 'count' column.")
        return None
    df['codon'] = df['codon'].str.upper().str.replace('T', 'U')
    target_dist = {}
    for aa, group in df.groupby('amino_acid'):
        total_freq = group['frequency'].sum()
        if total_freq > 0:
            dist = {row['codon']: row['frequency'] / total_freq for _, row in group.iterrows()}
            target_dist[aa] = dist
    return target_dist


class GradientOptimizer:
    def __init__(self, CodonEXP_model_dir,
                 CodonNAT_model_dir,
                 iterations=16,
                 batch_size=16, top_n_return=5,
                 naturalness_weight=1.0,
                 hallucination_naturalness_weight=1,
                 mutation_rate=0.15, patience=20,
                 use_reversibility_check=True,
                 max_iterations=96,
                 min_expression_threshold=0.9,
                 min_naturality_threshold=0.6,
                 codon_frequency_data=None,
                 codon_frequency_weight=1.0,
                 **kwargs):
        self.model_dir = CodonEXP_model_dir
        self.iterations = iterations
        self.batch_size = batch_size
        self.top_n_return = top_n_return
        self.perplexity_model_dir = CodonNAT_model_dir
        self.perplexity_weight = naturalness_weight
        self.hallucination_perplexity_weight = hallucination_naturalness_weight
        self.mutation_rate = mutation_rate
        self.patience = patience
        self.use_reversibility_check = use_reversibility_check
        self.max_iterations = max_iterations
        self.min_expression_threshold = min_expression_threshold
        self.min_naturality_threshold = min_naturality_threshold
        self.target_dist = codon_frequency_data
        self.codon_frequency_weight = codon_frequency_weight
        self.cds_tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
        self.protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.config.vocab_size = self.cds_tokenizer.vocab_size
        self.config.hidden_size = 1280
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.naturality_model = CustomPlantRNAModelmlm(self.config).to(self.device)
        self.naturality_model.eval()

        self.models = {}
        for fold in range(1, 6):
            model = CustomPlantRNAModel(self.config)
            model.eval()
            self.models[fold] = model

    def _get_all_sequence_codon_counts(self,sequence: str) -> Dict[str, Counter]:
        all_counts = {}
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i + 3]
            aa = codon_table.get(codon)
            if aa:
                if aa not in all_counts:
                    all_counts[aa] = Counter(
                        {syn_codon: 0 for
                         syn_codon in
                         synonymous_codons.get(aa,[])})
                if codon in all_counts[aa]:
                    all_counts[aa][codon] += 1
        return all_counts

    def _normalize_counts(self,counts_dict: Counter) -> Dict[str, float]:
        total_count = sum(counts_dict.values())
        if total_count == 0:
            return {codon: 1.0 / len(counts_dict) if len(counts_dict) > 0 else 0 for codon in counts_dict}
        return {codon: count / total_count for codon, count in counts_dict.items()}

    def _calculate_js_divergence(self, p: Dict,q: Dict,epsilon=1e-10) -> float:
        all_codons = set(p.keys()) | set(q.keys())
        p_vec = np.array([p.get(k, 0) for k in all_codons])
        q_vec = np.array([q.get(k, 0) for k in all_codons])
        p_vec += epsilon
        q_vec += epsilon
        p_vec /= np.sum(p_vec)
        q_vec /= np.sum(q_vec)
        m_vec = 0.5 * (p_vec + q_vec)
        kl_pm = np.sum(p_vec * np.log(p_vec / m_vec))
        kl_qm = np.sum(q_vec * np.log(q_vec / m_vec))
        return 0.5 * (kl_pm + kl_qm)

    def translate_cds(self, cds_sequence):
        if isinstance(cds_sequence, list):
            return [str(Seq(s).translate()).strip('*') for s in cds_sequence]
        return str(Seq(cds_sequence).translate()).strip('*')

    def calculate_naturalness_with_mlm(self,cds_sequences,nat_protein_emb):
        if not cds_sequences: return []
        n_seeds = 10
        seeds = range(42, 42 + n_seeds)
        mlm_collators = [MLMDataCollator(self.cds_tokenizer,seed=seed) for seed in seeds]
        all_scores_by_seed = [[] for _ in range(n_seeds)]
        for i in range(0, len(cds_sequences),self.batch_size):
            batch_sequences = cds_sequences[i:i + self.batch_size]
            cds_encoding = self.cds_tokenizer(
                batch_sequences,
                padding='max_length',
                truncation=True, max_length=1024,
                return_tensors='pt')
            batch_samples = [{'cds_input_ids':cds_encoding['input_ids'][j].squeeze(),'cds_attention_mask':cds_encoding['attention_mask'][j].squeeze(),}
                             for j in range(len(batch_sequences))]

            for seed_idx, mlm_collator in enumerate(mlm_collators):
                batch = mlm_collator(batch_samples)
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                with torch.no_grad():
                    self.naturality_model.to(self.device)
                    outputs = self.naturality_model(cds_input_ids=batch['cds_input_ids'],cds_attention_mask=batch['cds_attention_mask'],
                        pre_computed_protein_embeddings=nat_protein_emb)
                    self.naturality_model.plantrna.to('cpu')
                    logits, labels = outputs["logits"], batch['labels']
                    batch_naturalness_scores = []
                    for b in range(logits.size(0)):
                        sample_logits, sample_labels = logits[b], labels[b]
                        mask_positions = (sample_labels != -100)
                        if mask_positions.sum() == 0:
                            batch_naturalness_scores.append(1.0)
                            continue
                        masked_logits, masked_labels = sample_logits[mask_positions], sample_labels[mask_positions]
                        masked_probs = F.softmax(masked_logits, dim=-1)
                        label_probs = torch.gather(masked_probs, 1,masked_labels.unsqueeze(1)).squeeze(1)
                        label_probs = torch.clamp(label_probs, min=1e-9)
                        log_probs = torch.log(label_probs)
                        avg_log_prob = log_probs.mean().item()
                        naturalness_score = math.exp(avg_log_prob)
                        batch_naturalness_scores.append(naturalness_score)
                    all_scores_by_seed[seed_idx].extend(batch_naturalness_scores)

        avg_naturalness_scores = []
        num_sequences = len(cds_sequences)
        for seq_idx in range(num_sequences):
            scores_for_seq = [all_scores_by_seed[s][seq_idx] for s in range(n_seeds) if seq_idx < len(all_scores_by_seed[s])]
            if scores_for_seq:
                avg_naturalness_scores.append(sum(scores_for_seq) / len(scores_for_seq))
            elif num_sequences > 0:
                avg_naturalness_scores.append(0.0)

        if not avg_naturalness_scores and num_sequences > 0:
            return [0.0] * num_sequences
        return avg_naturalness_scores

    def predict_expression_batch(self,cds_sequences, expr_protein_embs):
        if not cds_sequences: return []
        expression_probs = []
        for i in range(0, len(cds_sequences),self.batch_size):
            batch_seqs = cds_sequences[i:i + self.batch_size]
            cds_enc = self.cds_tokenizer(
                batch_seqs, padding='max_length',
                truncation=True, max_length=1024,
                return_tensors='pt').to(self.device)

            batch_probs = []
            with torch.no_grad():
                for key, model in self.models.items():
                    model.to(self.device)
                    outputs = model(cds_input_ids=cds_enc['input_ids'],
                        cds_attention_mask=cds_enc['attention_mask'],
                        pre_computed_protein_embeddings=expr_protein_embs[model])
                    logits = outputs['logits']
                    probs = torch.sigmoid(logits).cpu().numpy()
                    batch_probs.append(probs)
                    model.plantrna.to('cpu')
            ensemble_probs = np.mean(batch_probs,axis=0).flatten()
            expression_probs.extend(ensemble_probs)
        return expression_probs

    def propose_mutations_with_gradients(self,cds_sequence,nat_protein_emb,expr_protein_embs,check=False):

        codons = [cds_sequence[i:i + 3] for i in
                  range(0, len(cds_sequence), 3)]
        cds_encoding = self.cds_tokenizer(
            cds_sequence, padding='max_length',
            truncation=True, max_length=1024,
            return_tensors='pt').to(self.device)


        all_expr_gains = []
        all_expr_probs = []
        for key, model in self.models.items():
            model.to(self.device)
            model.zero_grad()
            embedding_layer = model.plantrna.get_input_embeddings()
            embed_outputs = [None]
            def save_embed_output_hook(module,input,output):
                output.requires_grad_(True)
                output.retain_grad()
                embed_outputs[0] = output
                return output
            handle = embedding_layer.register_forward_hook(save_embed_output_hook)
            outputs = model(
                cds_input_ids=cds_encoding.input_ids,
                cds_attention_mask=cds_encoding.attention_mask,
                pre_computed_protein_embeddings=expr_protein_embs[model])
            with torch.no_grad():
                prob = torch.sigmoid(outputs['logits']).item()
                all_expr_probs.append(prob)
            outputs['logits'].backward()

            handle.remove()
            embed_output = embed_outputs[0]
            grads = embed_output.grad[0,1:len(codons) + 1, :]
            model_gains = self._calculate_gains_for_model(codons, grads, embedding_layer)
            all_expr_gains.append(model_gains)
            model.plantrna.to('cpu')
        current_expr_prob = sum(all_expr_probs) / len(all_expr_probs) if all_expr_probs else 0.0
        avg_expr_gains = self._average_gains(all_expr_gains)

        with torch.no_grad():
            self.naturality_model.to(self.device)
            outputs_nat = self.naturality_model(
                cds_input_ids=cds_encoding.input_ids,
                cds_attention_mask=cds_encoding.attention_mask,
                pre_computed_protein_embeddings=nat_protein_emb,
                labels=None)
            logits = outputs_nat['logits']
            naturalness_probs = {}
            for i, original_codon in enumerate(codons):
                aa = codon_table.get(original_codon, '*')
                if aa == '*' or len(synonymous_codons.get(aa,[])) <= 1: continue
                pos_logits = logits[0, 1 + i, :]
                pos_probs = F.softmax(pos_logits,dim=-1)
                codon_probs = {syn_codon: pos_probs[self.cds_tokenizer.convert_tokens_to_ids(syn_codon)].item() for
                    syn_codon in synonymous_codons.get(aa, []) if syn_codon != original_codon}
                if codon_probs:
                    naturalness_probs[i] = codon_probs

        all_initial_counts = {}
        initial_jsd_by_aa = {}
        if self.codon_frequency_weight > 0 and self.target_dist:
            all_initial_counts = self._get_all_sequence_codon_counts(cds_sequence)
            for aa, counts in all_initial_counts.items():
                if aa in self.target_dist:
                    dist = self._normalize_counts(counts)
                    initial_jsd_by_aa[aa] = self._calculate_js_divergence(dist,self.target_dist[aa])

        final_proposals = {}
        all_positions = set(
            avg_expr_gains.keys()) | set(
            naturalness_probs.keys())
        for i in all_positions:
            codon_gains = {}
            expr_pos_gains, nat_pos_probs = avg_expr_gains.get(i, {}), naturalness_probs.get(i,{})
            all_syn_codons = set(expr_pos_gains.keys()) | set(nat_pos_probs.keys())
            original_codon = codons[i]
            aa = codon_table.get(original_codon,'*')
            jsd_gain_cache = {}
            if aa in initial_jsd_by_aa:
                target_dist_for_aa = self.target_dist[aa]
                current_counts = all_initial_counts[aa]
                jsd_before = initial_jsd_by_aa[aa]
                aa_count = sum(current_counts.values())
                for syn_codon in synonymous_codons.get(aa, []):
                    if syn_codon == original_codon: continue
                    hypothetical_counts = current_counts.copy()
                    hypothetical_counts[original_codon] -= 1
                    hypothetical_counts[syn_codon] += 1
                    hypothetical_dist = self._normalize_counts(
                        hypothetical_counts)
                    jsd_after = self._calculate_js_divergence(
                        hypothetical_dist,
                        target_dist_for_aa)
                    jsd_gain = jsd_before - jsd_after
                    scaled_jsd_gain = jsd_gain * aa_count
                    jsd_gain_cache[syn_codon] = scaled_jsd_gain

            for codon in all_syn_codons:
                expr_gain, nat_prob = expr_pos_gains.get(codon,0.0), nat_pos_probs.get(codon,0.0)
                expr_gain_threshold = (1 - current_expr_prob) if check else 0.5
                sigmoid_js_gain_threshold = 0.0 if check else 0.5
                if expr_gain > min(expr_gain_threshold, 0.5):
                    js_gain = jsd_gain_cache.get(codon, 0.0)
                    sigmoid_js_score = torch.sigmoid(torch.tensor(js_gain,device=self.device)).item()
                    if sigmoid_js_score > sigmoid_js_gain_threshold:
                        total_gain = expr_gain * (nat_prob ** self.hallucination_perplexity_weight) * (sigmoid_js_score ** self.codon_frequency_weight)
                        if total_gain > 0:
                            codon_gains[codon] = total_gain
            if codon_gains:
                final_proposals[i] = codon_gains
        return final_proposals

    def _calculate_gains_for_model(self, codons,grads,embedding_layer):
        gains = {}
        with torch.no_grad():
            for i, original_codon in enumerate(codons):
                aa = codon_table.get(original_codon, '*')
                if aa == '*' or len(synonymous_codons.get(aa,[])) <= 1: continue
                grad_pos = grads[i]
                original_codon_id = self.cds_tokenizer.convert_tokens_to_ids(original_codon)
                E_i = embedding_layer(torch.tensor(original_codon_id,device=self.device))
                position_gains = {}
                for syn_codon in synonymous_codons.get(aa, []):
                    if syn_codon == original_codon:
                        continue
                    syn_codon_id = self.cds_tokenizer.convert_tokens_to_ids(
                        syn_codon)
                    E_new = embedding_layer(torch.tensor(syn_codon_id,device=self.device))
                    gain = torch.dot(grad_pos, (E_new - E_i))
                    position_gains[syn_codon] = torch.sigmoid(gain).item()
                if position_gains:
                    gains[i] = position_gains
        return gains

    def _average_gains(self, all_gains_list):
        if not all_gains_list: return {}
        avg_gains, num_models = {}, len(
            all_gains_list)
        for model_gains in all_gains_list:
            for pos, codon_gains_dict in model_gains.items():
                if pos not in avg_gains:
                    avg_gains[pos] = {}
                for codon, gain in codon_gains_dict.items():
                    avg_gains[pos][codon] = avg_gains[pos].get(codon,0.0) + gain
        for pos in avg_gains:
            for codon in avg_gains[pos]:
                avg_gains[pos][codon] /= num_models
        return avg_gains

    def _apply_random_synonymous_mutations(self,cds_sequence,num_mutations=1):
        codons = [cds_sequence[i:i + 3] for i in range(0, len(cds_sequence), 3)]
        mutable_indices = [i for i, codon in enumerate(codons) if codon_table.get(codon,'*') != '*' and len(synonymous_codons.get(codon_table.get(codon),[])) > 1]
        if not mutable_indices: return cds_sequence
        indices_to_mutate = random.sample(mutable_indices, min(num_mutations,len(mutable_indices)))
        mutated_codons = list(codons)
        for idx in indices_to_mutate:
            original_codon = mutated_codons[idx]
            aa = codon_table[original_codon]
            syn_options = [c for c in synonymous_codons[aa] if c != original_codon]
            if syn_options:
                mutated_codons[idx] = random.choice(syn_options)
        return "".join(mutated_codons)

    def optimize_with_gradients(self,cds_sequence, results_dir=None):
        original_protein = self.translate_cds(cds_sequence)
        protein_encoding = self.protein_tokenizer(
            original_protein,
            padding='max_length',
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            self.naturality_model.load_state_dict(load_file(f"{self.perplexity_model_dir}/model.safetensors"))
            self.naturality_model.to(self.device)
            nat_protein_emb = self.naturality_model.compute_protein_embeddings(
                protein_input_ids=protein_encoding.input_ids,
                protein_attention_mask=protein_encoding.attention_mask).to(self.device)
            self.naturality_model.plantrna.to('cpu')
            expr_protein_embs = {}
            for key, model in self.models.items():
                model_path = os.path.join(self.model_dir,f"classification-model-fold-{key}")
                model.load_state_dict(load_file(f"{model_path}/model.safetensors"))
                model.to(self.device)
                expr_protein_embs[model] = model.compute_protein_embeddings(
                    protein_input_ids=protein_encoding.input_ids,
                    protein_attention_mask=protein_encoding.attention_mask).to(self.device)
                model.plantrna.to('cpu')
        print("--- Evaluating initial sequence ---")
        initial_expr = self.predict_expression_batch([cds_sequence], expr_protein_embs)[0]
        initial_nat = self.calculate_naturalness_with_mlm([cds_sequence],nat_protein_emb)[0]
        initial_fitness = initial_expr * (initial_nat ** self.perplexity_weight if initial_nat > 0 else 0)

        evaluated_data = {cds_sequence: {
                'expression': initial_expr,
                'naturality': initial_nat,
                'fitness': initial_fitness}}

        log_records = []

        log_records.append({
            'iteration': 0,
            'fitness': initial_fitness,
            'naturalness': initial_nat,
            'expression': initial_expr,
            'sequence': cds_sequence,
        })
        all_sequences_and_origins = {cds_sequence: 0}
        current_sequence_for_generation = cds_sequence
        global_best_fitness = initial_fitness
        global_best_sequence = cds_sequence
        global_best_expression = initial_expr
        global_best_naturality = initial_nat
        iteration_of_best_sequence = 0
        total_iterations_run, cycle_number = 0, 0

        while True:
            cycle_number += 1
            print(f"\n==================== Starting generation of cycle {cycle_number} ====================")
            for _ in range(self.iterations):
                total_iterations_run += 1
                print(f"--- Iteration {total_iterations_run} ---", end='\r')
                all_gains_by_pos = self.propose_mutations_with_gradients(current_sequence_for_generation,nat_protein_emb, expr_protein_embs)
                best_proposals_by_pos = {}
                for pos, gains_dict in all_gains_by_pos.items():
                    if gains_dict:
                        best_codon, best_gain = max(gains_dict.items(),key=lambda item: item[1])
                        if best_gain > 0:
                            best_proposals_by_pos[pos] = (best_codon, best_gain)
                new_sequence = ''
                if not best_proposals_by_pos:
                    new_sequence = current_sequence_for_generation
                else:
                    sorted_proposals = sorted(best_proposals_by_pos.items(),key=lambda item: item[1][1], reverse=True)
                    num_codons = len(current_sequence_for_generation) // 3
                    k = max(1,int(num_codons * self.mutation_rate))
                    top_k_proposals = sorted_proposals[:k]
                    mutated_codons_list = [current_sequence_for_generation[i:i + 3] for i in range(0,len(current_sequence_for_generation),3)]
                    for codon_idx, (new_codon,gain) in top_k_proposals:
                        mutated_codons_list[codon_idx] = new_codon
                    if self.use_reversibility_check:
                        tentative_new_sequence = "".join(mutated_codons_list)
                        second_pass_gains = self.propose_mutations_with_gradients(
                            tentative_new_sequence,
                            nat_protein_emb,
                            expr_protein_embs,
                            check=True)
                        best_second_pass_proposals = {}
                        for pos, gains_dict in second_pass_gains.items():
                            if gains_dict:
                                best_codon, best_gain = max(gains_dict.items(),key=lambda item:item[1])
                                if best_gain > 0:
                                    best_second_pass_proposals[pos] = (best_codon,best_gain)

                        final_codons_list = list(mutated_codons_list)
                        if best_second_pass_proposals:
                            sorted_second_pass_proposals = sorted(best_second_pass_proposals.items(),key=lambda item:item[1][1],reverse=True)
                            top_k_second_pass_proposals = sorted_second_pass_proposals[:k]
                            for codon_idx, (new_codon,gain) in top_k_second_pass_proposals:
                                final_codons_list[codon_idx] = new_codon
                        new_sequence = "".join(final_codons_list)
                    else:
                        new_sequence = "".join(mutated_codons_list)

                while new_sequence in all_sequences_and_origins:
                    print(f"\nIteration {total_iterations_run}: Loop detected. Applying random mutations to escape.")
                    new_sequence = self._apply_random_synonymous_mutations(current_sequence_for_generation,num_mutations=1)

                current_sequence_for_generation = new_sequence
                all_sequences_and_origins[new_sequence] = total_iterations_run

            print(f"--- Evaluation after iteration {total_iterations_run} ---")

            all_generated_seqs_set = set(all_sequences_and_origins.keys())
            already_evaluated_seqs_set = set(evaluated_data.keys())
            new_sequences_to_evaluate = list(all_generated_seqs_set - already_evaluated_seqs_set)

            if new_sequences_to_evaluate:
                expression_scores = self.predict_expression_batch(new_sequences_to_evaluate,expr_protein_embs)
                naturality_scores = self.calculate_naturalness_with_mlm(new_sequences_to_evaluate,nat_protein_emb)

                for seq, expr, nat in zip(new_sequences_to_evaluate,expression_scores,naturality_scores):
                    fitness = expr * (nat ** self.perplexity_weight if nat > 0 else 0)
                    evaluated_data[seq] = {'expression': expr,'naturality': nat,'fitness': fitness}

                    origin_iteration = all_sequences_and_origins.get(seq, total_iterations_run)  # Fallback just in case
                    log_records.append({
                        'iteration': origin_iteration,
                        'fitness': fitness,
                        'naturalness': nat,
                        'expression': expr,
                        'sequence': seq,
                    })

            if results_dir and new_sequences_to_evaluate:
                log_df = pd.DataFrame(log_records)
                log_df = log_df.sort_values(by=['iteration', 'fitness'], ascending=[True, False])
                csv_path = os.path.join(results_dir, "optimization_log.csv")
                log_df.to_csv(csv_path, index=False, encoding='utf-8')


            current_best_sequence = max(evaluated_data,key=lambda k: evaluated_data[k]['fitness'])
            current_best_fitness = evaluated_data[current_best_sequence]['fitness']
            current_best_expression = evaluated_data[current_best_sequence]['expression']
            current_best_naturality = evaluated_data[current_best_sequence]['naturality']

            if current_best_fitness > global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_sequence = current_best_sequence
                global_best_expression = current_best_expression
                global_best_naturality = current_best_naturality
                iteration_of_best_sequence = all_sequences_and_origins[global_best_sequence]

            iterations_since_improvement = total_iterations_run - iteration_of_best_sequence
            if total_iterations_run >= self.max_iterations or (
                    iterations_since_improvement >= self.patience and global_best_expression > self.min_expression_threshold and global_best_naturality > self.min_naturality_threshold):
                break

        print("\n==================== Optimization Complete ====================")

        final_sorted_sequences = sorted(evaluated_data.items(),key=lambda item: item[1]['fitness'],reverse=True)
        top_results = final_sorted_sequences[:min(self.top_n_return,len(final_sorted_sequences))]
        original_data = evaluated_data.get(cds_sequence,{'expression': 0, 'naturality': 0,'fitness': 0})

        return {'original_sequence': cds_sequence,
            'optimized_sequence': global_best_sequence,
            'best_sequence_origin': f"Iteration {iteration_of_best_sequence}",
            'original_expression': original_data['expression'],
            'optimized_expression':evaluated_data[global_best_sequence]['expression'],
            'original_naturality': original_data['naturality'],
            'optimized_naturality':evaluated_data[global_best_sequence]['naturality'],
            'protein_sequence': original_protein,
            'top_sequences': [s for s, d in top_results],
            'top_expressions': [d['expression'] for s, d in top_results],
            'top_naturalities': [d['naturality'] for s, d in top_results],
            'top_fitness_scores': [d['fitness'] for s, d in top_results]}


def t_to_u(sequence):
    return sequence.replace('T', 'U')


def is_valid_cds(cds_sequence):
    if len(cds_sequence) % 3 != 0 or not re.fullmatch(
            '^[AUGC]*$', cds_sequence):
        return False
    stop_codons = {'UAA', 'UAG', 'UGA'}
    codons = [cds_sequence[i:i + 3] for i in
              range(0, len(cds_sequence), 3)]
    if any(codons[i] in stop_codons for i in
           range(len(codons) - 1)):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--CodonEXP_model_dir', type=str, required=True)
    parser.add_argument('--CodonNAT_model_dir', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_n_return', type=int, default=1)
    parser.add_argument('--perplexity_weight', type=float, default=1)
    parser.add_argument('--max_iterations', type=int, default=96)
    parser.add_argument('--min_expression_threshold', type=float, default=0.9)
    parser.add_argument('--min_naturality_threshold', type=float, default=0.6)
    parser.add_argument('--hallucination_perplexity_weight', type=float, default=1)
    parser.add_argument('--mutation_rate', type=float, default=0.15)
    parser.add_argument('--results_dir', type=str, default='./optimization_results')
    parser.add_argument('--use_reversibility_check', action='store_true')
    parser.add_argument('--codon_frequency_file', type=str, required=True)
    parser.add_argument('--codon_frequency_weight', type=float, default=1.0)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    target_distribution = parse_codon_frequency_csv(args.codon_frequency_file)
    if target_distribution is None:
        return

    optimizer_args = vars(args)
    optimizer_args['codon_frequency_data'] = target_distribution

    sequences = parse_fasta(args.input)
    optimized_sequences = {}
    optimizer = GradientOptimizer(**optimizer_args)

    for seq_id, cds_sequence in sequences.items():
        print(f"\n==================== Optimizing sequence: {seq_id} ====================")
        cds_sequence = t_to_u(cds_sequence.upper())
        if not is_valid_cds(cds_sequence):
            print(f"Sequence {seq_id} is invalid, skipping.")
            optimized_sequences[seq_id] = {
            'sequence': cds_sequence,
            'naturality': 'N/A',
            'expression': 'N/A',
            'fitness': 'N/A'
        }
            continue

        seq_name = re.sub(r'[^\w\.-]', '_', seq_id.split()[0])
        seq_results_dir = os.path.join(args.results_dir, seq_name)
        os.makedirs(seq_results_dir, exist_ok=True)
        result = optimizer.optimize_with_gradients(cds_sequence, results_dir=seq_results_dir)

        if not result or not result.get('top_sequences'):
            print(f"Sequence {seq_id} optimization failed, keeping the original sequence.")
            optimized_sequences[seq_id] = {'sequence': cds_sequence,'naturality': 'N/A','expression': 'N/A','fitness': 'N/A'}
            continue

        for i, seq in enumerate(result['top_sequences']):
            new_seq_id = f"{seq_id}_opt_{i + 1}"
            optimized_sequences[new_seq_id] = {'sequence': seq,
                'naturality':result['top_naturalities'][i],
                'expression':result['top_expressions'][i],
                'fitness':result['top_fitness_scores'][i]}

        seq_name = re.sub(r'[^\w\.-]', '_', seq_id.split()[0])
        seq_results_dir = os.path.join(args.results_dir, seq_name)
        os.makedirs(seq_results_dir,exist_ok=True)
        with open(os.path.join(seq_results_dir,"summary.txt"),'w', encoding='utf-8') as f:
            f.write(f"Original CDS: {result['original_sequence']}\n")
            f.write(f"Protein: {result['protein_sequence']}\n\n")
            f.write(f"Best sequence found at: {result['best_sequence_origin']}\n")

            original_expr = result['original_expression']
            original_nat = result['original_naturality']
            original_fitness = original_expr * (original_nat ** optimizer.perplexity_weight if original_nat > 0 else 0)

            f.write(f"Original fitness: {original_fitness:.4f}\n")
            f.write(f"  - Expression level: {original_expr:.4f}\n")
            f.write(f"  - Naturalness (MLM): {original_nat:.4f}\n\n")
            f.write("--- Top optimized sequences ---\n")
            for i in range(len(result['top_sequences'])):
                f.write(f"\nRank {i + 1}:\n")
                f.write(f"  Sequence: {result['top_sequences'][i]}\n")
                f.write(f"  Fitness: {result['top_fitness_scores'][i]:.4f}\n")
                f.write(f"  High-expression probability: {result['top_expressions'][i]:.4f}\n")
                f.write(f"  Naturalness: {result['top_naturalities'][i]:.4f}\n")
        print(f"\nOptimization for sequence {seq_id} is complete. Best sequence found at: {result['best_sequence_origin']}")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            for seq_id, seq_data in optimized_sequences.items():
                sequence = seq_data['sequence']
                naturality = seq_data['naturality']
                expression = seq_data['expression']
                fitness = seq_data['fitness']

                if naturality == 'N/A' or expression == 'N/A' or fitness == 'N/A':
                    f.write(f">{seq_id} naturalness=N/A expression=N/A fitness=N/A\n")
                else:
                    f.write(f">{seq_id} naturalness={naturality:.4f} expression={expression:.4f} fitness={fitness:.4f}\n")
                f.write(f"{sequence}\n")

    print(f"\nAll sequences processed. Optimized FASTA file saved to: {args.output}")


if __name__ == "__main__":
    main()
