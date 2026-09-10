# HalluCodon: a species-specific codon optimizer guided by multimodal language models and hallucination design



<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#Installation">Installation</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#optional-species">Optional Species</a></li>
    <li><a href="#how-to-add-one-new-species">How to add one new species</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

HalluCodon is a species-specific codon optimization framework designed for plant expression systems. It integrates pre-trained protein (ESM2) and RNA (mRNA-FM) language models, applying supervised fine-tuning on species-specific datasets to generate coding sequences optimized for improved protein expression.

![overview](./overview.png)

<!-- Installation -->

## Installation



### 1. Create a conda environment

   ```sh
   conda create -n HalluCodon python=3.10
   conda activate HalluCodon
   ```

### 2. Install dependencies
   ```sh
   git clone https://github.com/YuxuanLou/HalluCodon.git
   cd HalluCodon
   pip install -r requirements.txt
   wget -O multimolecule.tar.gz "https://zenodo.org/records/19807318/files/multimolecule.tar.gz?download=1"
   tar -zxvf multimolecule.tar.gz
   huggingface-cli download facebook/esm2_t33_650M_UR50D \
   --local-dir ./facebook/esm2_t33_650M_UR50D \
   --local-dir-use-symlinks False
   ```







<!-- USAGE EXAMPLES -->

## Usage

### 1. Initialize CDS

   ```sh
   python CodonIni.py \
   --model_path ./plantmodel/Ntabacum4097/Ntabacum4097-CodonNAT \
   --input_file ./input_pro.fasta \
   --output_file ./CodonIni.fasta
   ```
  **--model_path** The path where the trained species-specific CodonNAT model is stored.
  
  **--input_file** The amino acid sequence prepared for codon optimization. Input must be in standard FASTA format with a header line starting with ">" (e.g., ">RFP").
  
  **--output_file** The CDS obtained from the CodonIni step will be written into this file.

### 2. Optimize CDS with CodonGa

   ```sh
   python CodonGa.py \
   --CodonEXP_model_dir ./Ntabacum4097/Ntabacum4097-CodonEXP \
   --population_size 100 \
   --mutation_rate 0.05 \
   --crossover_rate 0.7 \
   --max_generations 100 \
   --batch_size 50 \
   --selection_top_percent 0.2 \
   --top_n 1 \
   --results ./results \
   --history ./history \
   --naturalness_weight 1 \
   --CodonNAT_model_dir ./Ntabacum4097/Ntabacum4097-CodonNAT \
   --input ./CodonIni.fasta \
   --output ./CodonGa.fasta
   ```

   **--CodonEXP_model_dir** The path where the trained species-specific CodonEXP model is stored.
   
   **--CodonNAT_model_dir** The path where the trained species-specific CodonNAT model is stored.
   
   **--naturalness_weight** Weight of naturalness in fitness calculation. The fitness calculation formula is: fitness = high-expression probability × (naturalness ^ this weight). A larger weight means naturalness has a more significant impact on fitness.
   
   **--population_size** The population size in the genetic algorithm, which refers to the number of sequences included in each generation. A larger population size may increase diversity but also raises computational costs.
   
   **--mutation_rate** The probability of synonymous substitution occurring at each codon in the sequence. The larger the value of this parameter, the greater the sequence variation between generations.
   
   **--crossover_rate** Controls the probability that parent sequences produce offspring through crossover operations. The larger the value of this parameter, the greater the sequence variation between generations.
   
   **--max_generations** The maximum number of iterations in the genetic algorithm.
   
   **--batch_size** Batch size for model prediction. When calculating high protein abundance probability and naturality, sequences will be grouped into batches of this size for model input, balancing computational efficiency and memory usage.
   
  **--selection_top_percent** The proportion of high-fitness sequences retained during the selection operation in the genetic algorithm.
  
  **--top_n** Number of optimal sequences to return after optimization (sorted by fitness).
  
  **--results** The storage path for detailed results.
  
  **--history** The storage path for the optimization history.
  
  **--input** This file needs to contain the CDS obtained from the CodonIni step. Input must be in standard FASTA format with a header line starting with ">" (e.g., ">RFP").
  
  **--output** The CDS obtained from the CodonGa step will be written into this file.

### 3. Optimize CDS with CodonHa

   ```sh
   python CodonHa.py \
   --CodonEXP_model_dir ./Ntabacum4097/Ntabacum4097-CodonEXP \
   --mutation_rate 0.15 \
   --iterations 16 \
   --max_iterations 96 \
   --min_expression_threshold 0.9 \
   --min_naturalness_threshold 0.6 \
   --batch_size 16 \
   --top_n 1 \
   --results_dir ./results \
   --naturalness_weight 1 \
   --hallucination_naturalness_weight 1 \
   --patience 20 \
   --CodonNAT_model_dir ./Ntabacum4097/Ntabacum4097-CodonNAT \
   --input ./CodonIni.fasta \
   --output ./CodonHa.fasta \
   --codon_frequency_file ./codon_freq/Tobacco-codon-count.csv
   ```

   **--CodonEXP_model_dir** The path where the trained species-specific CodonEXP model is stored.
   
   **--CodonNAT_model_dir** The path where the trained species-specific CodonNAT model is stored.
   
   **--naturalness_weight** Weight of naturalness in fitness calculation. The fitness calculation formula is: fitness = high-expression probability × (naturalness ^ this weight). A larger weight means naturalness has a more significant impact on fitness.
   
   **--mutation_rate** Proportion of codons allowed to mutate in each iteration (relative to total codon count), controlling the intensity of single mutation.
   
   **--iterations** Number of iterations for gradient-guided mutation in each generation, sequence performance is evaluated after each generation.
   
   **--max_iterations** Maximum number of iterations.
   
   **--min_expression_threshold** Expression threshold for early stopping mechanism: optimization can terminate early if the best sequence expression exceeds this value.
   
   **--min_naturalness_threshold** Naturality threshold for early stopping mechanism: optimization can terminate early if the best sequence naturality exceeds this value.
   
   **--batch_size** Batch size for model prediction. When calculating high protein abundance probability and naturality, sequences will be grouped into batches of this size for model input, balancing computational efficiency and memory usage.
   
   **--top_n** Number of optimal sequences to return after optimization (sorted by fitness).
   
   **--results_dir** The storage path for detailed results and optimization history.
   
   **--naturalness_weight** Weight of naturalness in fitness calculation. The fitness calculation formula is: fitness = high-expression probability × (naturalness ^ this weight). A larger weight means naturalness has a more significant impact on fitness.
   
   **--hallucination_naturalness_weight** Weight of naturalness in mutation gain calculation, adjusting the impact of naturalness on mutation selection.
   
   **--patience** Early stopping mechanism parameter: under the premise of meeting high-expression probability and naturalness thresholds, if no better solution is found for this number of consecutive iterations, optimization will terminate early.
   
   **--input** This file needs to contain the CDS obtained from the CodonIni step. Input must be in standard FASTA format with a header line starting with ">" (e.g., ">RFP").
  
  **--output** The CDS obtained from the CodonHa step will be written into this file.
  
  **--codon_frequency_file** Species-specific codon usage frequency file.
  
  ### 4. Optimize CDS with Ha-GC3

   ```sh
   python Ha-GC3.py \
   --CodonEXP_model_dir ./Ntabacum4097/Ntabacum4097-CodonEXP \
   --mutation_rate 0.15 \
   --iterations 16 \
   --max_iterations 96 \
   --min_expression_threshold 0.9 \
   --min_naturalness_threshold 0.6 \
   --batch_size 16 \
   --top_n 1 \
   --results_dir ./results \
   --naturalness_weight 1 \
   --hallucination_naturalness_weight 1 \
   --patience 20 \
   --CodonNAT_model_dir ./Ntabacum4097/Ntabacum4097-CodonNAT \
   --input ./CodonIni.fasta \
   --output ./Ha-GC3.fasta \
   --gc3_weight 5
   ```

   **--gc3_weight** Adjusts the GC3 content of the generated sequences. When this value is greater than 1, the use of GC3 codons is encouraged; otherwise, the use of GC3 codons is reduced.


## Optional Species
We trained the CodonNAT and CodonEXP models separately on 15 plant species. The species names and their corresponding weight storage paths are as follows:
Arabidopsis(https://zenodo.org/records/19126265), Canola(https://zenodo.org/records/19129186), Sweet orange(https://zenodo.org/records/19133772), Cotton(https://zenodo.org/records/19135167), Soybean(https://zenodo.org/records/19135889), Barley(https://zenodo.org/records/19136614), Medicago(https://zenodo.org/records/19143273), Tobacco(https://zenodo.org/records/19143653), Rice(https://zenodo.org/records/19144006), Earthmoss(https://zenodo.org/records/19144435), Tomato(https://zenodo.org/records/19150439), Potato(https://zenodo.org/records/19150918), Wheat(https://zenodo.org/records/19151412), Grape(https://zenodo.org/records/19151918), Maize(https://zenodo.org/records/19152348).

<!-- ADD A NEW SPECIES -->

## How to add one new species

To extend HalluCodon to a species not covered by the releases above, train both
models on that species' own data — **CodonNAT first** (it scores codon naturalness
and guides the optimizer search), then **CodonEXP** (it scores high-expression
probability). The commands below run from the repository root, use
**Escherichia coli** (genus *Escherichia*, NCBI taxid **511145**, strain K-12) as
the worked example, and keep all intermediate files under `./tmp/`.

### Step 1 · CodonNAT: collect data and train

**1a. Download the NCBI assembly summary** (resumable; skipped automatically if
it already exists — `02_build_top10_U.sh` can also fetch it on demand):

```sh
cd data_preparation/CodonNAT_data_generate
./01_download_assembly_summary.sh ../../../tmp/assembly_summary_genbank.txt
```

**1b. High-CSI CDS training set.** The one-command pipeline downloads the taxon's
reference CDS from NCBI, keeps complete ORFs, de-duplicates by translated protein,
builds a genome-wide codon-frequency table, ranks CDS by codon stability index
(CSI) against that table, and keeps the top 10% (converted to RNA alphabet):

```sh
./02_build_top10_U.sh Escherichia -o ../../../tmp \
    -a ../../../tmp/assembly_summary_genbank.txt
cd ../../..
# output: tmp/top10_U.csv  (columns: cds_sequence, protein_sequence, csi_value)
```

With a local genome CDS FASTA instead (e.g. K-12
`GCF_000005845.2_ASM584v2_cds_from_genomic.fna`), run the three steps manually:

```sh
mkdir -p tmp
# keep complete CDS only (ATG start, in-frame stop, length % 3 == 0)
python data_preparation/filter_cds.py Ecoli_cds.fna tmp/ecoli_filtered.csv
# genome-wide codon frequency table in codon_freq/ format (RNA/U)
python data_preparation/count_codon_freq.py tmp/ecoli_filtered.csv codon_freq/Ecoli-codon-count.csv
# rank by CSI vs that table, keep the top 10%
python get_high_csi-seq.py --input tmp/ecoli_filtered.csv \
    --codon_freq codon_freq/Ecoli-codon-count.csv --output tmp/Ecoli_top10.csv
```

Keep `codon_freq/Ecoli-codon-count.csv` — CodonHa uses it at optimization time.

**1c. Train CodonNAT** (masked-codon self-supervised fine-tuning):

```sh
python train_and_test/CodonNAT_train.py \
    --output_dir ./Ecoli_CodonNAT \
    --dataset_path ./tmp/Ecoli_top10.csv \
    --model_name Ecoli-CodonNAT
# final weights -> Ecoli_CodonNAT/Ecoli_top10-Ecoli-CodonNAT/model.safetensors
# best epoch kept by validation mask accuracy; weights only, no optimizer checkpoints
```

Both encoders start from pretrained weights (mRNA-FM for the CDS, ESM2-650M for
the protein) and are fine-tuned together at lr 1e-4 with AdamW (weight decay 0.01).
Training masks random codons and minimizes cross-entropy on the masked positions,
on an 80/10/10 train/val/test split (`random_state=42`), batch size 4, max length
1024, up to 50 epochs with early stopping (patience 5).

### Step 2 · CodonEXP: collect data and train

**2a. Expression-labeled CDS set.** Match a PaxDb protein-abundance dataset back
to the species CDS, binarize by abundance (top 1/3 = `high`, label 1; bottom 1/3 =
`low`, label 0; middle third dropped), and de-duplicate proteins with cd-hit at
90% identity:

```sh
cd data_preparation/CodonEXP_data_generate
# 1) E. coli proteins + one abundance dataset from PaxDb v5.0
./get-paxdb-proteins.sh 511145 ../../../tmp/fasta.v11.5.511145.fa
./get-paxdb-dataset.sh 511145 list                       # list available datasets
./get-paxdb-dataset.sh 511145 WHOLE_ORGANISM-integrated \
    ../../../tmp/511145-WHOLE_ORGANISM-integrated.txt
# 2) proteins -> CDS (exact match first, then one batched BLASTP run;
#    >=90% identity, >=50% coverage), binarize, cd-hit 0.9 de-duplication
python3 paxdb-codonexp.py \
    --cds ../../../Ecoli_cds.fna \
    --dataset ../../../tmp/511145-WHOLE_ORGANISM-integrated.txt \
    --proteins ../../../tmp/fasta.v11.5.511145.fa \
    --out ../../../Ecoli-0.9.csv \
    --threshold 90 --coverage 50 --parallel 8 --workdir ../../../tmp/paxdb-work
cd ../../..
# output columns: id, abundance, protein_sequence_ori, uniprot_id, exp, label,
#                 cds_sequence, protein_sequence, similarity, coverage
```

**2b. Train CodonEXP** (high/low classifier, 5-fold cross-validation):

```sh
python train_and_test/CodonEXP_train_and_test.py \
    --output_dir ./Ecoli-CodonEXP \
    --dataset_path ./Ecoli-0.9.csv
# fold weights -> Ecoli-CodonEXP/classification-model-fold-{1..5}
# per-fold and ensemble metrics on the held-out 20% test set are printed and saved
```

The data are split 80/20 with stratification (`random_state=42`), and the 80% is
further divided into 5 cross-validation folds (`random_state=100`). Both encoders
(mRNA-FM for the CDS, ESM2-650M for the protein) start from pretrained weights and
are fine-tuned at lr 1e-5, while the custom head (attention pooling plus an MLP
with a single logit) trains at lr 1e-3. The loss is `BCEWithLogitsLoss` plus an
equal-weight auxiliary loss on the CDS branch; the RNA/protein fusion weights are
learned and reported per fold. Each fold runs for 20 epochs (batch 4, eval 16, max
length 1024), and the best epoch is chosen by validation f1 — saved as weights
only, without optimizer state.

To keep one single model instead of five folds (e.g. one model per tissue or
condition), use `train_and_test/CodonEXP_train_single_tissue.py` with the same
flags.

### Step 3 · Use the new species

Point the optimizers at the trained weights and the species codon table:

```sh
# initialize CDS from a target protein
python CodonIni.py \
    --model_path ./Ecoli_CodonNAT/Ecoli_top10-Ecoli-CodonNAT \
    --input_file ./input_pro.fasta --output_file ./CodonIni.fasta

# hallucination-aided optimization
python CodonHa.py \
    --CodonEXP_model_dir ./Ecoli-CodonEXP \
    --CodonNAT_model_dir ./Ecoli_CodonNAT/Ecoli_top10-Ecoli-CodonNAT \
    --codon_frequency_file ./codon_freq/Ecoli-codon-count.csv \
    --input ./CodonIni.fasta --output ./CodonHa.fasta --results_dir ./results
```

`CodonGa.py` and `Ha-GC3.py` take the same model paths; see *Usage* above.

