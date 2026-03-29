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
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

HalluCodon is a species-specific codon optimization framework designed for plant expression systems. It integrates pre-trained protein (ESM2) and RNA (mRNA-FM) language models, applying supervised fine-tuning on species-specific datasets to generate coding sequences optimized for improved protein expression.



## Installation

Our CUDA version is 12.2.

1. Create a conda environment

   ```sh
   conda create -n HalluCodon python=3.10/
   conda activate HalluCodon
   ```

3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```







<!-- USAGE EXAMPLES -->

## Usage

1. Initialize CDS

   ```sh
   python CondaIni.py \
   --model_path ./plantmodel/Ntabacum4097/Ntabacum4097-CodonNAT \
   --input_file ./input_pro.fasta \
   --output_file ./CodonIni.fasta
   ```


2. Optimize CDS with codonEvo

   ```sh
   python CodonGa.py \
   --CodonEXP_model_dir ./plantmodel/Ntabacum4097/Ntabacum4097-CodonEXP \
   --population_size 100 \
   --mutation_rate 0.05 \
   --crossover_rate 0.7 \
   --max_generations 100 \
   --batch_size 50 \
   --selection_top_percent 0.2 \
   --results ./Ntabacum4097/results \
   --history ./Ntabacum4097/history \
   --natural_weight 1 \
   --CodonNAT_model_dir ./plantmodel/Ntabacum4097/Ntabacum4097-CodonNAT \
   --input ./CodonIni.fasta \
   --output ./CodonGa.fasta
   ```

The model weights and detailed explanations of the parameters will be made public after the manuscript is submitted or published.

3.  Optimize CDS with codonHallucination

   ```sh
   python CodonHallucination.py --model_dir ./plantmodel/plaNtabacum4097/Ntabacum4097-aux1-2-classify \
   --mutation_rate 0.15 \
   --iterations 16 \
   --max_iterations 96 \
   --min_expression_threshold 0.9 \
   --min_naturality_threshold 0.6 \
   --batch_size 16 \
   --top_n 1 \
   --results_dir ./Ntabacum4097/20-pro-100/results \
   --perplexity_weight 1 \
   --hallucination_perplexity_weight 1 \
   --patience 20 \
   --perplexity_model_dir ./plantmodel/Ntabacum4097/Ntabacum4097-finetune-mrnafm-with-pro/Ntabacum4097-finetune-top10csi-top10csi_mlm \
   --input ./Ntabacum4097/cds_list.fasta \
   --output ./Ntabacum4097/cds_codonHallucination_test.txt \
   --use_reversibility_check
   --codon_frequency_file ./Ntabacum4097/Ntabacum4097-codon_count.csv
   ```

<!-- LICENSE -->

## License

Distributed under the project_license. See `LICENSE.txt` for more information.



