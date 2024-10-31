# Shades of Zero: Distinguishing Impossibility from Inconceivability

This repository contains materials for the paper "Shades of Zero: Distinguishing Impossibility from Inconceivability"
by Jennifer Hu, Felix Sosa, and Tomer Ullman.

The materials are organized in the following folders:
- `analysis`: contains notebooks and scripts for reproducing the figures and
statistical analyses from the paper
- `data`: contains all stimuli and results from our experiments
- `figures`: contains rendered figures (in PDF format) generated from `analysis/figures.ipynb`
- `python`: contains scripts and source code for running language model evaluations (Experiment 3)

## Stimuli

The stimuli used in our experiments can be found at `data/stimuli/stimuli_with_syntax.csv`. 
The stimuli follow a wide format: each line corresponds to one item, 
which is identified by its unique number `item_id`. The four continuations
are labeled by their condition: e.g., the `improbable` column contains the
continuations in the improbable condition.

Please note that the `inconceivable_syntactic` column (corresponding to the
syntactic violation condition) was only used in the language model evaluation
(Experiment 3) and not in our human categorization or rating experiments
(Experiments 1 and 2, respectively). Please see the paper for more details.

## Evaluating language model surprisal

### Neural models (Huggingface)

To obtain language model surprisals on our stimuli, run
```bash
python python/get_surprisals.py \
    --input data/stimuli/stimuli_with_syntax.csv \
    --output data/exp3_model_surprisal/surprisals \
    --model <MODEL_NAME>
```
The `--model` argument takes a model identifier recognized by Huggingface.
Please see Table 2 of our paper for the exact model IDs used in our experiments.
You may need to log in with your Huggingface account to access gated models such as Llama. 

Note that this script requires the `minicons` package.

### Infini-gram model and n-gram frequencies

To estimate surprisals from the infini-gram model and n-gram frequency counts, run
```bash
python python/get_ngram_data.py \
    --get_counts \
    --get_probs \
    --ngram_index v4_dolma-v1_7_llama
```
The flag `--get_counts` tells the script to estimate n-gram counts, and the flag
`--get_probs` tells the script to estimate infini-gram probabilities.
You can omit either flag to only perform one of these actions.

For more details about the infini-gram engine, please see [https://infini-gram.io/](https://infini-gram.io/).

## Figures and analyses

The figures in the paper can be reproduced using the Jupyter notebook at
`analysis/figures.ipynb`. By default, the rendered figures are saved to the
`figures` folder in PDF format.

The statistical analyses can be reproduced using the R scripts in the folder
`analysis/r_scripts`. The outputs from the fitted models are saved to the folder
`analysis/r_output` in human-readable plain text files.