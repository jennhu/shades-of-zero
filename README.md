# Shades of Zero: Distinguishing Impossibility from Inconceivability

This repository contains materials for the paper "Shades of Zero: Distinguishing Impossibility from Inconceivability"
by Jennifer Hu, Felix Sosa, and Tomer Ullman.

The materials are organized in the following folders:
- `analysis`: contains notebooks and scripts for reproducing the figures and
statistical analyses from the paper
- `data`: contains all stimuli and results from our experiments
- `figures`: contains rendered figures (in PDF format) generated from `notebooks/figures.ipynb`
- `python`: contains scripts and source code for running language model evaluations (Experiment 2)

## Stimuli

The stimuli used in Experiments 1 and 2 (original 30 items) can be found at
`stimuli/stimuli.csv`. The stimuli follow a wide format: each line contains an 
item, which is identified by its unique number `item_id`. The six continuations
are labeled by their condition: e.g., the `improbable` column contains the
continuations in the improbable condition.

The file `stimuli_agree75_clean.csv` contains the stimuli used in Experiment 3.
Please see the paper for more details.

## Evaluating language model surprisal

To obtain language model surprisals on our stimuli, run
```bash
python python/get_surprisals.py \
    --input data/stimuli/stimuli.csv \
    --output data/exp2_model_surprisal/surprisals \
    --model <MODEL_NAME>
```
The `--model` argument takes a model identifier recognized by Huggingface.
In our experiments, we use `gpt2`, `meta-llama/Llama-2-7b-hf`, and 
`mistralai/Mistral-7B-v0.1`, but you can replace `<MODEL_NAME>` with any 
autoregressive Huggingface model identifier. You may need to log in with your 
Huggingface account to access gated models such as Llama. 

Note that these scripts require the `surprisal` package.
Please see the [documentation](https://aalok-sathe.github.io/surprisal/surprisal.html)
for detailed usage and installation instructions.

## Figures and analyses

The figures in the paper can be reproduced using the Jupyter notebook at
`analysis/figures.ipynb`.

The statistical analyses can be reproduced using the R scripts in the folder
`analysis/r_scripts`. The outputs from the fitted models are saved to the folder
`analysis/r_output` in human-readable plain text files.