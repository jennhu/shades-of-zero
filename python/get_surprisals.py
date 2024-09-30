import pandas as pd
import argparse
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import torch

from model import LM


def evaluate_model(df, model, sep=" ", check_tokenization=True):
    # Define conditions.
    conditions = [
        "probable", 
        "improbable",
        "impossible",
        "inconceivable",
        "inconceivable_syntactic", 
    ]

    # Initialize model outputs.
    summary_data, token_data = [], []

    # Iterate over rows, each of which corresponds to an experimental item.
    for _, row in tqdm(df.iterrows(), total=len(df.index)):
        for cond in conditions:
            prefix = row.sentence_prefix
            continuation = row[cond] + "."
            token_scores = model.scorer.token_score(
                prefix + sep + continuation,
                surprisal=False,
                prob=False,
                decode=False,
                bos_token=True
            )[0]

            # Aggregate logprobs corresponding to tokens in the continuation.
            # Tokenize the prefix by itself.
            prefix_tokens = model.tokenizer(prefix, add_special_tokens=True)["input_ids"]

            # Manually add BOS token if it is applicable.
            # Unfortunately, this needs to be done for certain models.
            if model.tokenizer.bos_token is not None and prefix_tokens[0] != model.tokenizer.bos_token:
                prefix_tokens = torch.tensor(
                    [model.tokenizer.bos_token_id] + prefix_tokens
                )

            # Check that the prefix tokens match the beginning.
            # This is optional, and can be skipped for speed.
            if check_tokenization:
                prefix_tokens = model.tokenizer.convert_ids_to_tokens(prefix_tokens)
                assert all(
                    prefix_tokens[i] == token_scores[i][0] 
                    for i in range(len(prefix_tokens))
                )

            continuation_scores = [
                t[1] for t in token_scores[len(prefix_tokens):]
            ]

            # Compute sum and mean logprob of continuation.
            continuation_sum_logprob = np.sum(continuation_scores)
            continuation_mean_logprob = np.mean(continuation_scores)

            # Update summary data.
            summary_data.append(dict(
                item_id=row.item_id,
                eval_prefix=prefix,
                condition=cond,
                continuation=continuation,
                continuation_sum_logprob=continuation_sum_logprob,
                continuation_mean_logprob=continuation_mean_logprob
            ))

            # Also record token-level data for reference and other metrics
            # such as SLOR.
            for token_idx, (token, token_score) in enumerate(token_scores):
                token_data.append(dict(
                    item_id=row.item_id,
                    eval_prefix=prefix,
                    condition=cond,
                    continuation=continuation,
                    token_idx=token_idx,
                    token=token,
                    token_logprob=token_score,
                    is_continuation_token=(token_idx>=len(prefix_tokens))
                ))
            
    summary_df = pd.DataFrame(summary_data)
    summary_df["model"] = model.model_name
    token_df = pd.DataFrame(token_data)
    token_df["model"] = model.model_name
    return summary_df, token_df

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on stimuli.")
    parser.add_argument("-i", "--input", type=Path, 
                        default="data/stimuli/stimuli.csv", 
                        help="Path to CSV file containing stimuli")
    parser.add_argument("-o", "--output", type=Path, 
                        default="data/exp2_model_surprisal/surprisals",
                        help="Path to directory where output files will be saved")
    parser.add_argument("-m", "--model", type=str, default="gpt2",
                        help="Huggingface model identifier")
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="Path to Huggingface cache directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Read stimulus data.
    print(f"Reading stimuli from {args.input}")
    df = pd.read_csv(args.input)

    # Make output directory.
    os.makedirs(args.output, exist_ok=True)

    # Replace the placeholders in the stimuli.
    noun_phrase, pronoun, poss = "He", "he", "his"
    df.replace({
        '\[NP\]': noun_phrase, # noun phrase placeholder
        '\[PN\]': pronoun, # pronoun placeholder
        '\[POSS\]': poss # possessive placeholder
    }, regex=True, inplace=True)

    # Initialize model.
    print(f"Initializing Huggingface model {args.model}")
    m = LM(args.model, cache_dir=args.cache_dir)

    # Evaluate model and save outputs.
    summary_df, token_df = evaluate_model(df, m)

    safe_model_name = args.model.split("/")[-1].lower()

    summary_output_file = Path(args.output, f"summary_{safe_model_name}.csv")
    token_output_file = Path(args.output, f"token_{safe_model_name}.csv")
    summary_df.to_csv(summary_output_file, index=False)
    token_df.to_csv(token_output_file, index=False)
    print(f"Saved model outputs!")