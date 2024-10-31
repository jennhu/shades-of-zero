import pandas as pd
import argparse
from tqdm import tqdm
import os
from pathlib import Path

import requests
from transformers import AutoTokenizer


def count_ngrams(df, index="v4_dolma-v1_7_llama"):
    # Define conditions.
    conditions = [
        "probable", 
        "improbable",
        "impossible",
        "inconceivable",
        "inconceivable_syntactic", 
    ]

    # Initialize counts.
    count_data = []

    # Iterate over rows, each of which corresponds to an experimental item.
    for _, row in tqdm(df.iterrows(), total=len(df.index)):
        for cond in conditions:
            continuation = row[cond]
            result = {
                "item_id": row["item_id"],
                "condition": cond,
                "continuation": continuation
            }
            payload = {
                'index': index,
                'query_type': 'count',
                'query': continuation
            }
            result.update(requests.post(
                'https://api.infini-gram.io/', json=payload
            ).json())
            result["n_tokens"] = len(result["token_ids"])
            count_data.append(result)

    counts = pd.DataFrame(count_data)
    return counts

def eval_ngrams(df, tokenizer, index="v4_dolma-v1_7_llama"):
    # Define conditions.
    conditions = [
        "probable", 
        "improbable",
        "impossible",
        "inconceivable",
        "inconceivable_syntactic", 
    ]

    # Initialize counts.
    token_data = []

    # Iterate over rows, each of which corresponds to an experimental item.
    for _, row in tqdm(df.iterrows(), total=len(df.index)):
        for cond in conditions:
            prefix = row.sentence_prefix
            continuation = row[cond]

            # Get token IDs associated with full sentence and the prefix.
            tokens = tokenizer.encode(prefix + " " + continuation)
            prefix_tokens = tokenizer.encode(prefix)
            
            # Iterate over tokens AFTER the prefix (i.e., the continuation).
            for token_idx in range(len(prefix_tokens), len(tokens)):
                # query ids: w_1, w_2, ..., w_{n_1}, w_n
                # infini-gram will score w_n conditioned on w_{n-1}
                query_ids = tokens[:token_idx+1] 
                result = {
                    "item_id": row["item_id"],
                    "condition": cond,
                    "continuation": continuation
                }
                payload = {
                    'index': index,
                    'query_type': 'infgram_prob',
                    'query_ids': query_ids
                }
                result.update(requests.post(
                    'https://api.infini-gram.io/', json=payload
                ).json())
                
                token_data.append(result)

    token_probs = pd.DataFrame(token_data)
    return token_probs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on stimuli.")
    parser.add_argument("--input", type=Path, 
                        default="data/stimuli/stimuli_with_syntax.csv", 
                        help="Path to CSV file containing stimuli")
    parser.add_argument("--output", type=Path, 
                        default="data/exp3_model_surprisal/infinigram",
                        help="Path to directory where output files will be saved")
    parser.add_argument("--ngram_index", type=str, default="v4_dolma-v1_7_llama")
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="Path to Huggingface cache directory")
    parser.add_argument("--get_counts", action="store_true", default=False)
    parser.add_argument("--get_probs", action="store_true", default=False)
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

    # Get counts.
    if args.get_counts:
        print("Getting n-gram counts")
        counts = count_ngrams(df, index=args.ngram_index)
        counts.to_csv(
            Path(args.output, f"counts_{args.ngram_index}.csv"),
            index=False
        )

    # Get infini-gram probabilities.
    if args.get_probs:
        print("Getting infini-gram probabilities")
        if "gpt2" in args.ngram_index:
            tokenizer_model = "gpt2"
        elif "llama" in args.ngram_index:
            tokenizer_model = "meta-llama/Llama-2-7b-hf"
        else:
            tokenizer_model = "allenai/OLMo-7B"

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model, 
            add_bos_token=True, 
            add_eos_token=False,
            cache_dir=args.cache_dir
        )
        print(f"Vocabulary size (tokenizer {tokenizer_model}):", tokenizer.vocab_size)

        probs = eval_ngrams(df, tokenizer, index=args.ngram_index)
        probs.to_csv(
            Path(args.output, f"probs_{args.ngram_index}.csv"),
            index=False
        )