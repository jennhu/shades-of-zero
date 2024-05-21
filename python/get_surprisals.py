import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path

from model import LM


def evaluate_model(df, model):
    # Define conditions.
    conditions = [
        "probable", "improbable",
        "impossible_physics", "impossible_magic",
        "inconceivable_syntactic", "inconceivable_semantic"
    ]

    # Initialize model outputs.
    outputs = []

    # Iterate over rows, each of which corresponds to an experimental item.
    for _, row in tqdm(df.iterrows(), total=len(df.index)):
        for cond in conditions:
            prefix = row.sentence_prefix
            continuation = row[cond] + "."
            surp = model.get_surprisal_of_continuation(prefix, continuation)
            outputs.append(dict(
                item_id=row.item_id,
                prefix=row.prefix,
                eval_prefix=prefix,
                condition=cond,
                continuation=continuation,
                continuation_tokens=surp.text,
                continuation_sum_surprisal=surp
            ))
            
    output_df = pd.DataFrame(outputs)
    output_df["model"] = model.model_name
    return output_df

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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Read stimulus data.
    print(f"Reading stimuli from {args.input}")
    df = pd.read_csv(args.input)

    # Replace the placeholders in the stimuli.
    np, pn, poss = "He", "he", "his"
    df.replace({
        '\[NP\]': np, # noun phrase placeholder
        '\[PN\]': pn, # pronoun placeholder
        '\[POSS\]': poss # possessive placeholder
    }, regex=True, inplace=True)

    # Initialize model.
    print(f"Initializing Huggingface model {args.model}")
    m = LM(args.model)

    # Evaluate model and save outputs.
    result = evaluate_model(df, m)
    safe_model_name = args.model.split("/")[-1].lower()
    output_file = Path(args.output, f"{safe_model_name}.csv")
    result.to_csv(output_file, index=False)
    print(f"Saved model outputs to {output_file}")