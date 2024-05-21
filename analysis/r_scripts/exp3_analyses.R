library("tidyverse")
library("lme4")
library("lmerTest")

# NOTE: Set your working directory to the root of this repo (shades-of-zero)

df <- read.csv("data/exp3_human_rt/critical_clean.csv")
output_dir <- "analysis/r_output"

for (model_name in c("gpt2", "Llama-2-7b-hf", "Mistral-7B-v0.1")) {
  # Read model surprisal data.
  model_df <- read.csv(
    sprintf("data/exp2_model_surprisal/surprisals/%s.csv", model_name)
  )
  
  # Combine human binary judgment data with model surprisal data.
  data <- merge(df, model_df, by=c("item_id", "condition"))
  
  # Fit model.
  m <- glmer(
    response ~ continuation_sum_surprisal + (1 | subject_id) + (1 | item_id), 
    family=binomial(link="logit"),
    data=data
  )
  
  # Write summary to file.
  sink(sprintf("%s/%s_glmer_judgment_vs_surprisal.csv", output_dir, model_name))
  print(summary(m))
  sink()
}

