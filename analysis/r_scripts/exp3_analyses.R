library("tidyverse")
library("lme4")
library("lmerTest")

# NOTE: Set your working directory to the root of this repo (shades-of-zero)

df <- read.csv("data/exp3_human_rt/critical_clean.csv")
output_dir <- "analysis/r_output"

# Test for difference in mean RT between condition pairs.
t.test(
  filter(df, condition=="inconceivable_semantic")$rt_normalized,
  filter(df, condition=="improbable")$rt_normalized,
  alternative="less"
)
t.test(
  filter(df, condition=="inconceivable_semantic")$rt_normalized,
  filter(df, condition=="impossible_physics")$rt_normalized,
  alternative="less"
)
t.test(
  filter(df, condition=="inconceivable_semantic")$rt_normalized,
  filter(df, condition=="impossible_magic")$rt_normalized,
  alternative="less"
)


for (model_name in c("gpt2", "Llama-2-7b-hf", "Mistral-7B-v0.1")) {
  # Read model surprisal data.
  model_df <- read.csv(
    sprintf("data/exp2_model_surprisal/surprisals/%s.csv", model_name)
  )
  
  # Combine human binary judgment data with model surprisal data.
  data <- merge(df, model_df, by=c("item_id", "condition"))
  
  # Fit judgment vs surprisal model.
  m <- glmer(
    response ~ continuation_sum_surprisal + (1 | subject_id) + (1 | item_id), 
    family=binomial(link="logit"),
    data=data
  )
  # Write summary to file.
  sink(sprintf("%s/exp3_%s_judgment_vs_surprisal.txt", output_dir, model_name))
  print(summary(m))
  sink()
  
  # Fit RT vs surprisal models.
  m.1 <- lm(
    rt_normalized ~ continuation_sum_surprisal,
    data=data
  )
  print(summary(m.1))
  m.2 <- lm(
    rt_normalized ~ poly(continuation_sum_surprisal, 2),
    data=data
  )
  # Write summary to file.
  sink(sprintf("%s/exp3_%s_rt_vs_surprisal.txt", output_dir, model_name))
  print(summary(m.1))
  print(cor.test(data$rt_normalized, data$continuation_sum_surprisal))
  print(summary(m.2))
  print(sprintf("BIC for first-order model: %f", BIC(m.1)))
  print(sprintf("BIC for second-order model: %f", BIC(m.2)))
  sink()
}

