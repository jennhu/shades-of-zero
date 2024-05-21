library("tidyverse")
library("lme4")
library("lmerTest")
library("rstatix")

# NOTE: Set your working directory to the root of this repo (shades-of-zero)

output_dir <- "analysis/r_output"

# Read data containing surprisal and ngram values.
df <- read.csv("data/exp2_model_surprisal/surprisal_ngram_data.csv")
df$condition <- factor(
  df$condition, 
  levels=c("probable", "improbable", "impossible_physics", "impossible_magic", "inconceivable_semantic", "inconceivable_syntactic")
)

# Iterate over models.
for (model_name in unique(df$model)) {
  # Get data corresponding to this model.
  safe_model_name = sapply(strsplit(model_name, "/"), tail, 1)
  data = df %>% filter(model==model_name)

  # ANOVA: check if mean surprisals are different across conditions.
  m.aov <- data %>% anova_test(continuation_sum_surprisal ~ condition)
  write.csv(m.aov, sprintf("%s/%s_means_anova.csv", output_dir, safe_model_name))
  # Post-hoc pairwise comparisons.
  pwc <- data %>% tukey_hsd(continuation_sum_surprisal ~ condition)
  write.csv(pwc, sprintf("%s/%s_means_tukey.csv", output_dir, safe_model_name))
  
  # Prepare data to fit LMER models.
  # First, remove rows where log count is -inf.
  data = data %>% filter_if(~is.numeric(.), all_vars(!is.infinite(.)))
  # Next, backward difference coding of condition: 
  # https://stats.oarc.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/#backward
  backward.diff = matrix(c(
    -5/6, 1/6, 1/6, 1/6, 1/6, 1/6, 
    -4/6, -4/6, 2/6, 2/6, 2/6, 2/6,
    -3/6, -3/6, -3/6, 3/6, 3/6, 3/6,
    -2/6, -2/6, -2/6, -2/6, 4/6, 4/6,
    -1/6, -1/6, -1/6, -1/6, -1/6, 5/6
  ), ncol = 5)
  contrasts(data$condition) = backward.diff

  # Fit LMER model with *no* ngram predictor.
  m_no_ngram <- lmer(
    continuation_sum_surprisal ~ condition + (1 | item_id),
    data=data
  )
  sink(sprintf("%s/%s_lmer_no_ngram.csv", output_dir, safe_model_name))
  print(summary(m_no_ngram))
  sink()
  
  # Fit LMER model *with* ngram predictor.
  m_with_ngram <- lmer(
    continuation_sum_surprisal ~ condition + log_joint_count + (1 | item_id),
    data=data
  )
  sink(sprintf("%s/%s_lmer_with_ngram.csv", output_dir, safe_model_name))
  print(summary(m_with_ngram))
  sink()
  
  # Compare model 1 to model 2 using ANOVA.
  comparison <- anova(m_no_ngram, m_with_ngram)
  write.csv(
    comparison, 
    sprintf("%s/%s_lmer_comparison.csv", output_dir, safe_model_name)
  )
}

