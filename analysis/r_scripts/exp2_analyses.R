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
  levels=c("probable", "improbable", 
           "impossible_physics", "impossible_magic", 
           "inconceivable_semantic", "inconceivable_syntactic")
)

# Iterate over models.
for (model_name in unique(df$model)) {
  # Get data corresponding to this model.
  safe_model_name = sapply(strsplit(model_name, "/"), tail, 1)
  data = df %>% filter(model==model_name)
  
  # Prepare data to fit LMER models.
  # First, remove rows where log count is -inf.
  data = data %>% filter_if(~is.numeric(.), all_vars(!is.infinite(.)))
  # Next, backward difference coding of condition: 
  # https://stats.oarc.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/#backward
  backward.diff = matrix(c(
    -5/6, 1/6, 1/6, 1/6, 1/6, 1/6, # 1: improbable vs probable
    -4/6, -4/6, 2/6, 2/6, 2/6, 2/6, # 2: impossible_physics vs improbable
    -3/6, -3/6, -3/6, 3/6, 3/6, 3/6, # 3: impossible_magic vs impossible_physics
    -2/6, -2/6, -2/6, -2/6, 4/6, 4/6, # 4: inconceivable_semantic vs impossible_magic
    -1/6, -1/6, -1/6, -1/6, -1/6, 5/6 # 5: inconceivable_syntactic vs inconceivable_semantic
  ), ncol = 5)
  contrasts(data$condition) = backward.diff
  
  # Fit LMER model with ngram predictor.
  m <- lmer(
    continuation_sum_surprisal ~ condition + log_joint_count + (1 | item_id),
    data=data
  )
  sink(sprintf("%s/exp2_%s_surprisal_vs_condition.txt", output_dir, safe_model_name))
  print(summary(m))
  sink()
}

