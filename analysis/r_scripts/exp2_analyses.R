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
  levels=c(
    "probable", 
    "improbable", 
    "impossible", 
    "inconceivable", 
    "inconceivable_syntactic")
)

# Iterate over models.
for (model_name in unique(df$model)) {
  # Get data corresponding to this model.
  safe_model_name = sapply(strsplit(model_name, "/"), tail, 1)
  data = df %>% filter(model==model_name)
  
  # Prepare data to fit LMER models.
  # Backward difference coding of condition: 
  # https://stats.oarc.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/#backward
  backward.diff = matrix(c(
    -4/5, 1/5, 1/5, 1/5, 1/5, # 1: improbable vs probable
    -3/5, -3/5, 2/5, 2/5, 2/5, # 2: impossible vs improbable
    -2/5, -2/5, -2/5, 3/5, 3/5, # 3: inconceivable vs impossible
    -1/5, -1/5, -1/5, -1/5, 4/5 # 4: inconceivable_syntactic vs inconceivable
  ), ncol = 4)
  contrasts(data$condition) = backward.diff
  
  # Fit LMER model with ngram predictor.
  m <- lmer(
    continuation_mean_surprisal ~ condition + log_count + (1 | item_id),
    data=data
  )
  sink(sprintf("%s/exp2_%s_surprisal_vs_condition.txt", output_dir, safe_model_name))
  print(summary(m))
  sink()
}

