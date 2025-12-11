# Improved Multinomial Logistic Regression Model for Cuisine Prediction
# Implements enhanced feature engineering, preprocessing, and model improvements
# Based on best practices from Kaggle "What's Cooking" competition solutions

library(jsonlite)
library(tidytext)
library(glmnet)
library(caret)
library(stringr)  # Explicitly load stringr for text processing

my_tidyverse <- function() {
  pkgs <- c(
    "ggplot2", "dplyr", "tidyr", "readr",
    "purrr", "tibble", "stringr", "forcats"
  )
  invisible(lapply(pkgs, library, character.only = TRUE))
}

my_tidyverse()


# ============================================================================
# 1. PREPROCESSING: Better ingredient normalization and cleaning
# ============================================================================

normalize_ingredients <- function(ingredients) {
  # Normalize ingredient text: lowercase, remove special characters, trim whitespace
  ingredients %>%
    tolower() %>%
    str_replace_all("[^a-z0-9\\s]", "") %>%  # Remove special characters except spaces
    str_replace_all("\\s+", " ") %>%         # Replace multiple spaces with single space
    str_trim() %>%                            # Trim leading/trailing whitespace
    .[. != ""]                                # Remove empty strings
}

# ============================================================================
# 2. FEATURE ENGINEERING: Create both binary and TF-IDF features
# ============================================================================

# Load data
cat("Loading data...\n")
train_data <- fromJSON("data/train.json", simplifyDataFrame = TRUE)
test_data <- fromJSON("data/test.json", simplifyDataFrame = TRUE)

# Apply preprocessing to ingredients
cat("Preprocessing ingredients...\n")
train_data <- train_data %>%
  mutate(
    ingredients_clean = map(ingredients, normalize_ingredients),
    text = map_chr(ingredients_clean, ~paste(.x, collapse = " "))
  ) %>%
  select(id, cuisine, text, ingredients_clean)

test_data <- test_data %>%
  mutate(
    ingredients_clean = map(ingredients, normalize_ingredients),
    text = map_chr(ingredients_clean, ~paste(.x, collapse = " "))
  ) %>%
  select(id, text, ingredients_clean)

# Create TF-IDF features
cat("Creating TF-IDF features...\n")
train_tfidf_data <- train_data %>%
  unnest_tokens(word, text) %>%
  count(id, word, sort = TRUE) %>%
  bind_tf_idf(word, id, n)

# Calculate ingredient importance scores (aggregate TF-IDF)
ingredient_scores <- train_tfidf_data %>%
  group_by(word) %>%
  summarise(
    total_tfidf = sum(tf_idf),
    doc_freq = n_distinct(id),
    .groups = "drop"
  ) %>%
  arrange(desc(total_tfidf))

# Filter out very rare ingredients (appear in < 2 documents) and very common ones
# This helps with feature selection and reduces noise
ingredient_scores_filtered <- ingredient_scores %>%
  filter(doc_freq >= 2, doc_freq <= nrow(train_data) * 0.95)  # Remove very rare and very common

cat("Total unique ingredients:", nrow(ingredient_scores), "\n")
cat("After filtering rare/common:", nrow(ingredient_scores_filtered), "\n")

# Select top ingredients by TF-IDF score
top_n_ingredients <- 500
top_ingredients <- ingredient_scores_filtered %>%
  slice_head(n = top_n_ingredients) %>%
  pull(word)

cat("Selected top", length(top_ingredients), "ingredients by TF-IDF score\n")

# Create TF-IDF matrix for training
train_tfidf <- train_tfidf_data %>%
  filter(word %in% top_ingredients) %>%
  cast_dtm(id, word, tf_idf)

train_tfidf_matrix <- as.matrix(train_tfidf)
train_tfidf_matrix <- matrix(as.numeric(train_tfidf_matrix), 
                             nrow = nrow(train_tfidf_matrix), 
                             ncol = ncol(train_tfidf_matrix))
rownames(train_tfidf_matrix) <- train_tfidf$dimnames$Docs
colnames(train_tfidf_matrix) <- train_tfidf$dimnames$Terms

# Create binary features (ingredient presence/absence) for training
cat("Creating binary features...\n")
train_binary_data <- train_data %>%
  unnest_tokens(word, text) %>%
  filter(word %in% top_ingredients) %>%
  distinct(id, word) %>%
  mutate(present = 1)

train_binary <- train_binary_data %>%
  cast_dtm(id, word, present)

train_binary_matrix <- as.matrix(train_binary)
train_binary_matrix[train_binary_matrix > 0] <- 1  # Ensure binary
rownames(train_binary_matrix) <- train_binary$dimnames$Docs
colnames(train_binary_matrix) <- train_binary$dimnames$Terms

# Combine TF-IDF and binary features
# Strategy: Use TF-IDF as primary features, add binary indicators as additional features
cat("Combining TF-IDF and binary features...\n")

# Align matrices to have same ingredients
common_ingredients <- intersect(colnames(train_tfidf_matrix), colnames(train_binary_matrix))
train_tfidf_matrix <- train_tfidf_matrix[, common_ingredients, drop = FALSE]
train_binary_matrix <- train_binary_matrix[, common_ingredients, drop = FALSE]

# Ensure both matrices have same row order
common_rows <- intersect(rownames(train_tfidf_matrix), rownames(train_binary_matrix))
train_tfidf_matrix <- train_tfidf_matrix[common_rows, , drop = FALSE]
train_binary_matrix <- train_binary_matrix[common_rows, , drop = FALSE]

# Create binary features with prefix to distinguish from TF-IDF
binary_colnames <- paste0("bin_", colnames(train_binary_matrix))
colnames(train_binary_matrix) <- binary_colnames

# Combine matrices: TF-IDF features first, then binary features
train_combined_matrix <- cbind(train_tfidf_matrix, train_binary_matrix)

# Align with labels
train_labels <- train_data$cuisine[match(rownames(train_combined_matrix), train_data$id)]
train_labels <- factor(train_labels)

# Remove rows with missing labels
valid_rows <- !is.na(train_labels)
train_combined_matrix <- train_combined_matrix[valid_rows, ]
train_labels <- train_labels[valid_rows]

cat("Final training matrix dimensions:", dim(train_combined_matrix), "\n")
cat("Number of cuisines:", nlevels(train_labels), "\n")

# ============================================================================
# Prepare test data with same preprocessing
# ============================================================================

cat("\nPreparing test data...\n")
test_df <- test_data %>%
  select(id, text)

# Create test TF-IDF matrix
test_tfidf <- test_df %>%
  unnest_tokens(word, text) %>%
  filter(word %in% top_ingredients) %>%
  count(id, word) %>%
  cast_dtm(id, word, n)

test_tfidf_matrix <- as.matrix(test_tfidf)
test_tfidf_matrix <- matrix(as.numeric(test_tfidf_matrix), 
                            nrow = nrow(test_tfidf_matrix), 
                            ncol = ncol(test_tfidf_matrix))
rownames(test_tfidf_matrix) <- test_tfidf$dimnames$Docs
colnames(test_tfidf_matrix) <- test_tfidf$dimnames$Terms

# Create test binary matrix
test_binary <- test_df %>%
  unnest_tokens(word, text) %>%
  filter(word %in% top_ingredients) %>%
  distinct(id, word) %>%
  mutate(present = 1) %>%
  cast_dtm(id, word, present)

test_binary_matrix <- as.matrix(test_binary)
test_binary_matrix[test_binary_matrix > 0] <- 1
rownames(test_binary_matrix) <- test_binary$dimnames$Docs
colnames(test_binary_matrix) <- test_binary$dimnames$Terms

# Align test matrices with training vocabulary
# TF-IDF features - ensure same columns as training
missing_tfidf_cols <- setdiff(colnames(train_tfidf_matrix), colnames(test_tfidf_matrix))
if(length(missing_tfidf_cols) > 0) {
  missing_matrix <- matrix(0, nrow = nrow(test_tfidf_matrix), ncol = length(missing_tfidf_cols))
  colnames(missing_matrix) <- missing_tfidf_cols
  test_tfidf_matrix <- cbind(test_tfidf_matrix, missing_matrix)
}
# Reorder to match training
test_tfidf_matrix <- test_tfidf_matrix[, colnames(train_tfidf_matrix), drop = FALSE]

# Binary features - ensure same columns as training
test_binary_colnames <- paste0("bin_", colnames(test_binary_matrix))
colnames(test_binary_matrix) <- test_binary_colnames

missing_binary_cols <- setdiff(binary_colnames, test_binary_colnames)
if(length(missing_binary_cols) > 0) {
  missing_matrix <- matrix(0, nrow = nrow(test_binary_matrix), ncol = length(missing_binary_cols))
  colnames(missing_matrix) <- missing_binary_cols
  test_binary_matrix <- cbind(test_binary_matrix, missing_matrix)
}
# Reorder to match training
test_binary_matrix <- test_binary_matrix[, binary_colnames, drop = FALSE]

# Combine test matrices in same order as training
test_combined_matrix <- cbind(test_tfidf_matrix, test_binary_matrix)

# Ensure all test IDs are included
all_test_ids <- test_data$id
test_ids_in_matrix <- as.integer(rownames(test_combined_matrix))
missing_test_ids <- setdiff(all_test_ids, test_ids_in_matrix)

if(length(missing_test_ids) > 0) {
  missing_rows <- matrix(0, nrow = length(missing_test_ids), ncol = ncol(test_combined_matrix))
  rownames(missing_rows) <- as.character(missing_test_ids)
  colnames(missing_rows) <- colnames(test_combined_matrix)
  test_combined_matrix <- rbind(test_combined_matrix, missing_rows)
}

cat("Final test matrix dimensions:", dim(test_combined_matrix), "\n")

# ============================================================================
# 3. MODEL IMPROVEMENTS: Elastic Net with better cross-validation
# ============================================================================

set.seed(42)

cat("\n=== MODEL TRAINING ===\n")

# Try multiple alpha values (0 = Ridge, 0.5 = Elastic Net, 1 = Lasso)
alpha_values <- c(0.5, 1.0)  # Try elastic net and lasso
best_alpha <- NULL
best_cv_error <- Inf
best_cv_model <- NULL

cat("Tuning alpha parameter (elastic net vs lasso)...\n")
for(alpha in alpha_values) {
  cat("\nTesting alpha =", alpha, "\n")
  
  # Use cross-validation with 3 folds for faster training
  cv_glmnet <- cv.glmnet(
    x = train_combined_matrix,
    y = train_labels,
    family = "multinomial",
    type.measure = "class",
    nfolds = 3,              # 3 folds for faster training
    alpha = alpha,           # Elastic net (0.5) or Lasso (1.0)
    parallel = FALSE,
    maxit = 100000,
    standardize = TRUE       # Standardize features
  )
  
  cv_error <- min(cv_glmnet$cvm)
  cat("  Min CV error:", cv_error, "\n")
  cat("  Lambda.min:", cv_glmnet$lambda.min, "\n")
  cat("  Lambda.1se:", cv_glmnet$lambda.1se, "\n")
  
  if(cv_error < best_cv_error) {
    best_cv_error <- cv_error
    best_alpha <- alpha
    best_cv_model <- cv_glmnet
  }
}

cat("\n=== BEST MODEL SELECTED ===\n")
cat("Best alpha:", best_alpha, "\n")
cat("Best CV error:", best_cv_error, "\n")
cat("Lambda.min:", best_cv_model$lambda.min, "\n")
cat("Lambda.1se:", best_cv_model$lambda.1se, "\n")

# Use lambda.1se for more stable/conservative regularization
best_lambda <- best_cv_model$lambda.1se
final_glmnet_model <- best_cv_model$glmnet.fit

cat("\nUsing lambda.1se =", best_lambda, "for final model\n")

# ============================================================================
# Make predictions
# ============================================================================

cat("\n=== MAKING PREDICTIONS ===\n")

# Training accuracy
train_pred <- predict(final_glmnet_model, train_combined_matrix, s = best_lambda, type = "class")
train_pred <- factor(train_pred[, 1], levels = levels(train_labels))
train_accuracy <- mean(train_pred == train_labels)
cat("Training accuracy:", train_accuracy, "\n")

# Test predictions
cat("Making predictions on test data...\n")
test_predictions <- predict(final_glmnet_model, test_combined_matrix, s = best_lambda, type = "class")
test_predictions <- factor(test_predictions[, 1], levels = levels(train_labels))

# Create submission dataframe
submission <- data.frame(
  id = as.integer(rownames(test_combined_matrix)),
  cuisine = as.character(test_predictions)
)

# Ensure test IDs are in correct order
submission <- submission[match(test_data$id, submission$id), ]

# Save predictions
output_file <- "predictions_logistic_regression_improved.csv"
write_csv(submission, output_file)
cat("\nPredictions saved to", output_file, "\n")

# ============================================================================
# Summary
# ============================================================================

cat("\n=== MODEL SUMMARY ===\n")
cat("Alpha (regularization type):", best_alpha, "\n")
cat("Lambda (regularization strength):", best_lambda, "\n")
cat("Number of features:", ncol(train_combined_matrix), "\n")
cat("Training accuracy:", train_accuracy, "\n")
cat("Cross-validation error:", best_cv_error, "\n")
