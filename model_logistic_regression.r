# Multinomial Logistic Regression Model for Cuisine Prediction
# Uses TF-IDF features with L1 regularization and faster cross-validation

library(jsonlite)
library(tidyverse)
library(tidytext)
library(glmnet)


# Load data
train_data <- fromJSON("data/train.json", simplifyDataFrame = TRUE)
test_data <- fromJSON("data/test.json", simplifyDataFrame = TRUE)

# Prepare training data: create document-term matrix with TF-IDF
train_df <- train_data %>%
  mutate(text = map_chr(ingredients, ~paste(.x, collapse = " "))) %>%
  select(id, cuisine, text)

# Create TF-IDF features and select top 500 ingredients by TF-IDF score
train_tfidf_data <- train_df %>%
  unnest_tokens(word, text) %>%
  count(id, word, sort = TRUE) %>%
  bind_tf_idf(word, id, n)

# Calculate aggregate TF-IDF score per ingredient (sum across all documents)
ingredient_scores <- train_tfidf_data %>%
  group_by(word) %>%
  summarise(total_tfidf = sum(tf_idf), .groups = "drop") %>%
  arrange(desc(total_tfidf))

# Select top 500 ingredients
top_500_ingredients <- ingredient_scores %>%
  slice_head(n = 500) %>%
  pull(word)

cat("Selected top 500 ingredients by TF-IDF score\n")
cat("Total ingredients before filtering:", nrow(ingredient_scores), "\n")
cat("Ingredients after filtering:", length(top_500_ingredients), "\n")

# Filter to top 500 ingredients and cast to DTM
train_tfidf <- train_tfidf_data %>%
  filter(word %in% top_500_ingredients) %>%
  cast_dtm(id, word, tf_idf)

# Convert to matrix and align with cuisine labels
train_matrix <- as.matrix(train_tfidf)
# Ensure matrix is numeric
train_matrix <- matrix(as.numeric(train_matrix), nrow = nrow(train_matrix), ncol = ncol(train_matrix))
rownames(train_matrix) <- train_tfidf$dimnames$Docs
colnames(train_matrix) <- train_tfidf$dimnames$Terms

train_labels <- train_df$cuisine[match(rownames(train_matrix), train_df$id)]
train_labels <- factor(train_labels)

# Remove any rows with missing labels
valid_rows <- !is.na(train_labels)
train_matrix <- train_matrix[valid_rows, ]
train_labels <- train_labels[valid_rows]

# Prepare test data: create TF-IDF features using training vocabulary
test_df <- test_data %>%
  mutate(text = map_chr(ingredients, ~paste(.x, collapse = " "))) %>%
  select(id, text)

# Get vocabulary from training data (top 500 ingredients)
vocab <- train_tfidf$dimnames$Terms

# Create test TF-IDF matrix using only the top 500 ingredients
test_tfidf <- test_df %>%
  unnest_tokens(word, text) %>%
  filter(word %in% vocab) %>%
  count(id, word) %>%
  cast_dtm(id, word, n)

# Align test matrix columns with training matrix
test_matrix <- as.matrix(test_tfidf)
# Ensure matrix is numeric
test_matrix <- matrix(as.numeric(test_matrix), nrow = nrow(test_matrix), ncol = ncol(test_matrix))
rownames(test_matrix) <- test_tfidf$dimnames$Docs
colnames(test_matrix) <- test_tfidf$dimnames$Terms

# Ensure all test IDs are included (even if no matching ingredients)
all_test_ids <- test_data$id
test_ids_in_matrix <- as.integer(rownames(test_matrix))
missing_test_ids <- setdiff(all_test_ids, test_ids_in_matrix)

# Add rows for test IDs with no matching ingredients
if(length(missing_test_ids) > 0) {
  missing_rows <- matrix(0, nrow = length(missing_test_ids), ncol = ncol(test_matrix))
  rownames(missing_rows) <- as.character(missing_test_ids)
  colnames(missing_rows) <- colnames(test_matrix)
  test_matrix <- rbind(test_matrix, missing_rows)
}

# Add missing columns (set to 0) and reorder to match training
missing_cols <- setdiff(colnames(train_matrix), colnames(test_matrix))
if(length(missing_cols) > 0) {
  missing_matrix <- matrix(0, nrow = nrow(test_matrix), ncol = length(missing_cols))
  colnames(missing_matrix) <- missing_cols
  test_matrix <- cbind(test_matrix, missing_matrix)
}
test_matrix <- test_matrix[, colnames(train_matrix), drop = FALSE]

set.seed(42)

# Tune lambda parameter using glmnet's built-in cross-validation
cat("Tuning lambda parameter (fast CV)...\n")
cv_glmnet <- cv.glmnet(
  x = train_matrix,
  y = train_labels,
  family = "multinomial",
  type.measure = "class",
  nfolds = 3,  # fewer folds than before for speed
  alpha = 1,   # Lasso regularization
  parallel = FALSE,
  maxit = 100000  # increase max iterations
)

# Use lambda.1se for more stable/conservative regularization
best_lambda <- cv_glmnet$lambda.1se
cat("Lambda.1se (conservative):", best_lambda, "\n")
cat("Lambda.min (min CV error):", cv_glmnet$lambda.min, "\n")
cat("Approx. CV misclassification error at lambda.1se:", cv_glmnet$cvm[cv_glmnet$lambda == best_lambda], "\n")

# Use the model from cv.glmnet (already trained on full data with lambda sequence)
# This avoids convergence issues since cv.glmnet trains on the full dataset
cat("\nUsing model from cross-validation (already trained on full data)...\n")
final_glmnet_model <- cv_glmnet$glmnet.fit

# Make predictions on test data
cat("Making predictions on test data...\n")
test_predictions <- predict(final_glmnet_model, test_matrix, s = best_lambda, type = "class")
test_predictions <- factor(test_predictions[, 1], levels = levels(train_labels))

# Create submission dataframe
submission <- data.frame(
  id = as.integer(rownames(test_matrix)),
  cuisine = as.character(test_predictions)
)

# Ensure test IDs are in correct order (match original test data)
submission <- submission[match(test_data$id, submission$id), ]

# Save predictions
write_csv(submission, "predictions_logistic_regression.csv")
cat("\nPredictions saved to predictions_logistic_regression.csv\n")

# Calculate training accuracy
train_pred <- predict(final_glmnet_model, train_matrix, s = best_lambda, type = "class")
train_pred <- factor(train_pred[, 1], levels = levels(train_labels))
cat("Final model accuracy on training data:", mean(train_pred == train_labels), "\n")
