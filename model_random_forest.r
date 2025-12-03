# Random Forest Model for Cuisine Prediction
# Uses TF-IDF features and cross-validation

library(jsonlite)
library(tidyverse)
library(tidytext)
library(randomForest)
library(caret)

# Load data
train_data <- fromJSON("data/train.json", simplifyDataFrame = TRUE)
test_data <- fromJSON("data/test.json", simplifyDataFrame = TRUE)

# Prepare training data: create document-term matrix with TF-IDF
train_df <- train_data %>%
  mutate(text = map_chr(ingredients, ~paste(.x, collapse = " "))) %>%
  select(id, cuisine, text)

# Create TF-IDF features
train_tfidf <- train_df %>%
  unnest_tokens(word, text) %>%
  count(id, word, sort = TRUE) %>%
  bind_tf_idf(word, id, n) %>%
  cast_dtm(id, word, tf_idf)

# Convert to matrix and align with cuisine labels
train_matrix <- as.matrix(train_tfidf)
train_labels <- train_df$cuisine[match(rownames(train_matrix), train_df$id)]
train_labels <- factor(train_labels)

# Prepare test data: create TF-IDF features using training vocabulary
test_df <- test_data %>%
  mutate(text = map_chr(ingredients, ~paste(.x, collapse = " "))) %>%
  select(id, text)

# Get vocabulary from training data
train_words <- train_df %>%
  unnest_tokens(word, text) %>%
  distinct(word) %>%
  pull(word)

# Create test TF-IDF matrix
test_tfidf <- test_df %>%
  unnest_tokens(word, text) %>%
  filter(word %in% train_words) %>%
  count(id, word) %>%
  cast_dtm(id, word, n)

# Align test matrix columns with training matrix
test_matrix <- as.matrix(test_tfidf)

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

# Cross-validation setup
set.seed(42)
cv_folds <- createFolds(train_labels, k = 5, returnTrain = FALSE)

# Perform cross-validation
cv_results <- list()
for(i in 1:length(cv_folds)) {
  cat("CV Fold", i, "of", length(cv_folds), "\n")
  
  train_idx <- unlist(cv_folds[-i])
  val_idx <- cv_folds[[i]]
  
  # Train model on fold
  rf_model <- randomForest(
    x = train_matrix[train_idx, ],
    y = train_labels[train_idx],
    ntree = 200,
    mtry = sqrt(ncol(train_matrix)),
    importance = TRUE
  )
  
  # Predict on validation set
  val_pred <- predict(rf_model, train_matrix[val_idx, ])
  val_accuracy <- mean(val_pred == train_labels[val_idx])
  
  cv_results[[i]] <- val_accuracy
  cat("Fold", i, "Accuracy:", val_accuracy, "\n")
}

# Print CV results
cat("\nCross-Validation Results:\n")
cat("Mean CV Accuracy:", mean(unlist(cv_results)), "\n")
cat("SD CV Accuracy:", sd(unlist(cv_results)), "\n")

# Train final model on all training data
cat("\nTraining final model on all training data...\n")
final_rf_model <- randomForest(
  x = train_matrix,
  y = train_labels,
  ntree = 200,
  mtry = sqrt(ncol(train_matrix)),
  importance = TRUE
)

# Make predictions on test data
cat("Making predictions on test data...\n")
test_predictions <- predict(final_rf_model, test_matrix)

# Create submission dataframe
submission <- data.frame(
  id = as.integer(rownames(test_matrix)),
  cuisine = as.character(test_predictions)
)

# Ensure test IDs are in correct order (match original test data)
submission <- submission[match(test_data$id, submission$id), ]

# Save predictions
write_csv(submission, "predictions_random_forest.csv")
cat("\nPredictions saved to predictions_random_forest.csv\n")
cat("Final model accuracy on training data:", mean(predict(final_rf_model) == train_labels), "\n")
