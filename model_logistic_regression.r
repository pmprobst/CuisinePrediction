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

# Get vocabulary from training data (cheaper than re-tokenizing train_df)
vocab <- train_tfidf$dimnames$Terms

# Create test TF-IDF matrix
test_tfidf <- test_df %>%
  unnest_tokens(word, text) %>%
  filter(word %in% vocab) %>%
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
  parallel = FALSE
)

best_lambda <- cv_glmnet$lambda.min
cat("Best lambda:", best_lambda, "\n")
cat("Approx. CV misclassification error at best lambda:", min(cv_glmnet$cvm), "\n")

# Train final model on all training data
cat("\nTraining final model on all training data...\n")
final_glmnet_model <- glmnet(
  x = train_matrix,
  y = train_labels,
  family = "multinomial",
  lambda = best_lambda,
  alpha = 1
)

# Make predictions on test data
cat("Making predictions on test data...\n")
test_predictions <- predict(final_glmnet_model, test_matrix, type = "class")
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
train_pred <- predict(final_glmnet_model, train_matrix, type = "class")
train_pred <- factor(train_pred[, 1], levels = levels(train_labels))
cat("Final model accuracy on training data:", mean(train_pred == train_labels), "\n")
