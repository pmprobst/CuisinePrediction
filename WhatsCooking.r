library(jsonlite)
library(tidytext)
library(tidyverse)


train_data <- fromJSON("data/train.json", simplifyDataFrame = TRUE)
test_data <- fromJSON("data/test.json", simplifyDataFrame = TRUE)
head(train_data)

# Unnest ingredients and save to CSV
train_unnested <- train_data %>% 
    unnest(ingredients)

test_unnested <- test_data %>%
    unnest(ingredients)

# Write to CSV files
write_csv(train_unnested, "data/train_unnested.csv")
write_csv(test_unnested, "data/test_unnested.csv")