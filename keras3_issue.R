library(keras3)
reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters


## -------------------------------------------------------------------------
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in seq_along(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)


## -------------------------------------------------------------------------
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  labels <- labels + 1
  for(i in seq_along(labels)) {
    j <- labels[[i]]
    results[i, j] <- 1
  }
  results
}
y_train <- to_one_hot(train_labels)
y_test <- to_one_hot(test_labels)


## -------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(46, activation = "softmax")


sessioninfo::session_info()
reticulate::py_list_packages()
