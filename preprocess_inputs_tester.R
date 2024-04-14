library(keras3)
library(tensorflow)
library(tfdatasets)
library(reticulate)


conv_base <- application_vgg16(weights = "imagenet",
                               include_top = FALSE)
freeze_weights(conv_base)

# Cases
r_array <- array(dim = c(100, 100, 3), sample(0:255, replace = T))

r_array_to_tensor <- as_tensor(r_array)

r_array_loop <- as.array(r_array_to_tensor)

tensor <- tf$ones(shape(100, 100, 3))

tensor_to_r_array <- as.array(tensor)

tensor_int <- tf$ones(shape(100, 100, 3), dtype = "int32")

tensor_int_to_r_array <- as.array(tensor_int)



# Processed tensors/arrays
processed_r_array <- application_preprocess_inputs(conv_base, r_array)

processed_r_array_to_tensor <- application_preprocess_inputs(conv_base, r_array_to_tensor)

processed_r_array_loop <- application_preprocess_inputs(conv_base, r_array_loop)

processed_tensor <- application_preprocess_inputs(conv_base, tensor)

processed_tensor_to_r_array <- application_preprocess_inputs(conv_base, tensor_to_r_array)

processed_tensor_int_to_r_array <- application_preprocess_inputs(conv_base, tensor_int_to_r_array)

# Types
typeof(r_array)
typeof(r_array_loop)
typeof(tensor_to_r_array)
typeof(tensor_int)
typeof(tensor_int_to_r_array)

# Shapes
r_array_to_tensor$dtype
tensor$dtype
tensor_int$dtype

# error seems to come from converting tensors to R arrays due to the default type being float64.
# application_preprocess_inputs() doesn't seem to like doubles, which is what happens when converting
# a float64 tensor to an R array. This can be solved by using the dtype argument, but doesn't seem to be an
# option with image_dataset_from_directory(). 