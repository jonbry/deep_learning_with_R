library(keras3)
library(tensorflow)

r_array <- array(1:6, c(2, 3))
tf_tensor <- as_tensor(r_array)
tf_tensor

methods(class = class(shape())[1])

as_tensor(0, shape = c(2, 3))

as_tensor(1:6, shape = c(2, 3))
array(1:6, dim= c(2,3))

array_reshape(1:6, c(2, 3), order = "C")

array_reshape(1:6, c(2, 3), order = "F")

as_tensor(1:6, shape = c(NA, 3))

x <- as_tensor(1, shape = c(64, 3, 32, 10))
y <- as_tensor(2, shape = c(32, 10))
z <- x + y

tf$ones(shape(1, 3))

tf$random$normal(shape(1, 3), mean = 0, stddev = 1)

tf$ones(c(2L, 1L)) 

x <- as_tensor(1 , shape = c(2, 2))
x[1, 1] <- 0

v <- tf$Variable(initial_value = tf$random$normal(shape(3, 1)))
v

v$assign(tf$ones(shape(3,1)))

with(tf$device('CPU'), {v[1, 1]$assign(3)})
v$assign_add(tf$ones(shape(3, 1)))
