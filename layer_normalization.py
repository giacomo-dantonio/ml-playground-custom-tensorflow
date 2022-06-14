"""
Chapter 12 - Exercise 12

Implement a custom layer that performs *Layer normalization*:

a. The `build()` method should define two trainable weights **α** and
   **β**, both of shape `input_shape[-1]` and data type `tf.float32`.
   **α** should be initialized with 1s, and **β** with 0s.

b. The `call()` method should compute the mean *μ* and the standard deviation
   *σ* of each instance's features. For this you can use
   `tf.nn.moments(inputs, axes=-1, keepdims=True)`, which returns the
   mean *μ* and the variance *σ*² of all instances (compute the square
   root of the variance to get the standard deviation). Then the function
   should compute and return **α** ⊗ (**X** - *μ*) / (*σ* + *ε*) + **β**,
   where ⊗ represents itemwise multiplication (*) and *ε* is a smoothing
   term (small constant to avoid division by zero, e.g., 0.001).

c. Ensure that your custom layer produces the same (or very nearly the same)
   output as the `keras.layers.LyerNormalization` layer.
"""