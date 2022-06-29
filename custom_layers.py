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
   output as the `keras.layers.LayerNormalization` layer.
"""

from numpy import dtype, float32
import tensorflow as tf

class Normalization(tf.keras.layers.Layer):
   def __init__(self, ε=1e-3, **kwargs):
      self.ε = ε
      super().__init__(**kwargs)

   def build(self, batch_input_shape):
      self.α = self.add_weight(
         name="α",
         shape=batch_input_shape[-1],
         dtype=tf.float32,
         initializer=tf.ones_initializer
      )
      self.β = self.add_weight(
         name="β",
         shape=batch_input_shape[-1],
         dtype=tf.float32,
         initializer=tf.zeros_initializer
      )

      # IMPORTANT: the super call must be at the end of the method
      super().build(batch_input_shape)

   def call(self, X):
      μ, σ = tf.nn.moments(X, axes=-1, keepdims=True)
      return self.α * (X - μ) / tf.sqrt(σ + self.ε) + self.β

   def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

   def get_config(self):
        base_config = super().get_config()
        return {**base_config, "ε": self.ε}

