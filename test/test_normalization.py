"""
Ensure that your custom layer produces the same (or very nearly the same)
output as the `keras.layers.LayerNormalization` layer.
"""

import unittest
import tensorflow as tf

import custom_layers

class TestLayerNormalization(unittest.TestCase):
    def test_custom_same_as_keras(self):
        "The custom lyer produces the same output as keras"

        precision = 1e-5

        shape = (32, 32, 3)
        k_layer = tf.keras.layers.Normalization(mean=42, variance=2.71)
        k_layer.build(shape)

        c_layer = custom_layers.Normalization()
        c_layer.build(shape)

        for _ in range(10):
            t = tf.random.uniform(shape)
            k_out = k_layer(t)
            c_out = c_layer(t)

            residual = tf.abs(k_out - c_out)
            tf.debugging.assert_less_equal(residual, precision)

if __name__ == "__main__":
    unittest.main()