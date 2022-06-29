"""
Ensure that your custom layer produces the same (or very nearly the same)
output as the `keras.layers.LayerNormalization` layer.
"""

import unittest
import tensorflow as tf

import custom_layers

class TestLayerNormalization(unittest.TestCase):
    def test_custom_same_as_keras(self):
        "The custom layer produces the same output as keras"

        precision = 1e-5

        shape = (32, 32, 3)
        k_layer = tf.keras.layers.LayerNormalization()
        k_layer.build(shape)

        c_layer = custom_layers.Normalization()
        c_layer.build(shape)

        for _ in range(10):
            t = tf.random.uniform(shape)
            rand_alpha = tf.random.uniform(shape[-1:])
            rand_beta = tf.random.uniform(shape[-1:])

            c_layer.set_weights([rand_alpha, rand_beta])
            k_layer.set_weights([rand_alpha, rand_beta])

            k_out = k_layer(t)
            c_out = c_layer(t)

            residual = tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(k_out, c_out))

            tf.debugging.assert_less_equal(residual, precision)

if __name__ == "__main__":
    unittest.main()