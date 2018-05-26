import tensorflow as tf
from tensorflow.contrib import layers


def baseline(x, params, is_training):
    afn = dict(
        normalizer_fn=layers.batch_norm,
        normalizer_params=dict(is_training=is_training),
    )

    for i in range(3):
        x = layers.conv2d(x, 16, 5, **afn)
        x = layers.max_pool2d(x, 2, 2)

    gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    # вместо полносвязного слоя удобно взять свертку 1х1 на нужное количество классов
    x = tf.layers.conv2d(gap, params.num_classes, 1, activation=None)

    return tf.squeeze(x, [1, 2])
