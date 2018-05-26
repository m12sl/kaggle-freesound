import tensorflow as tf
from tensorflow.contrib import signal
import numpy as np
from models import baseline


def model_handler(features, labels, mode, params, config):
    extractor = tf.make_template(
        'extractor', baseline,
        create_scope_now_=True,
    )

    wav = features['wav']
    specgram = signal.stft(wav, 400, 160)

    phase = tf.angle(specgram) / np.pi
    amp = tf.log1p(tf.abs(specgram))

    x = tf.stack([amp, phase], axis=3)
    x = tf.to_float(x)

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)
    print(logits)
    predictions = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        )

        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.AdamOptimizer(lr),
            learning_rate_decay_fn=_learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        acc, acc_op = tf.metrics.accuracy(labels, tf.argmax(logits, axis=-1))
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        )
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                acc=(acc, acc_op),
            )
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': predictions,
            'prediction': tf.argmax(predictions, 1),
            'fname': features['fname'],
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=model_handler,
        config=config,
        params=hparams,
    )
