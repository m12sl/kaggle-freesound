from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import signal
import numpy as np
from models import baseline


def model_handler(features, labels, mode, params, config):
    """
    :param features: выход очереди, по сути словарь {'img': np.array(...), 'fname': "hello.jpg", ...}
    :param labels: тензор с метками, их можно положить в features, но ради самых частых применений их вынесли отдельно
    :param mode: один из трех вариантов tf.estimator.ModeKeys.TRAIN/EVAL/PREDICT
    :param params: параметры модельки (количество слоев, learning_rate, ..., keep_prob)
    :param config: сейчас не используется
    :return: правильно заполненный tf.estimator.EstimatorSpec(**specs), см документацию и комменты в коде
    """
    # todo: добавьте сюда выбор модельки по параметру из params
    extractor = tf.make_template(
        'extractor', baseline,
        create_scope_now_=True,
    )

    wav = features['wav']  # здесь будет тензор [bs, timesteps]
    specgram = signal.stft(wav, 400, 160)  # здесь комплекснозначный тензор [bs, time_bins, freq_bins]

    phase = tf.angle(specgram) / np.pi
    amp = tf.log1p(tf.abs(specgram))

    x = tf.stack([amp, phase], axis=3)  # здесь почти обычная картинка  [bs, time_bins, freq_bins, 2]
    x = tf.to_float(x)

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)
    predictions = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        )

        # todo: обязательно попробуйте другие варианты изменния lr
        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.AdamOptimizer(lr),  # оптимизатор точно стоит потюнить
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
            # здесь можно пробрасывать что угодно
            'predictions': predictions,  # весь вектор предсказаний
            'prediction': tf.argmax(predictions, 1),  # топовая метка
            'fname': features['fname'],  # имя файла, удобный ход
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        # функция создающая из аргументов (features, labels, mode, ...) несколько версий графа в зависимости от mode
        model_fn=model_handler,
        # TF-специфичный конфиг, параметры сессии, конфигурация кластера, goto definition
        config=config,
        # параметры модельки или обучения
        params=hparams,
    )
