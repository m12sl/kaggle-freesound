import argparse
import tensorflow as tf
import os
import pandas as pd
from tqdm import tqdm
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
import json

import base
import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default='./tmp/')
    parser.add_argument('--datadir',
                        type=str, default='/data/kaggle-freesound/')
    return parser.parse_args()


def main(args):
    # восстанавливаем сохраненые конфиги и словарь
    with open(os.path.join(args.modeldir, 'hparams.json'), 'r') as fin:
        params = json.load(fin)

    with open(os.path.join(args.modeldir, 'vocab.json'), 'r') as fin:
        vocab = json.load(fin)
        vocab = {int(k): v for k, v in vocab.items()}

    hparams = tf.contrib.training.HParams(**params)
    # все тот же костыль для некоторых машин
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=args.modeldir, session_config=session_config)

    # создаем модельку
    model = base.create_model(config=run_config, hparams=hparams)

    # готовим данные для теста из sample_submission
    df = pd.read_csv(os.path.join(args.datadir, 'sample_submission.csv'))
    df.label = 0
    df.fname = [
        os.path.join(args.datadir, 'audio_test', _)
        for _ in df.fname.values]

    # predict все равно работает по одному примеру, так что давайте уберем батчи
    # так мы сможем работать с записями целиком
    # NB: стоит проверить, правильно ли работает pad_value
    test_input_fn = generator_input_fn(
        x=utils.fast_datagenerator(df, params, 'test'),
        batch_size=1,
        shuffle=False,
        num_epochs=1,
        queue_capacity=hparams.batch_size,
        num_threads=1,
        pad_value=0.0,
    )

    it = model.predict(input_fn=test_input_fn)  # это итератор

    # далее немного грязно, отрефакторите, добавьте информацию о фолдах, если нужно
    submission = dict()
    for output in tqdm(it):
        path = output['fname'].decode()
        fname = os.path.basename(path)
        # допускается предсказывать три метки на каждую запись
        predicted = " ".join([vocab[i] for i in output['top3']])
        submission[fname] = predicted

    with open(os.path.join(args.modeldir, 'submission.csv'), 'w') as fout:
        fout.write('fname,label\n')
        for fname, pred in submission.items():
            fout.write("{},{}\n".format(fname, pred))


if __name__ == "__main__":
    args = _parse_args()
    main(args)
