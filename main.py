import argparse
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
import json

import base
import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./tmp/')
    parser.add_argument('--datadir',
                        type=str, default='/data/kaggle-freesound/')
    parser.add_argument('--model',
                        type=str, default='baseline', choices=["baseline", ])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--clip_gradients', type=float, default=15.0)
    parser.add_argument('--seed', type=int, default=2018)
    return parser.parse_args()


def main(args):
    os.makedirs(os.path.join(args.outdir, 'eval'), exist_ok=True)

    df = pd.read_csv(os.path.join(args.datadir, 'train.csv'))
    labels = sorted(set(df.label.values))

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    with open(os.path.join(args.outdir, 'vocab.json'), 'w') as fout:
        json.dump(id2label, fout, indent=2)

    df['label'] = [label2id[_] for _ in df.label.values]
    df['fname'] = [
        os.path.join(args.datadir, 'audio_train', _) for _ in df.fname.values]

    idx = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.33, random_state=2018, shuffle=True)
    df_train, df_val = df.iloc[idx_train], df.iloc[idx_val]

    # todo: get it from args
    params = {'num_classes': len(labels), **args.__dict__}

    hparams = tf.contrib.training.HParams(**params)

    with open(os.path.join(args.outdir, 'hparams.json'), 'w') as fout:
        json.dump(params, fout, indent=2)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=args.outdir, session_config=session_config)

    train_input_fn = generator_input_fn(
        x=utils.fast_datagenerator(df_train, hparams, 'train'),
        target_key='target',
        batch_size=hparams.batch_size,
        shuffle=True,
        num_epochs=10,
        queue_capacity=3 * hparams.batch_size,
        num_threads=1,
    )

    val_input_fn = generator_input_fn(
        x=utils.fast_datagenerator(df_val, hparams, 'val'),
        target_key='target',
        batch_size=hparams.batch_size,
        shuffle=False,
        num_epochs=None,
        queue_capacity=3 * hparams.batch_size,
        num_threads=1,
    )

    est = base.create_model(config=run_config, hparams=hparams)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn)

    tf.estimator.train_and_evaluate(est, train_spec, eval_spec)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
