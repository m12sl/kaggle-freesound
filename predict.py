import argparse
import numpy as np
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
    with open(os.path.join(args.modeldir, 'hparams.json'), 'r') as fin:
        params = json.load(fin)

    with open(os.path.join(args.modeldir, 'vocab.json'), 'r') as fin:
        vocab = json.load(fin)
        vocab = {int(k): v for k, v in vocab.items()}

    hparams = tf.contrib.training.HParams(**params)
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=args.modeldir, session_config=session_config)

    model = base.create_model(config=run_config, hparams=hparams)

    df = pd.read_csv(os.path.join(args.datadir, 'sample_submission.csv'))
    df.label = 0
    df.fname = [
        os.path.join(args.datadir, 'audio_test', _)
        for _ in df.fname.values]

    test_input_fn = generator_input_fn(
        x=utils.fast_datagenerator(df, params, 'test'),
        batch_size=hparams.batch_size,
        shuffle=False,
        num_epochs=1,
        queue_capacity=hparams.batch_size,
        num_threads=1
    )
    it = model.predict(input_fn=test_input_fn)

    submission = dict()
    for t in tqdm(it):
        path = t['fname'].decode()
        fname = os.path.basename(path)
        predicted = vocab[t['prediction']]

        submission[fname] = predicted

    with open(os.path.join(args.modeldir, 'submission.csv'), 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in submission.items():
            fout.write(f"{fname},{label}\n")


if __name__ == "__main__":
    args = _parse_args()
    main(args)
