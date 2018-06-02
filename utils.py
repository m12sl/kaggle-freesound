# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import os
import re


def get_new_model_path(path, suffix=''):
    numered_runs = []
    for x in os.listdir(path):
        r = re.match('(\d+)', x)
        if r:
            numered_runs.append((os.path.join(path, x), int(r.group())))

    numered_runs.sort(key=lambda t: t[1])
    if len(numered_runs) == 0:
        new_number = 0
    else:
        _, nums = zip(*numered_runs)
        new_number = nums[-1] + 1
    if suffix != '':
        suffix = '_' + suffix
    t = os.path.join(path, '{}{}'.format(new_number, suffix))
    os.mkdir(t)
    os.mkdir(os.path.join(t, 'eval'))
    return t


# обычный генератор
def datagenerator(df, params, mode='train'):
    idx = np.arange(len(df))

    def generator():
        if mode == 'train':
            np.random.shuffle(idx)
        for i in idx:
            fname = df.iloc[i].fname
            label = df.iloc[i].label
            try:
                # todo: добавить аугментаций
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max
                L = 8000
                if len(wav) < L:
                    continue
                beg = np.random.randint(0, len(wav) - L)

                # важно чтобы все значения в словаре были нумпайными
                yield dict(
                    target=np.int32(label),
                    wav=wav[beg: beg + L],
                    fname=np.string_(fname),  # <<< NB
                )

            except Exception as err:
                print(err, fname)

    return generator


# генератор с предзагрузкой датасета в память, требует памяти
def fast_datagenerator(df, params, mode='train'):
    def _read(fname):
        _, wav = wavfile.read(fname)
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        return wav

    idx = np.arange(len(df))
    data = [
        (df.iloc[i].fname, _read(df.iloc[i].fname), df.iloc[i].label)
        for i in tqdm(idx)]

    L = params.signal_len
    def generator():
        if mode == 'train':
            np.random.shuffle(idx)
        for i in idx:
            fname, wav, label = data[i]
            try:
                _wav = np.concatenate([np.zeros(L // 2, dtype=np.float32), wav, np.zeros(L // 2, dtype=np.float32)])
                if mode == 'test':
                    yield dict(
                        target=np.int32(label),
                        wav=wav,
                        fname=np.string_(fname),
                    )
                else:
                    beg = np.random.randint(0, len(_wav) - L)
                    _wav = _wav[beg: beg + L]

                    yield dict(
                        target=np.int32(label),
                        wav=_wav,
                        fname=np.string_(fname),
                    )

            except Exception as err:
                print(err, fname)

    return generator
