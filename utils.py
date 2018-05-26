import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def datagenerator(df, params, mode='train'):
    idx = np.arange(len(df))

    def generator():
        if mode == 'train':
            np.random.shuffle(idx)
        for i in idx:
            fname = df.iloc[i].fname
            label = df.iloc[i].label
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max
                L = 8000
                if len(wav) < L:
                    continue
                beg = np.random.randint(0, len(wav) - L)

                yield dict(
                    target=np.int32(label),
                    wav=wav[beg: beg + L],
                    fname=np.string_(fname),
                )

            except Exception as err:
                print(err, fname)

    return generator


def fast_datagenerator(df, params, mode='train'):
    def _read(fname):
        _, wav = wavfile.read(fname)
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        return wav

    idx = np.arange(len(df))
    data = [
        (df.iloc[i].fname, _read(df.iloc[i].fname), df.iloc[i].label)
        for i in tqdm(idx)]

    def generator():
        if mode == 'train':
            np.random.shuffle(idx)
        for i in idx:
            fname, wav, label = data[i]
            try:
                L = 8000
                if len(wav) < L:
                    continue
                beg = np.random.randint(0, len(wav) - L)

                yield dict(
                    target=np.int32(label),
                    wav=wav[beg: beg + L],
                    fname=np.string_(fname),
                )

            except Exception as err:
                print(err, fname)

    return generator
