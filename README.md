# Kaggle Freesound Baseline Solution TF.Estimators

https://www.kaggle.com/c/freesound-audio-tagging


Suppose you have data:
```bash
/data/kaggle-freesound/
├── audio_test [9400 entries exceeds filelimit, not opening dir]
├── audio_test.zip
├── audio_train [9473 entries exceeds filelimit, not opening dir]
├── audio_train.zip
├── sample_submission.csv
└── train.csv
```


To train model use 
```bash
python main.py --datadir /your/data/path --outdir ./first-try/
```

To submission generation:

```bash
python predict.py --modeldir ./first-try
```





