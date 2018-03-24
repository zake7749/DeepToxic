# DeepToxic

This is part of 27th solution for the [toxic comment classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/). For easy understanding, I only uploaded what I used in the final stage, and did not attach any experimental or deprecated codes.

## Dataset and External pretrained embeddings

You can fetch the dataset [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). I used 3 kind of word embeddings:

* [FastText crawl 300d 2M](https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m)
* [glove.840B.300d](https://nlp.stanford.edu/projects/glove/) 
* [glove.twitter.27B](https://nlp.stanford.edu/projects/glove/)

## Overview

### Preprocessing

We trained our models on 3 datasets with different preprocessing:

* original dataset with spellings correction: by comparing the Levenshtein distance and a lot of regular expressions.
* original dataset with pos taggings:  We generate the part of speech (POS) tagging for every comment by TextBlob and concatenate the word embedding and POS embedding as a single one. Since TextBlob drops some tokens and punctuations when generating the POS sequences, that gives our models another view. 
* Riad's dataset: with very heavily data-cleaning, spelling correction and translation

### Models

In our case, the simpler, the better. I tried some complicated structures (RHN, DPCNN, HAN). Most of them had performed very well locally but got lower AUC on the leaderboard. The models I kept trying during the final stage are the following two:

Pooled RNN (public: 0.9862, private: 0.9858)
![pooledRNN](https://i.imgur.com/AQkbPn7.png)

Kmax text CNN (public: 0.9856 , private: 0.9849)
![kmaxCNN](https://i.imgur.com/WfbXVh3.png)

As many competitors pointed out, dropout and batch-normalization are the keys to prevent overfitting. By applying the dropout on the word embedding directly and behind the pooling does great regularization both on train set and test set. Although model with many dropouts takes about 5 more epochs to coverage, it boosts our scores significantly. For instance, my RNN boosts from 0.9853 (private: 0.9850) to 0.9862 (private: 0.9858) after adding dropout layers.

For maximizing the utility of these datasets, besides training on the original labels, we also add a meta-label "bad_comment". If a comment is labeled, then it's considered to be a bad comment. The hypothesizes between these two labels sets are slightly different but with almost the same LB score, which leaves us room for the ensemble.

In order to increase the diversity and to deal with some toxic typos, we trained the models both on char-level and word-level. The results of char-level perform a bit worse (for charRNN: 0.983 on LB, 0.982 on PB, charCNN: 0.9808 on LB, 0.9801 on PB) but it does have a pretty low correlation with word-level models. By simply bagging my char-level and word-level result, it is good enough to push me over 0.9869 in the private test set. By the way, the hyperparameters influence the performance hugely in the char-based models. A large batch size (256), very long sequence length (1000) would ordinarily get a considerable result even though it takes much time for the K-fold validation. (my char-based models usually converge after 60~70 epochs which is about 5 times more than my word-based models.)

## Performance of Single models

Scored by AUC on the private testset.

### Word level

|Model|Fasttext|Glove|Twitter|
|-----|--------|-----|-------|
|AVRNN|0.9858|0.9855|0.9843|
|Meta-AVRNN|0.9850|0.9849|No data|
|Pos-AVRNN|0.9850|No data|0.9841|
|AVCNN|0.9846|0.9845|0.9841|
|Meta-AVCNN|0.9844|0.9844|No data|
|Pos-AVCNN|0.9850|No data|No data|
|KmaxTextCNN|0.9849|0.9845|0.9835|
|TextCNN|0.9837|No data|No data|
|RCNN|0.9847|0.9842|0.9832|
|RHN|0.9842|No data|No data|

### Char level


|Model|AUC|
|-----|------|
|AVRNN|0.9821|
|KmaxCNN|0.9801|
|AVCNN|0.9797|

