import pandas as pd
import numpy as np


class DataLoader(object):

    def __init__(self):
        pass

    def load_dataset(self, dataset_path):
        '''return a pandas processed csv'''
        return pd.read_csv(dataset_path)

    def load_clean_words(self, clean_words_path):
        '''return a dict whose key is typo, value is correct word'''
        clean_word_dict = {}
        with open(clean_words_path, 'r', encoding='utf-8') as cl:
            for line in cl:
                line = line.strip('\n')
                typo, correct = line.split(',')
                clean_word_dict[typo] = correct
        return clean_word_dict

    def load_embedding(self, embedding_path, keras_like=True):
        '''return a dict whose key is word, value is pretrained word embedding'''
        if keras_like:
            embeddings_index = {}
            f = open(embedding_path, 'r', encoding='utf-8')
            for line in f:
                values = line.split()
                try:
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except:
                    print("Err on ", values[:2])
            f.close()
            print('Total %s word vectors.' % len(embeddings_index))
            return embeddings_index