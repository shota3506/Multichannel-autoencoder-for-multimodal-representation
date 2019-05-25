import numpy as np
import os
from utils.utils import *


def load_word_vecs(file):
    word_vecs = {}
    with open(file) as f:
        for l in f:
            word, word_vec = l.strip().split(' ', 1)
            word_vecs[word] = np.array(word_vec.split(), dtype='float')
    return word_vecs


if __name__ == '__main__':
    # word_file = os.path.join('..', 'data', 'glove.840B.300d', 'glove.840B.300d.txt')
    word_file = os.path.join('..', 'result', 'GatedMultichannelAutoEncoder.txt')

    word_sim_files = ['men-3k.txt', 'simple-999.txt', 'sensim.txt', 'vissim.txt.', 'simverb-3500.txt',
                  'wordsim353.txt', 'wordrel353.txt', 'association.dev.txt', 'association.dev.b.txt']
    word_sim_file = os.path.join('..', 'evaluation', word_sim_files[0])

    print("Loading word vectors...")

    word_vecs = load_word_vecs(word_file)

    print("Calculating spearmans rho...")

    manual_dict, predict_dict = {}, {}

    not_found, total_size = (0, 0)
    for l in open(word_sim_file, 'r'):
        word1, word2, val = l.strip().lower().split()
        if word1 in word_vecs and word2 in word_vecs:
            manual_dict[(word1, word2)] = float(val)
            predict_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
        else:
            not_found += 1
        total_size += 1
    print("Total size: %d" % total_size)
    print("Not Found: %d" % not_found)
    print("Spearmans rho: %.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(predict_dict)))
