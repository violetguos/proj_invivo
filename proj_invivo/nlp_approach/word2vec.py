import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors  # can save
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import itertools
import pickle
sys.path.append("../../")
from proj_invivo.baseline.most_freq import impute_label


def tokenize(mode='char'):
    """converts a complete string representation in SMiles,
    split into list of charaters/tokens"""
    data, label = impute_label()
    if mode == 'char':
        data = [list(data[i]) for i in range(len(data))]
    else:
        data = list(data)
    return data, label


def embedding_train(smiles_data, label):
    """
    Train our own embedding
    """

    model = Word2Vec(sentences=smiles_data,
                     size=10, window=2, min_count=0, workers=2, sg=0)
    model.save('w2vmodel')


def embedding_test(data, label, plot_mode=False):
    """
    Test embedding, plot them in 2d
    """
    model = Word2Vec.load('w2vmodel')
    # find unique symbols
    symbols = [list(np.unique(np.array(data[i]))) for i in range(len(data))]
    unique_symbols = np.unique(np.array(list(itertools.chain(*symbols))))
    print("all unique symbols\n", unique_symbols)
    all_vectors = model.wv[unique_symbols]
    if plot_mode:
        pca = PCA(n_components=2)
        result = pca.fit_transform(all_vectors)
        # plotting function found on medium
        plt.scatter(result[:, 0], result[:, 1])
        words = unique_symbols
        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        plt.savefig('word2vec.jpg')
        plt.show()

    # save the trained vectors
    with open("word_vectors.pickle", 'wb') as jar:
        pickle.dump(all_vectors, jar, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    data, label = tokenize()
    embedding_train(data, label)
    embedding_test(data, label)


if __name__ == '__main__':
    main()
