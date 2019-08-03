import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors  # can save
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import itertools

sys.path.append("../../")
from proj_invivo.baseline.most_freq import impute_label



def tokenize(mode='char'):
    "converts a complete string representation in SMiles, split into list of charaters/tokens"
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

def main():
    data, label = tokenize()
    embedding_train(data, label)
    print("dta", len(data))
    embedding_test(data, label)


def embedding_test(data, label):
    """
    Test embedding
    """
    model = Word2Vec.load('w2vmodel')
    # find unique symbols
    symbols = [list(np.unique(np.array(data[i]))) for i in range(len(data))]
    unique_symbols = np.unique(np.array(list(itertools.chain(*symbols))))
    print(unique_symbols)
    X = model.wv[unique_symbols]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])
    words = unique_symbols
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.savefig('word2vec.jpg')
    plt.show()

if __name__ == '__main__':
    main()
