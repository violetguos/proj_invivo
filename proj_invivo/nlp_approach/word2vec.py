import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors  # can save
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

sys.path.append("../../")
from proj_invivo.baseline.most_freq import impute_label
# Commit message
# this runs
# prints a numpy array vector


def tokenize(mode='char'):
    "converts a complete string representation in SMiles, split into list of charaters/tokens"
    data, label = impute_label()
    if mode == 'char':
        data = [list(data[i]) for i in range(len(data))]
    else:
        data = list(data)
    return data, label


def embedding_train():
    """
    Train our own embedding
    """

    smiles_data, label = tokenize()
    model = Word2Vec(sentences=smiles_data,
                     size=10, window=2, min_count=2, workers=2, sg=0)
    model.save('w2vmodel')
    return smiles_data, label

def main():
    # tokenize()
    data, label = embedding_train()
    print(data[0])
    embedding_test(data, label)


def embedding_test(data, label):
    """
    Test embedding
    """
    model = Word2Vec.load('w2vmodel')
    unique_symbols = np.unique(np.unique(np.array(data[0])))
    X = model.wv[unique_symbols]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])
    words = unique_symbols
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()

if __name__ == '__main__':
    main()
