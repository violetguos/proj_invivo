import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors # can save

# Commit message
# this runs
# prints a numpy array vector

def load_data():
    """
    Loads the data csv file from the OS
    For now, return a dummy
    """
    dummy = [['C', '(','C' ,')''H', '3', 'N'],
    ['C', 'H', '(','C' ,')''H', '3', 'N']]
    return dummy

def embedding_train():
    """
    Train our own embedding
    """
    smiles_dummy_data = load_data()
    model = Word2Vec(sentences=smiles_dummy_data,
                        size=10, window=2, min_count=2, workers=2, sg=0)
    model.save('newmodel')

def main():
    # TODO ARGE PARSER

    embedding_test()

def embedding_test():
    """
    Test embedding
    """
    model = Word2Vec.load('newmodel')
    print(model.wv['C'])

if __name__ == '__main__':
    main()
