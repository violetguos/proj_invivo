import numpy as np
from sklearn.dummy import DummyClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
import proj_invivo.utils.config as cfg

def load_data(data_path='../../data.csv', test_mode=True):
    """
    Returns the smiles column of data csv file and the labels
    """
    data_reader = pd.read_csv(data_path)
    data = data_reader['smiles']
    # print("data", type(data))

    label = data_reader.iloc[:, 1:cfg.NUM_TARGET]
    # print("labels", label)

    if test_mode:
        data = data[0:10]
        label = label[0:10]

    return data, label


def impute_label(data_path='../../data.csv', plot_mode=False):
    """
    Find the indices of the most correlated target other than itself
    Return as numpy array
    """
    data, label = load_data(data_path, test_mode=False)
    corr_mat = label.corr().values
    if plot_mode:
        plt.matshow(label.corr())
        plt.savefig('correlation.jpg')

        plt.show()
        plt.title("correlation of targets")

    correlated_indices = []
    label = label.values
    data = data.values
    for i in range(corr_mat.shape[1]):
        # -2: find the largests 2
        # -1: maximum first
        # 1: get the second largest since correlation of one with one's self
        # is always 100%, the maximum
        correlated_indices.append(corr_mat[i].argsort()[-2:][::-1][1])

    # cannot use any automatic fill nan funciton because we want to assign different val
    # to each different NaN at difference positions based on its correlation
    # matrix

    for i in range(len(label)):
        nan_idx = np.isnan(label[i])
        for j in range(len(nan_idx)):
            if nan_idx[j]:
                label[i][j] = label[i][correlated_indices[j]]
    # if still NaN, replace with most frequent
    label = np.nan_to_num(label)


    return data, label


def judge_model(x_train, x_test, y_train, y_test, model):
    print('-' * 20)
    print('Baseline Performance')
    print('-> Acc:', accuracy_score(y_train, model.predict(x_train)))
    print('-> Acc:', accuracy_score(y_test, model.predict(x_test)))

    #print('-> AUC:', roc_auc_score(y_train, model.predict_proba(x_train)[1]))


def train(data, label):
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, test_size=0.25)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x_train, y_train)
    judge_model(x_train, x_test, y_train, y_test, dummy)


def main():
    data, label = impute_label()
    train(data, label)


if __name__ == '__main__':
    main()
