import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
import pickle
import sys
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from gensim.models import Word2Vec
from torch.autograd import Variable
import matplotlib.pyplot as plt

sys.path.append("../../")
import proj_invivo.utils.config as cfg
from proj_invivo.nlp_approach.word2vec import tokenize


class LookUpTable():
    """Looks up the word embedding given a character in SMILES"""

    def __init__(self):
        model = Word2Vec.load('w2vmodel')
        self.character_vector_lookup = model

    def create_emb_layer(self, char):
        try:
            weights_matrix = self.character_vector_lookup[char]
        except KeyError:
            weights_matrix = np.random.normal(
                scale=0.6, size=(cfg.EMBED_DIM, ))
        return weights_matrix


class SmilesDataset(Dataset):
    """
    Pytorch dataset class for our word vectors
    """

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.__load_data()

    def __load_data(self):
        data, label = tokenize()
        self.data = data
        self.label = label
        self.len_total = len(label)

    def __len__(self):
        return self.len_total

    def __getitem__(self, idx):
        """
        returns the item in the datset
        can use nn.memmap to optimize loading if the dataset gets bigger
        Here we use the pickl in __load_data
        """
        label = self.label[idx]
        one_hot_label = []
        # convert in one hot
        for i in label:
            if i == 0:
                vec = torch.tensor([1.0, 0.0], dtype=torch.float)
            else:
                vec = torch.tensor([0.0, 1.0], dtype=torch.float)
            one_hot_label.append(vec)
        return self.data[idx], one_hot_label


class GRUarch(nn.Module):
    """
    Defines the gru architecture, not including the optimization process
    The architecture and plotting code reused from my group project
    see the project here if interested
    https://github.com/violetguos/ift6135-assignment3/blob/47ee6ab806548fa137d51519061050f8af8a5298/problem3/gan.py#L109
    """

    def __init__(self):
        super(GRUarch, self).__init__()
        self.rnn1 = nn.GRU(input_size=10, hidden_size=128, num_layers=1)
        # use sigmoid for recurrent models
        self.activation = nn.Sigmoid()
        self.bin_clfs = nn.ModuleList(
            # 11 targets, each has 2 possibilities
            [nn.Linear(128, 2) for i in range(11)]
        )

    def forward(self, x, hidden):
        x, hidden = self.rnn1(x, hidden)
        x = self.activation(x)
        x = [clf(x) for clf in self.bin_clfs]
        x = [self.activation(xi) for xi in x]
        return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, 128).zero_())


def torch_loader(batch_size):
    """
    Returns type data loader from the NLTK word vectors
    """
    data_word_vectors = SmilesDataset()
    len_total = data_word_vectors.len_total
    valid_split = 0.2
    indices = list(range(len_total))
    split = int(np.floor(valid_split * len_total))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Now use random indices to assign samples for each dataloader
    train_loader = DataLoader(dataset=data_word_vectors,
                              batch_size=batch_size,
                              sampler=train_sampler)

    valid_loader = DataLoader(dataset=data_word_vectors,
                              batch_size=batch_size,
                              sampler=valid_sampler)
    return train_loader, valid_loader


class GRUmodel():

    def __init__(self, config=None, device='cpu', batch_size=None, model_path=None):
        # Set up model
        self.device = device
        self.model = GRUarch()
        if model_path:
            self.load_model(model_path)
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.name = 'gru'
        self.embedding = LookUpTable()
        # init the training curve log to track losses
        self.train_losses, self.valid_losses = [], []
        self.train_acc = []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def log_learning_curves(self):
        '''
        Logs the discriminator cross-entropy loss to a csv.
        '''
        header = 'epoch,train_loss\n'
        num_epochs = len(self.train_losses)
        with open(os.path.join('log', 'crossentropy.log'), 'w') as fp:
            fp.write(header)
            for e in range(num_epochs):
                fp.write('{},{}\n'.format(
                    e, self.train_losses[e]))

    def train_epoch(self, train_loader, loss_fn=None, optimizer=None):
        """
        Defines the training inside each individual epoch
        We now only work with batch size of 1 
        see explanation below
        """
        self.model.train()
        loss_fn.zero_grad()
        hidden = self.model.init_hidden(1)

        # I didn't append <EOS> to molecule SMILE strings
        # first loop iterates the whole dataset
        # second loop iterates the characters inside one string data we have
        total = 0
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            y = torch.stack(y)
            for char in x:
                char_embedding = self.embedding.create_emb_layer(char)
                char_embedding = torch.tensor(
                    char_embedding, dtype=torch.float)
                char_embedding = char_embedding.to(self.device)
                char_embedding = char_embedding.unsqueeze(0)

                y_pred, _ = self.model(char_embedding, hidden)
                y_pred = torch.stack(y_pred)

                train_loss = loss_fn(y_pred, y)
                train_loss.backward()
                optimizer.step()

                _, predicted = torch.max(y_pred.data, 1)
                _, labels = torch.max(y, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            self.train_losses.append(train_loss)
            acc = 100 * correct / total
            self.train_acc.append(acc)

        return train_loss

    def train(self, train_loader, valid_loader, num_epochs=10):
        '''
        Wrapper function for training on training set + evaluation on validation set.
        # NOTE: validation not implemented due to time constraints
        '''
        adam_optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        adam_optim.zero_grad()
        bce_loss_fn = nn.BCELoss()
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer=adam_optim, loss_fn=bce_loss_fn)
            print("*" * 20)

            print('Epoch {}:'.format(epoch))
            print(' \t train_loss: {}'.format(train_loss))


def main():
    batch_size = 1

    train_loader, valid_loader = torch_loader(batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru = GRUmodel(device=device, batch_size=batch_size)
    gru.train(train_loader, valid_loader)
    # gru.log_learning_curves()
    gru.save_model('gru.pt')


def plot():

    with open('crossentropy.log') as fp:
        lines = fp.readlines()
    train_losses = []
    for line in lines[1:]:    # Skip header
        line_split = line.split(',')
        line_split = [float(x) for x in line_split]
        train_losses.append(line_split[1])
    epochs = list(range(len(train_losses)))
    epochs = [float(epoch) for epoch in epochs]

    train_losses = np.array(train_losses)
  
    epochs = range(1, 10)
    # Now plot
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (7, 5)

    plt.plot(epochs, train_losses, color='blue', linestyle='solid', label='Training losses')
    plt.xlabel('Number of epochs')
    plt.xticks(epochs)

    plt.ylabel('Cross-entropy loss')

    legend = plt.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.savefig('training_losses')
    plt.show()


if __name__ == '__main__':
    plot()
