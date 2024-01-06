from starter_code.utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)  # W_1
        self.h = nn.Linear(k, num_question)  # W_2

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        sig = nn.Sigmoid()

        hidden = sig(torch.matmul(inputs, torch.transpose(self.g.weight, 0, 1)) + self.g.bias)
        out = sig(torch.matmul(hidden, torch.transpose(self.h.weight, 0, 1)) + self.h.bias)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_loss_per_epoch = []
    valid_acc_per_epoch = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + ((lamb * 0.5) * (model.get_weight_norm()))
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        train_loss_per_epoch.append(train_loss)
        valid_acc_per_epoch.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    return train_loss_per_epoch, valid_acc_per_epoch

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    # part (C), pick k_star and tune hyperparameters
    k_choice = [10, 50, 100, 200, 500]
    lr_choice = [0.01, 0.05, 0.1]
    epoch_choice = [10, 15, 20]

    max_valid_acc = -float("inf")
    best_k = -1
    best_lr = -1
    best_epoch = -1

    for k in k_choice:
        for lr in lr_choice:
            valid_acc = -float("inf")
            model = None
            for num_epoch in epoch_choice:
                model = AutoEncoder(train_matrix.shape[1], k)
                train(model, lr, 0, train_matrix, zero_train_matrix, valid_data, num_epoch)
                valid_acc = evaluate(model, zero_train_matrix, valid_data)
                print("model finished training with hyperparameters k:{}\tlr:{}\tnum_epoch:{}".format(k, lr, num_epoch))

                if valid_acc > max_valid_acc:
                    max_valid_acc = valid_acc
                    best_k = k
                    best_lr = lr
                    best_epoch = num_epoch

    print("best k: {}\tbest lr: {}\t best num_epoch:{}\t resulted in valid acc: {}".format(best_k, best_lr, best_epoch,
                                                                                           max_valid_acc))

    # Set model hyperparameters.
    k = best_k
    model = AutoEncoder(train_matrix.shape[1], k)

    # Set optimization hyperparameters.
    lr = best_lr
    num_epoch = best_epoch

    # part(D), plot train loss and valid acc per epoch
    train_loss_per_epoch, valid_acc_per_epoch = train(model, lr, 0, train_matrix, zero_train_matrix, valid_data,
                                                      num_epoch)
    x = [i for i in range(1, num_epoch + 1)]

    # Plotting Training Loss
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot in the first position
    plt.plot(x, train_loss_per_epoch, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()

    # Plotting Validation Accuracy
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot in the second position
    plt.plot(x, valid_acc_per_epoch, label='Validation Accuracy', color='orange', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.xticks(x)
    # Show the plot
    plt.savefig('autoencoder_per_epoch.png')

    print("Test accuracy for best hyperparameters: {}".format(evaluate(model, zero_train_matrix, test_data)))

    # part (E)
    lambda_choice = [0.001, 0.01, 0.1, 1]
    best_lamb = -1
    max_valid_acc_with_lamb = -float("inf")

    for lamb in lambda_choice:
        model = AutoEncoder(train_matrix.shape[1], k)
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        valid_acc = evaluate(model, zero_train_matrix, valid_data)

        if valid_acc > max_valid_acc_with_lamb:
            best_lamb = lamb
            max_valid_acc_with_lamb = valid_acc

    lamb = best_lamb

    print("The best lambda was was: {}".format(lamb))

    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)

    final_valid_acc = evaluate(model, zero_train_matrix, valid_data)
    final_test_acc = evaluate(model, zero_train_matrix, test_data)

    print("The final validation accuracy was: {}".format(final_valid_acc))
    print("The final test accuracy was: {}".format(final_test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
