import matplotlib.pyplot as plt
import numpy as np

from utils import *

np.random.seed(311)


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        prob_correct = sigmoid(theta[u] - beta[q])
        log_lklihood += data["is_correct"][i] * np.log(prob_correct) \
                        + (1 - data["is_correct"][i]) * np.log(1 - prob_correct)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    updated_theta = theta.copy()
    updated_beta = beta.copy()
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        prob_correct = sigmoid(x)
        updated_theta[u] -= lr * (prob_correct - data["is_correct"][i])
        updated_beta[q] += lr * (prob_correct - data["is_correct"][i])
    theta, beta = updated_theta, updated_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.

    theta = np.ones(542)
    beta = np.ones(1774)

    nll_train_acc_lst, nll_val_acc_lst = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        nll_train_acc_lst.append(neg_lld)
        nll_val_acc_lst.append(neg_log_likelihood(val_data, theta=theta, beta=beta))
        val_acc = evaluate(val_data, theta, beta)
        train_acc = evaluate(data, theta, beta)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if score > best_val_acc:
            best_val_acc = score
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies vs. Iterations (Baseline IRT)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('mod_irt_base.png')
    plt.show()

    # TODO: You may change the return values to achieve what you want.
    # print("Best validation accuracy: {}".format(best_val_acc))
    return theta, beta, nll_train_acc_lst, nll_val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 100
    theta, beta, train_acc_lst, val_acc_lst = irt(train_data, val_data, lr, iterations)
    validation_score = evaluate(val_data, theta, beta)
    test_score = evaluate(test_data, theta, beta)
    print("Validation Accuracy: {}".format(validation_score))
    print("Test Accuracy: {}".format(test_score))

    # Plot 1: Combined Accuracy vs Iterations
    plt.plot(range(1, iterations + 1), train_acc_lst, label='Training Accuracy')
    plt.plot(range(1, iterations + 1), val_acc_lst, label='Validation Accuracy')
    plt.title('Neg-Loglikelihood Accuracy vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log Likelihood Accuracy')
    plt.legend()

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plots
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1, j2, j3 = np.random.choice(1774, 3, replace=False)
    theta_vals = np.linspace(-4.0, 4.0, 100)

    prob_j1 = sigmoid(theta_vals - beta[j1])
    prob_j2 = sigmoid(theta_vals - beta[j2])
    prob_j3 = sigmoid(theta_vals - beta[j3])

    plt.plot(theta_vals, prob_j1, label='Question 1')
    plt.plot(theta_vals, prob_j2, label='Question 2')
    plt.plot(theta_vals, prob_j3, label='Question 3')
    plt.title('Probability of Correctness vs Theta')
    plt.xlabel('Theta')
    plt.ylabel('Probability of Correctness')
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # lr = [0.001, 0.01, 0.1, 1]
    # iterations = [100, 200, 300, 400, 500]
    # val_acc = []
    # test_acc = []
    # best_lr = None
    # best_iter = None
    # best_val_acc = 0.0
    # best_theta = None
    # best_beta = None
    # for alpha in lr:
    #     for i in iterations:
    #         print("Learning rate: {} \t Iterations: {}".format(alpha, i))
    #         theta, beta, val_acc_lst = irt(train_data, val_data, alpha, i)
    #         score = evaluate(test_data, theta, beta)
    #         print("Test accuracy: {}".format(score))
    #         val_acc.append(val_acc_lst[-1])
    #         test_acc.append(score)
    #         if score > best_val_acc:
    #             best_lr = alpha
    #             best_iter = i
    #             best_val_acc = score
    #             best_theta = theta
    #             best_beta = beta
    #


if __name__ == "__main__":
    main()
