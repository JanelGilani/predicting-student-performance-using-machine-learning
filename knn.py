import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from utils import *

from project.utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    res = KNNImputer(n_neighbors=k, weights="uniform")
    transformed_matrix = res.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, transformed_matrix.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k = [1, 6, 11, 16, 21, 26]
    user_val_acc = []
    item_val_acc = []

    # User-based k-NN
    for i in k:
        user_val_acc.append(knn_impute_by_user(sparse_matrix, val_data, i))

    best_user_k = k[user_val_acc.index(max(user_val_acc))]
    print("Best k for User-based: {}".format(best_user_k))
    user_val_acc = knn_impute_by_user(sparse_matrix, val_data, best_user_k)
    print("Validation Accuracy (User-based): {}".format(user_val_acc))
    user_test_acc = knn_impute_by_user(sparse_matrix, test_data, best_user_k)
    print("Test Accuracy (User-based): {}".format(user_test_acc))

    # Item-based k-NN
    for i in k:
        item_val_acc.append(knn_impute_by_item(sparse_matrix, val_data, i))

    best_item_k = k[item_val_acc.index(max(item_val_acc))]
    print("Best k for Item-based: {}".format(best_item_k))
    item_val_acc = knn_impute_by_item(sparse_matrix, val_data, best_item_k)
    print("Validation Accuracy (Item-based): {}".format(item_val_acc))
    item_test_acc = knn_impute_by_item(sparse_matrix, test_data, best_item_k)
    print("Test Accuracy (Item-based): {}".format(item_test_acc))

    # Plot the result.
    # plt.plot(k, user_val_acc, marker="o", label="User-based")
    # plt.plot(k, item_val_acc, marker="o", label="Item-based")
    # plt.xlabel("k (Number of Nearest Neighbors)")
    # plt.ylabel("Accuracy")
    # plt.title("Validation Accuracy vs k")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("knn.png")
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
