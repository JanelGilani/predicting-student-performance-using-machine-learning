from part_a.item_response import *
from part_a.ensemble import *
import matplotlib.pyplot as plt
import itertools

np.random.seed(311)


def compute_neg_log_likelihood(data, thetas, betas, r, ds):
    log_likelihood = 0

    for i, q_id in enumerate(data["question_id"]):
        u_id = data["user_id"][i]
        x = ds[q_id] * (thetas[u_id] - betas[q_id])
        probability = sigmoid(x) * (1 - r) + r

        correct = data["is_correct"][i]
        log_likelihood += correct * np.log(probability) + (1 - correct) * np.log(1 - probability)

    return -log_likelihood


def marks_matrix(data):
    matrix = np.empty(shape=(542, 1774))
    matrix[:] = np.NaN
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        mark = data["is_correct"][i]
        matrix[u, q] = mark
    return matrix


def gd_update(data, lr, theta, beta, r, d, lambda_value):
    num_users = len(theta)
    num_questions = len(beta)

    theta_mat = np.tile(theta, (num_questions, 1)).T
    beta_mat = np.tile(beta, (num_users, 1))
    k_mat = np.tile(d, (num_users, 1))

    train_mat = marks_matrix(data)
    padded_train_matrix = np.nan_to_num(train_mat)

    nan_mask = np.isnan(train_mat)

    x_all = (theta_mat - beta_mat) * k_mat
    prob_all = sigmoid(x_all) * (1 - r) + r
    prob_all[nan_mask] = 0

    update_theta = lr * (np.sum((prob_all - padded_train_matrix) * k_mat, axis=1) * (1 - r) + lambda_value * theta)
    theta -= update_theta

    update_beta = lr * (np.sum((padded_train_matrix - prob_all) * k_mat, axis=0) * (1 - r) + lambda_value * beta)
    beta -= update_beta

    update_d = lr * (
                np.sum((prob_all - padded_train_matrix) * (theta_mat - beta_mat), axis=0) * (1 - r) + lambda_value * d)
    d -= update_d

    update_r = lr * (1 - sigmoid(x_all).mean())
    r -= update_r
    r = np.clip(r, 0, 1)

    return theta, beta, r, d


def modified_irt(data, val_data, learning_rate, num_iterations, regularization):
    num_users = 542
    num_questions = 1774

    learned_theta = np.ones(num_users)
    learned_beta = np.ones(num_questions)
    r_value = 0
    learned_d = np.ones(num_questions)

    train_neg_llks, val_neg_llks = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_iterations):
        train_neg_llks.append(compute_neg_log_likelihood(data, learned_theta, learned_beta, r_value, learned_d))
        val_neg_llks.append(compute_neg_log_likelihood(val_data, learned_theta, learned_beta, r_value, learned_d))

        learned_theta, learned_beta, r_value, learned_d = gd_update(
            data, learning_rate, learned_theta, learned_beta, r_value, learned_d, regularization
        )

        val_accuracy = evaluate(val_data, learned_theta, learned_beta, r_value, learned_d)
        train_accuracy = evaluate(data, learned_theta, learned_beta, r_value, learned_d)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch: {epoch + 1}/{num_iterations}\tTrain Accuracy: {train_accuracy}\tValidation Accuracy: {val_accuracy}")

    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies vs. Iterations (Modified IRT)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('mod_irt.png')
    plt.show()

    return learned_theta, learned_beta, r_value, learned_d


def evaluate(data, theta, beta, r, d):
    """ Evaluate the model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: N x S matrix
    :param r: float
    :param d: Vector
    :param q_meta: D x S matrix
    :return: float
    """
    pred = []

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = ((theta[u] - beta[q]) * d[q]).sum()
        p_a = sigmoid(x) * (1 - r) + r
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


# def hyperparameter_tuning(train_data, val_data, test_data):
#     learning_rates = [0.001, 0.01, 0.1]
#     iterations = [50]
#     lambdas = [0.1, 0.01, 0.001]
#
#     best_val_accuracy = 0.0
#     best_hyperparameters = None
#
#     for lr, num_iter, reg_lambda in itertools.product(learning_rates, iterations, lambdas):
#         theta, beta, r, d = modified_irt(train_data, val_data, lr, num_iter, reg_lambda)
#         val_accuracy = evaluate(val_data, theta, beta, r, d)
#
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             best_hyperparameters = (lr, num_iter, reg_lambda)
#
#     # Test the model with the best hyperparameters on the test set
#     best_lr, best_iter, best_lambda = best_hyperparameters
#     theta, beta, r, d = modified_irt(train_data, test_data, best_lr, best_iter, best_lambda)
#     test_accuracy = evaluate(test_data, theta, beta, r, d)
#
#     return best_hyperparameters, test_accuracy
#
#

#
# def plot_accuracy_vs_r(train_data, val_data, test_data, lr, iterations, lambd, theta, beta, r_values, d):
#     val_accuracies = []
#     test_accuracies = []
#
#     for r in r_values:
#         # Keeping other parameters constant and modifying only r
#         theta_temp, beta_temp, r_temp, d_temp = modified_irt(train_data, val_data, lr, iterations, lambd)
#         final_val_acc = evaluate(val_data, theta_temp, beta_temp, r, d_temp)
#         final_test_acc = evaluate(test_data, theta_temp, beta_temp, r, d_temp)
#         val_accuracies.append(final_val_acc)
#         test_accuracies.append(final_test_acc)
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(r_values, val_accuracies, label='Validation Accuracy')
#     plt.plot(r_values, test_accuracies, label='Test Accuracy')
#     plt.title('Effect of changing r on Accuracies')
#     plt.xlabel('r values')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()


# def plot_accuracy_vs_d(train_data, val_data, test_data, lr, iterations, lambd, theta, beta, r, d_values):
#     val_accuracies = []
#     test_accuracies = []
#
#     for d_val in d_values:
#         # Create a copy of the original d vector to modify individual elements
#         d_temp = d_val.copy()
#
#         # Modify individual elements of d based on d_values
#         for i, val in enumerate(d_val):
#             d_temp[i] = val
#
#         # Evaluate accuracy with modified d vector
#         final_val_acc = evaluate(val_data, theta, beta, r, d_temp)
#         final_test_acc = evaluate(test_data, theta, beta, r, d_temp)
#         val_accuracies.append(final_val_acc)
#         test_accuracies.append(final_test_acc)
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(d_values, val_accuracies, label='Validation Accuracy')
#     plt.plot(d_values, test_accuracies, label='Test Accuracy')
#     plt.title('Effect of changing elements of d on Accuracies')
#     plt.xlabel('d values')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()
#

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    lambd = 0.1
    iterations = 100
    theta, beta, r, d = modified_irt(train_data, val_data, lr, iterations, lambd)

    final_val_acc = evaluate(val_data, theta, beta, r, d)
    final_test_acc = evaluate(test_data, theta, beta, r, d)
    print(
        f"Modified IRT Validation Accuracy is {final_val_acc}\n"
        f"Modified IRT Test Accuracy is {final_test_acc}")

    final_train_acc = evaluate(train_data, theta, beta, r, d)
    print(f"Modified IRT Training Accuracy {final_train_acc}")



    # j1, j2, j3 = np.random.choice(1774, 3, replace=False)
    # theta_vals = np.linspace(-4.0, 4.0, 100)
    #
    # prob_j1 = sigmoid((theta_vals - beta[j1]) * d[j1]) * (1 - r) + r
    # prob_j2 = sigmoid((theta_vals - beta[j2]) * d[j2]) * (1 - r) + r
    # prob_j3 = sigmoid((theta_vals - beta[j3]) * d[j3]) * (1 - r) + r
    #
    # # plt.plot(theta_vals, prob_j1, label='Question 1')
    # # plt.plot(theta_vals, prob_j2, label='Question 2')
    # plt.plot(theta_vals, prob_j3, label='Question 1')
    # plt.title('Probability of Correctness vs Theta (Modified IRT)')
    # plt.xlabel('Theta')
    # plt.ylabel('Probability of Correctness')
    # plt.legend()
    # plt.show()
    # plt.savefig('mod_irt_sample_questions.png')


    # train_data = load_train_csv("../data")
    # val_data = load_valid_csv("../data")
    # test_data = load_public_test_csv("../data")
    #
    # lr = 0.01
    # lambd = 0.1
    # iterations = 100
    # theta, beta, r, d = modified_irt(train_data, val_data, lr, iterations, lambd)
    #
    # r_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Modify these values as needed
    #
    # num_elements_d = 1774  # Adjust this according to the dimension of your 'd' vector
    # num_samples = 3  # Number of samples to generate
    #
    # # Generate random values for individual elements of d
    # d_values = []
    # for _ in range(num_samples):
    #     d_sample = np.random.rand(num_elements_d)  # Random values between 0 and 1
    #     d_values.append(d_sample)

    # Plotting accuracy vs r
    # plot_accuracy_vs_r(train_data, val_data, test_data, lr, iterations, lambd, theta, beta, r_values, d)

    # Plotting accuracy vs d
    # plot_accuracy_vs_d(train_data, val_data, test_data, lr, iterations, lambd, theta, beta, r, d_values)

    # # Load data
    # train_data = load_train_csv("../data")
    # val_data = load_valid_csv("../data")
    # test_data = load_public_test_csv("../data")
    #
    # best_hyperparams, test_acc = hyperparameter_tuning(train_data, val_data, test_data)
    # print(
    #     f"Best Hyperparameters: Learning Rate = {best_hyperparams[0]}, Num Iterations = {best_hyperparams[1]}, Lambda = {best_hyperparams[2]}")
    # print(f"Test Accuracy with Best Hyperparameters: {test_acc}")
    #


if __name__ == "__main__":
    main()
