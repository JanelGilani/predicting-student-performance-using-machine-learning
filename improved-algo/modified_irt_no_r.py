from part_a.item_response import *
from part_a.ensemble import *
import matplotlib.pyplot as plt
import itertools

np.random.seed(311)


def compute_neg_log_likelihood(data, thetas, betas, r):
    log_likelihood = 0

    for i, q_id in enumerate(data["question_id"]):
        u_id = data["user_id"][i]
        x = (thetas[u_id] - betas[q_id])
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


def gd_update(data, lr, theta, beta, r, lambda_value):
    num_users = len(theta)
    num_questions = len(beta)

    theta_mat = np.tile(theta, (num_questions, 1)).T
    beta_mat = np.tile(beta, (num_users, 1))

    train_mat = marks_matrix(data)
    padded_train_matrix = np.nan_to_num(train_mat)

    nan_mask = np.isnan(train_mat)

    x_all = theta_mat - beta_mat
    prob_all = sigmoid(x_all) * (1 - r) + r
    prob_all[nan_mask] = 0

    update_theta = lr * (np.sum((prob_all - padded_train_matrix), axis=1) * (1 - r) + lambda_value * theta)
    theta -= update_theta

    update_beta = lr * (np.sum((padded_train_matrix - prob_all), axis=0) * (1 - r) + lambda_value * beta)
    beta -= update_beta

    update_r = lr * (1 - sigmoid(x_all).mean())
    r -= update_r
    r = np.clip(r, 0, 1)

    return theta, beta, r


def modified_irt(data, val_data, learning_rate, num_iterations, regularization):
    num_users = 542
    num_questions = 1774

    learned_theta = np.ones(num_users)
    learned_beta = np.ones(num_questions)
    r_value = 0

    train_neg_llks, val_neg_llks = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_iterations):
        train_neg_llks.append(compute_neg_log_likelihood(data, learned_theta, learned_beta, r_value))
        val_neg_llks.append(compute_neg_log_likelihood(val_data, learned_theta, learned_beta, r_value))

        learned_theta, learned_beta, r_value = gd_update(
            data, learning_rate, learned_theta, learned_beta, r_value, regularization
        )

        val_accuracy = evaluate(val_data, learned_theta, learned_beta, r_value)
        train_accuracy = evaluate(data, learned_theta, learned_beta, r_value)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch: {epoch + 1}/{num_iterations}\tTrain Accuracy: {train_accuracy}\tValidation Accuracy: {val_accuracy}")

    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies vs. Iterations (Modified IRT with no r)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('mod_irt_no_r.png')
    plt.show()

    return learned_theta, learned_beta, r_value


def evaluate(data, theta, beta, r):
    pred = []

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q])
        p_a = sigmoid(x) * (1 - r) + r
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    lambd = 0.1
    iterations = 100
    theta, beta, r = modified_irt(train_data, val_data, lr, iterations, lambd)

    final_val_acc = evaluate(val_data, theta, beta, r)
    final_test_acc = evaluate(test_data, theta, beta, r)
    print(
        f"Modified IRT (without r) Validation Accuracy is {final_val_acc}\n"
        f"Modified IRT (without r) Test Accuracy is {final_test_acc}")

    final_train_acc = evaluate(train_data, theta, beta, r)
    print(f"Modified IRT (without r) Training Accuracy {final_train_acc}")


if __name__ == "__main__":
    main()
