from part_a.item_response import *
from part_a.ensemble import *
import itertools

np.random.seed(311)


def neg_log_likelihood(data, theta, beta, r):
    log_likelihood = 0

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        prob = sigmoid(x) * (1 - r) + r

        correct = data["is_correct"][i]
        log_likelihood += correct * np.log(prob) + (1 - correct) * np.log(1 - prob)

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
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, 1774))
    beta_mat = np.ones((542, 1)) @ np.expand_dims(beta, axis=0)
    k_mat = np.ones((542, 1))

    train_matrix = marks_matrix(data)
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    nan_mask = np.isnan(train_matrix)

    x = (theta_mat - beta_mat)
    prob = sigmoid(x) * (1 - r) + r
    prob[nan_mask] = 0

    theta -= lr * (np.sum((prob - zero_train_matrix) * k_mat, axis=1) * (1 - r) + lambda_value * theta)
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, 1774))

    x = (theta_mat - beta_mat)
    prob = sigmoid(x) * (1 - r) + r
    prob[nan_mask] = 0

    beta -= lr * (np.sum((zero_train_matrix - prob) * k_mat, axis=0) * (1 - r) + lambda_value * beta)
    beta_mat = np.ones((542, 1)) @ np.expand_dims(beta, axis=0)

    x = (theta_mat - beta_mat)
    prob = sigmoid(x) * (1 - r) + r
    prob[nan_mask] = 0

    x = (theta_mat - beta_mat)
    prob = sigmoid(x) * (1 - r) + r
    prob[nan_mask] = 0

    r -= lr * (1 - sigmoid(x).mean())
    r = max(0, r)
    r = min(1, r)

    return theta, beta, r


def modified_irt(data, val_data, lr, iterations, lambda_value):
    theta = np.ones(542)
    beta = np.ones(1774)
    r = 0

    train_neg_llds, val_neg_llds = [], []
    train_accuracies, val_accuracies = [], []

    for i in range(iterations):
        train_neg_llds.append(neg_log_likelihood(data, theta, beta, r))
        val_neg_llds.append(neg_log_likelihood(val_data, theta, beta, r))

        theta, beta, r = gd_update(data, lr, theta, beta, r, lambda_value)

        val_acc = evaluate(val_data, theta, beta, r)
        train_acc = evaluate(data, theta, beta, r)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print("Epoch: {}/{} \t Train Acc: {}\t Valid Acc: {}".format(i, iterations, train_acc, val_acc))

    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies vs. Iterations (Modified IRT with no d)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('mod_irt_no_d.png')
    plt.show()

    return theta, beta, r


def evaluate(data, theta, beta, r):
    pred = []

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        p_a = sigmoid(x) * (1 - r) + r
        pred.append(p_a >= 0.5)

    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    lambda_value = 0.1
    iterations = 100
    theta, beta, r = modified_irt(train_data, val_data, lr, iterations, lambda_value)

    final_val_acc = evaluate(val_data, theta, beta, r)
    final_test_acc = evaluate(test_data, theta, beta, r)
    print(
        f"Modified IRT (without d) Validation Accuracy is {final_val_acc}\nModified IRT (without d) Test Accuracy is {final_test_acc}")

    final_train_acc = evaluate(train_data, theta, beta, r)
    print(f"Modified IRT (without d) Training Accuracy is {final_train_acc}")


if __name__ == "__main__":
    main()
