from part_a.item_response import *

np.random.seed(311)


def bootstrap_sampling(data, num_resamples):
    resampled_data = []
    user_ids = np.array(data["user_id"])
    question_ids = np.array(data["question_id"])
    is_correct_vals = np.array(data["is_correct"])

    for _ in range(num_resamples):
        random_indices = np.random.randint(0, len(data["user_id"]), len(data["user_id"]))
        sampled_data = create_sampled_data(user_ids, question_ids, is_correct_vals, random_indices)
        resampled_data.append(sampled_data)
    return resampled_data


def create_sampled_data(user_ids, question_ids, is_correct_vals, random_indices):
    sampled_data = {
        "user_id": user_ids[random_indices],
        "question_id": question_ids[random_indices],
        "is_correct": is_correct_vals[random_indices]
    }
    return sampled_data


def irt_ensemble_predict(data, theta_list, beta_list, majority=False, threshold=0.5):
    ensemble_predictions = np.zeros(len(data["is_correct"]))
    for idx, theta in enumerate(theta_list):
        temp_pred = irt_single_predict(data, theta, beta_list[idx])
        if majority:
            temp_pred = (temp_pred >= threshold).astype(int)
        ensemble_predictions += temp_pred
    return ((ensemble_predictions / len(theta_list)) >= threshold).astype(int)


def irt_single_predict(data, theta, beta, binary=False, threshold=0.5):
    irt_predictions = []
    for i, question_id in enumerate(data["question_id"]):
        user_id = data["user_id"][i]
        x = (theta[user_id] - beta[question_id]).sum()
        probability_correct = sigmoid(x)
        irt_predictions.append(probability_correct)
    irt_predictions = np.array(irt_predictions)
    if binary:
        return (irt_predictions >= threshold).astype(int)
    return irt_predictions


def ensemble_evaluate(data, predictions):
    return np.sum((np.array(data["is_correct"]) == np.array(predictions))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    num_resamples = 3
    resampled_train_data = bootstrap_sampling(train_data, num_resamples)
    lr = 0.01
    iterations = 30

    theta_list, beta_list = [], []
    for i in range(num_resamples):
        print(f"Training Model {i + 1}")
        theta, beta, _, _ = irt(resampled_train_data[i], val_data, lr, iterations)
        theta_list.append(theta)
        beta_list.append(beta)

    ensemble_train = irt_ensemble_predict(train_data, theta_list, beta_list)
    ensemble_val = irt_ensemble_predict(val_data, theta_list, beta_list)
    ensemble_test = irt_ensemble_predict(test_data, theta_list, beta_list)

    ensemble_train_accuracy = ensemble_evaluate(train_data, ensemble_train)
    ensemble_val_accuracy = ensemble_evaluate(val_data, ensemble_val)
    ensemble_test_accuracy = ensemble_evaluate(test_data, ensemble_test)

    # Original accuracies from item_response.py
    print("Without Ensemble (Non-Ensemble):")
    theta, beta, train_acc_lst, val_acc_lst = irt(train_data, val_data, lr, iterations)
    validation_score = evaluate(val_data, theta, beta)
    test_score = evaluate(test_data, theta, beta)

    print("Non-Ensemble Validation Accuracy: {}".format(validation_score))
    print("Non-Ensemble Test Accuracy: {}".format(test_score))
    print("Ensemble Train Accuracy: ", ensemble_train_accuracy)
    print("Ensemble Validation Accuracy: ", ensemble_val_accuracy)
    print("Ensemble Test Accuracy: ", ensemble_test_accuracy)


if __name__ == "__main__":
    main()
