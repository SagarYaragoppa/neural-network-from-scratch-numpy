from model import *

def gradient_descent(X, Y, alpha, iterations):
    W1, B1, W2, B2 = initialize_parameters()

    accuracies = []
    losses = []
    iteration_list = []

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)

        dW1, dB1, dW2, dB2 = backward_propagation(
            W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y
        )

        W1, B1, W2, B2 = update_parameters(
            W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha
        )

        if i % 100 == 0:
            acc = get_accuracy(get_predictions(A2), Y)
            loss = compute_loss(A2, Y)

            accuracies.append(acc)
            losses.append(loss)
            iteration_list.append(i)

            print(f"Iteration {i} | Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    return W1, B1, W2, B2, iteration_list, accuracies, losses
