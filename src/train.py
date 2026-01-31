from model import *

def gradient_descent(X, Y, alpha, iterations):
    W1, B1, W2, B2, W3, B3 = initialize_parameters()


    accuracies = []
    losses = []
    iteration_list = []

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(
        W1, B1, W2, B2, W3, B3, X
        )


        dW1, dB1, dW2, dB2, dW3, dB3 = backward_propagation(
        W1, B1, W2, B2, W3, B3,
        Z1, A1, Z2, A2, Z3, A3, X, Y
        )


        W1, B1, W2, B2, W3, B3 = update_parameters(
        W1, B1, W2, B2, W3, B3,
        dW1, dB1, dW2, dB2, dW3, dB3, alpha
        )   


        if i % 100 == 0:
            acc = get_accuracy(get_predictions(A3), Y)
            loss = compute_loss(A3, Y)

            accuracies.append(acc)
            losses.append(loss)
            iteration_list.append(i)

            print(f"Iteration {i} | Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    return W1, B1, W2, B2, W3, B3, iteration_list, accuracies, losses

