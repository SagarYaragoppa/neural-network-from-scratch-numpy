from model import *

def validate(W1, B1, W2, B2, W3, B3, X_val, Y_val):
    _, _, _, _, _, A3_val = forward_propagation(
        W1, B1, W2, B2, W3, B3, X_val
    )

    val_acc = get_accuracy(get_predictions(A3_val), Y_val)
    print("Validation Accuracy =", val_acc)
    return val_acc


def test(W1, B1, W2, B2, W3, B3, X_test, Y_test):
    _, _, _, _, _, A3_test = forward_propagation(
        W1, B1, W2, B2, W3, B3, X_test
    )

    test_acc = get_accuracy(get_predictions(A3_test), Y_test)
    print("Final Test Accuracy =", test_acc)
    return test_acc
