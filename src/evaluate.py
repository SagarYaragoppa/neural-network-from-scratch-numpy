from model import *

def validate(W1, B1, W2, B2, X_val, Y_val):
    _, _, _, A2_val = forward_propagation(W1, B1, W2, B2, X_val)
    val_acc = get_accuracy(get_predictions(A2_val), Y_val)
    print("Validation Accuracy =", val_acc)
    return val_acc


def test(W1, B1, W2, B2, X_test, Y_test):
    _, _, _, A2_test = forward_propagation(W1, B1, W2, B2, X_test)
    test_acc = get_accuracy(get_predictions(A2_test), Y_test)
    print("Final Test Accuracy =", test_acc)
    return test_acc
