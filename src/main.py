import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from train import gradient_descent
from evaluate import validate, test


# getting the csv file from data folder
path_1 = r'data\mnist_train.csv'
data_1 = pd.read_csv(path_1)
# print(data_1.head())
# print(data_1.shape)

#converting the csv data into an array and shuffling it
# here m is total number of images that is 60000  
data_1 = np.array(data_1)
m, n = data_1.shape 
np.random.shuffle(data_1)
# print(m,n) 

path_2 = r'data\mnist_test.csv'
data_2 = np.array(pd.read_csv(path_2))



#dividing the data into training data and testing/validating data 
train_data = data_1[0:int(0.8*m), :]#take the starting 80 percent of the data for training 
val_data = data_1[int(0.8*m):m, :] #and rest 20 percent for testing 



X_train = train_data[:, 1:].T #in x_train we have removed the first column that contains label column so that the model dont know the number we are give to it
#also we transposed the matrix so that each column has data of image
X_train = X_train / 255.0 # here we are scaling the data that means we are converting the value of brightness from the range 0-255 to 0-1
Y_train = train_data[:, 0] #here it only contains the real value that what number the image has 

# print(X_train.shape)
# print(Y_train.shape)
# print(X_train)
# print(Y_train)

#similar to what we did above
X_val = val_data[:, 1:].T
X_val = X_val / 255.0
Y_val = val_data[:, 0]
# print(X_val.shape)
# print(Y_val.shape)
# print(X_val)
# print(Y_val)


X_test = data_2[:, 1:].T
X_test = X_test / 255.0
Y_test = data_2[:, 0]

W1, B1, W2, B2, iters, accs, losses = gradient_descent(
    X_train, Y_train, 0.1, 1000
)

# validation
validate(W1, B1, W2, B2, X_val, Y_val)

plt.figure()
plt.plot(iters, accs)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Iterations")
plt.show()


plt.figure()
plt.plot(iters, losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss vs Iterations")
plt.show()

#  testing
test(W1, B1, W2, B2, X_test, Y_test)