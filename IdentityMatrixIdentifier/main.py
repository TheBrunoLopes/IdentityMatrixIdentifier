# Imports for the objects used in the next steps
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(653424)

# Initializing an identity matrix
identity_matrix = np.identity(8)
# ------------------------------------------
# Step a.
#    --> Creating and initializing a 8x3x8 neural network
# ------------------------------------------
print("Step a. Creating and initializing a 8x3x8 neural network...")

model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=8))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=8, activation='softmax'))

# The Adam optimizer introduced in this paper: https://arxiv.org/abs/1412.6980
# and the nadam ( Nesterov Adam Optimizer ) talked about in http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
# provided the best results compared to other optimizers that I tried and are provided by Keras
model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

# ------------------------------------------
# Step b.
#    --> Training the network to learn the 8x8 identity matrix
# ------------------------------------------
# We train using batches of 32 samples.
# Every batch is generated at run time instead of creating a huge dataset before the training starts.
# Training data and label data are the same due to the nature of our problem.
print("Step b. Training the network to learn the 8x8 identity matrix...")
history = np.empty((3000, 2))

for i in range(3000):
    data = identity_matrix[np.random.choice(identity_matrix.shape[0], 32), :]
    history[i] = model.train_on_batch(data, data)

# Making a plot to show the evolution of the model training
# Looking at plots like this and tweaking the parameters is very important to create good models
plt.plot(history[:, 0], label="Loss Function (Cross Entropy)")
plt.plot(history[:, 1], label="Accuracy")
plt.legend()
plt.show()

data = identity_matrix[np.random.choice(identity_matrix.shape[0], 100), :]
# print(model.metrics_names)
print("Trained a model with the accuracy of {:.2%}".format(model.evaluate(data, data, batch_size=100)[1]))

# ------------------------------------------
# Step c.
#    --> Prediction of the neural network for every row of
# #             the 8x8 identity matrix
# ------------------------------------------
print("Step c. Prediction of the neural network for every row of "
      "the 8x8 identity matrix")
# In the next few lines we can observe the lambda function 'row_identifier' being applied to every row of
# the identity matrix
# The first row is identified with the number 1
row_identifier = lambda row: print(np.argmax(model.predict(np.array([row]), batch_size=1)) + 1)
np.apply_along_axis(row_identifier, axis=1, arr=identity_matrix)
# The expected outcome is 1 2 3 4 5 6 7 8 (prediction for every row of the identity matrix)
