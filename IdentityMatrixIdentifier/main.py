# Imports for the objects used in the next steps
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#np.random.seed(65424)
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

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ------------------------------------------
# Step b.
#    --> Training the network to learn the 8x8 identity matrix
# ------------------------------------------

# one_hot_labels = keras.utils.to_categorical(y_train, num_classes=8)
# 10000 iterations
print("Step b. Training the network to learn the 8x8 identity matrix...")
for i in range(10000):
    data = identity_matrix[np.random.choice(identity_matrix.shape[0], 32), :]
    model.train_on_batch(data, data)

data = identity_matrix[np.random.choice(identity_matrix.shape[0], 100), :]
# print(model.metrics_names)
print("Created a model with the accuracy of {:.2%}".format(model.evaluate(data, data, batch_size=100)[1]))


# ------------------------------------------
# Step c.
#    --> Prediction of the neural network for every row of
# #             the 8x8 identity matrix
# ------------------------------------------
print("Step c. Prediction of the neural network for every row of "
      "the 8x8 identity matrix")
# The first row is identified with the number 1


#row_identifier = lambda row: print(np.argmax(model.predict([row], batch_size=1))+1)
#np.apply_along_axis(row_identifier, axis=1, arr=identity_matrix)

#print(np.argmax(model.predict([row], batch_size=1))+1)

print(model.predict(identity_matrix, batch_size=8))
