import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core  import Dense, Activation
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier



def plot_training_loss(losses, algorithm):
    X = np.linspace(0,MaxEpoch, MaxEpoch, endpoint=True).reshape([MaxEpoch,1])
    print(X.shape)
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("Training Loss "+algorithm)
    plt.plot(X, losses, color= "blue")
    plt.tight_layout()
    plt.savefig(f"{algorithm}: traing loss")

def plot_validation_accuracy(accs, algorithm):
    X = np.linspace(0,MaxEpoch, MaxEpoch, endpoint=True).reshape([MaxEpoch,1])
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("Validation Accuracy "+algorithm)
    plt.plot(X, accs, color= "red")
    plt.tight_layout()
    plt.savefig(f"{algorithm}: validatioin Accuracy")


# data import
csv_file = 'Cancer_Data.csv'
data = pd.read_csv('Cancer_Data.csv') 

# data editing
data.drop(["Unnamed: 32","id"],axis=1,inplace = True) 
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# Split the data by x and t
t = data.diagnosis.values    # target
X_data = data.drop(["diagnosis"],axis = 1)
pd.DataFrame(X_data)
pd.DataFrame(t)

# Normalization
X = (X_data - np.min(X_data))/(np.max(X_data)-np.min(X_data)) #Normalization Formula
pd.DataFrame(X)

# Train-Test-Validation 
X = pd.DataFrame(X).to_numpy()
X_train, X_test, t_train, t_test = train_test_split(X,t,test_size=69,random_state=42)


# Algorithm 2 : Neural Network
MaxEpoch = 1000
model = Sequential()
batch_size = 50
model.add(Dense(15,activation= "relu"))
model.add(Dense(10,activation= "relu"))
model.add(Dense(1,activation= "sigmoid"))

# Compile the model
binary_accuracy = tf.keras.metrics.BinaryAccuracy()
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=[binary_accuracy] )

# Train the model
callback1 = ModelCheckpoint(filepath="my_best_model.hdf5", monitor="loss", save_weights_only=False)
callback2 = EarlyStopping(monitor="loss", mode="min",patience=30, verbose=1)
history = model.fit(X_train, t_train, validation_split=0.2, epochs = MaxEpoch , batch_size=50, callbacks=[callback1,callback2])

# learning curves of model accuracy ( train, validation )
# scores = model.evaluate(X_test,t_test, verbose=0)
history_dict = history.history
nn1_train_losses = history_dict['loss']
nn1_train_accs = history_dict['binary_accuracy']
nn1_val_losses = history_dict["val_loss"]
nn1_val_accs = history_dict["val_binary_accuracy"]

# plot_training_loss(nn1_train_losses, "Neural network 1")
# plot_validation_accuracy(nn1_val_accs, "Neural network 1")

# test model 1
_, nn1_test_accuracy = model.evaluate(X_test, t_test, verbose= 0 )


# -------------Neural network model 2------------------ : 
model2 = Sequential()
batch_size = 50
model2.add(Dense(4,activation= "relu"))
model2.add(Dense(1,activation= "sigmoid"))

# Compile the model
binary_accuracy = tf.keras.metrics.BinaryAccuracy()
model2.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=[binary_accuracy] )

# Train the model
callback1 = ModelCheckpoint(filepath="my_best_model.hdf5", monitor="loss", save_weights_only=False)
callback2 = EarlyStopping(monitor="loss", mode="min",patience=30, verbose=1)
history = model2.fit(X_train, t_train, validation_split=0.2, epochs = MaxEpoch , batch_size=50, callbacks=[callback1,callback2])

# learning curves of model accuracy ( train, validation )
# scores = model.evaluate(X_test,t_test, verbose=0)
history_dict = history.history
nn2_train_losses = history_dict['loss']
nn2_train_accs = history_dict['binary_accuracy']
nn2_val_losses = history_dict["val_loss"]
nn2_val_accs = history_dict["val_binary_accuracy"]

# test model 2
_, nn2_test_accuracy = model2.evaluate(X_test, t_test, verbose= 0 )

# baseline accuracy
print("-----------Neural network model 1----------- ")
print("baseline accururacy for nn model1: ",(t_test.shape[0]-sum(t_test))/t_test.shape[0])
nn1_prediction = model.predict(X_test) # float [0,1] 
nn_accuracy = accuracy_score(t_test,nn1_prediction.round())
precision_score_nn = precision_score(t_test, nn1_prediction.round() )
recall_score_nn = recall_score(t_test, nn1_prediction.round())
f_score_nn = f1_score(t_test, nn1_prediction.round())

print("   precision_score: ", precision_score_nn)
print("   recall score: ", recall_score_nn)
print("   f1 score: ", f_score_nn)

# confusion matrix 
print("Confusion matrix for nn model1: ")
confusion_matrix_nn1 = confusion_matrix(t_test, nn1_prediction.round() )
print(confusion_matrix_nn1)
print("Accuracy for Neural Network: ", nn1_test_accuracy)


print("-----------Neural network model 2----------- ")
print("Accuracy for Neural Network: ", nn2_test_accuracy)

# baseline accuracy
nn2_prediction = model2.predict(X_test) # float [0,1] 
nn_accuracy = accuracy_score(t_test,nn2_prediction.round())
print((t_test.shape[0]-sum(t_test))/t_test.shape[0])
precision_score_nn = precision_score(t_test, nn2_prediction.round() )
recall_score_nn = recall_score(t_test, nn2_prediction.round())
f_score_nn = f1_score(t_test, nn2_prediction.round())

print("   precision_score: ", precision_score_nn)
print("   recall score: ", recall_score_nn)
print("   f1 score: ", f_score_nn)

# confusion matrix 
print("Confusion matrix for nn model2: ")
confusion_matrix_nn2 = confusion_matrix(t_test, nn2_prediction.round() )
print(confusion_matrix_nn2)

# plot
training_loss = {"model1":nn1_train_losses, "model2": nn2_train_losses}
validation_accs = {'model1': nn1_val_accs, "model2":nn2_val_accs}

X = np.linspace(0,MaxEpoch,MaxEpoch,endpoint=True).reshape([MaxEpoch,1])
plt.figure()
plt.xlabel('epoch')
plt.ylabel('training loss')
for (model, losses) in training_loss.items():
    plt.plot(X, training_loss[model], label= "model: "+model)
plt.legend(loc = "upper right")
plt.savefig('NN_Training_loss.jpg')

plt.figure()
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
for (model, acc) in validation_accs.items():
    plt.plot(X, validation_accs[model], label= "decay: "+model)
plt.legend(loc = "upper right")
plt.savefig('NN_Validation_Accuracy.jpg')





