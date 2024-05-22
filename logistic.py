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

def sig(z):
    return 1/(1+np.exp(-z)) # y = sigmoid(z)

def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    X_trained = X[:400]
    X_val = X[400:]
    t_trained = t[:400]
    t_val = t[400:]

    N_train = X_trained.shape[0]

    
    w = np.zeros([X_trained.shape[1], 1])
    b = 0
    accuracy_best = 0 
    w_best,b_best, epoch_best =None, None, None

    training_loss = []
    validation_accs = []
    batch_number = int(np.ceil(N_train/batch_size))

    for e in range(MaxEpoch):
        loss_this_epoch = 0
        for batch in range(int(np.ceil(N_train/batch_size))):
            X_batch = X_trained[batch*batch_size: (batch+1)*batch_size] 
            t_batch= t_trained[batch*batch_size: (batch+1)*batch_size] 

            z = np.matmul(X_batch,w)+b # (60,d) (d,1) (1,1) -> 60, 1
            t_batch = np.reshape(t_batch,(batch_size,1)) # previous: (60,)
            

            # loss = -log(y) it t = 1 if or -log(1-y) if t = 0
            # [[34]]
            loss_batch = -np.matmul(np.transpose(t_batch),np.log(sig(z)))-np.matmul(np.transpose(1-t_batch),np.log(1-sig(z)))
            loss_this_epoch+= loss_batch[0][0]/ batch_size
            
            # update gradient
            gradient_w = np.matmul(np.transpose(X_batch),(sig(z)-t_batch)) # g_w = transpose(x) * (y-t)
            gradient_b= np.sum((sig(z)-t_batch)) # sum (yi-ti)
            w-= alpha* (gradient_w)
            b-= alpha*(gradient_b)

        training_loss.append(loss_this_epoch/batch_number)
        
        # Validation Accuracy
        predict_val = predict_logistic_regression(X_val,w,b)
        accuracy = get_accuracy(predict_val,t_val)
        validation_accs.append(accuracy)
        
        if accuracy> accuracy_best:
            w_best = w
            b_best = b
            epoch_best = e

    return e,w_best, b_best, training_loss, validation_accs

def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    z = np.matmul(X,w)+b # (300,2), (2,1) (1,1)
    y = sig(z)
    t_hat = [0 for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        if y[i]> 0.5:
            t_hat[i] = 1
        else:
            t_hat[i] = 0

    return t_hat

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    match = 0
    for i in range(len(t)):
        if t[i]==t_hat[i]:
            match+=1
    acc = 1/len(t)* match

    return acc


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

# Algorithm 1 : Logistic Regression
alpha = 0.1
MaxEpoch = 1000
batch_size = 50
epoch_best, w, b, logistic_training_loss, logistic_validation_accs = train_logistic_regression(X_train, t_train)
t_hat = predict_logistic_regression(X_train, w, b)
print("Best epoch: ", epoch_best)
print("Accuracy of logistic regression on D_train:", get_accuracy(t_hat, t_train))

plot_training_loss(logistic_training_loss, "Logistic Regression")
plot_validation_accuracy(logistic_validation_accs, "Logistic Regression")

t_hat_test = predict_logistic_regression(X_test,w,b )
print("Accuracy of Logistic regression on D_test:", get_accuracy(t_hat_test, t_test))

decay_vals = [0,0.1,0.005, 0.0005, 0.0001]        # weight decay
alpha_list = [0.1, 0.001, 0.0001]

train_losses ={}
valid_acc = {} 

for alpha in alpha_list:
    # TODO: report 3 number, plot 2 curves
    epoch_best, W_best, b_best,train_losses_alpha, valid_accs_alpha = train_logistic_regression(X_train, t_train)
    train_losses[alpha] = train_losses_alpha
    valid_acc[alpha] = valid_accs_alpha

X = np.linspace(0,MaxEpoch,MaxEpoch,endpoint=True).reshape([MaxEpoch,1])
plt.figure()
plt.xlabel('epoch')
plt.ylabel('training loss')
for (alpha, losses) in train_losses.items():
    plt.plot(X, train_losses[alpha], label= "alpha: "+str(alpha))
plt.legend(loc = "upper right")
plt.savefig('Logistic_Training_loss_alpha.jpg')

plt.figure()
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
for (alpha, acc) in valid_acc.items():
    plt.plot(X, valid_acc[alpha], label= "alpha: "+str(alpha))
plt.legend(loc = "upper right")
plt.savefig('Logistic_Validation_Accuracy_alpha.jpg')












