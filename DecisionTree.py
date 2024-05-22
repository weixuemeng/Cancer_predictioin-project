import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core  import Dense, Activation
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import cross_val_score



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

batch_size = 50

# Decision tree classifier 
# Tune hyperparameter: Max_depth
MaxEpoch= 50
test = []
for i in range(10):
    tree_classifier = DecisionTreeClassifier(max_depth= i+1,criterion="entropy", random_state= 30, splitter="random")
    tree_classifier = tree_classifier.fit(X_train, t_train)
    X_pred = tree_classifier.predict(X_test)
    score = accuracy_score(t_test, X_pred)
    test.append(score)

print("Max accuracy in test set: ", max(test))

plt.xlabel('max_depth')
plt.ylabel('Accuracy Score')
plt.plot(range(1,11),test, color = "red", label = "max_depth")
plt.legend()
plt.savefig('DecisionTree_Tunning.jpg')




