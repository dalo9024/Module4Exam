# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:29:02 2024

@author: Daniel Long
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


#load in dataset
data = pd.read_csv('data.csv')

#drop columns not needed for credit analysis
credit = data.drop(columns=['User', 'Purchase History'])

#seperate features and the label
X = credit[['Length Credit History (yrs)', 'Current Income (Thousands)']]
y = credit['Good Credit Customer']

#training and testing spplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#plot the training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train['Length Credit History (yrs)'], X_train['Current Income (Thousands)'],
            c=y_train.map({0: 'red', 1: 'blue'}), marker='o', edgecolor='k')
plt.xlabel('Length of Credit History (Years)')
plt.ylabel('Current Income (Thousands)')
plt.title('Training Data: Credit History vs. Income')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Bad Credit Customer'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Good Credit Customer')],
           loc='upper right')
plt.tight_layout()  
plt.show()

#decision tree with depth of 2
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

#visualization of decision tree
plt.figure(figsize=(10, 8))
tree.plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['Bad Credit', 'Good Credit'], rounded=True)
plt.title('Decision Tree to Predict Good Credit Customers')
plt.show()

#predictions of test data
y_pred = clf.predict(X_test)

#calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

#generate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad Credit', 'Good Credit'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()