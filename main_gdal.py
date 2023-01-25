# Import the necessary modules
from osgeo import gdal
import numpy as np
import pandas as pd


trainGT2d = np.load('C:/Users/gozud/Desktop/MLProject/ProjectFiles/DataNumpy/test.npy')
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
trainGT1d = trainGT2d.reshape(trainGT2d.shape[0] * trainGT2d.shape[1], 1)

dfTrainLabels = pd.DataFrame(trainGT1d)

np.save('train_gt.npy', trainGT1d)

dataTraing = np.load('C:/Users/gozud/Desktop/MLProject/ProjectFiles/DataNumpy/train.npy')
dataTraining1d = dataTraing.reshape(dataTraing.shape[0] * dataTraing.shape[1], -1)
dfTrain = pd.DataFrame(dataTraing)

final_data = pd.concat([dfTrainLabels, dfTrain])

train_label_data = pd.concat([dfTrainLabels, dfTrain], axis=1)
train_label_data.columns=['Code', 'Blue', 'Green', 'Red', 'NIR']
train_label_data.to_csv('train.csv')

np.save('train.npy', dataTraining1d)


testGT2d = np.load('C:/Users/gozud/Desktop/MLProject/ProjectFiles/DataNumpy/test.npy')
testGT1d = testGT2d.reshape(testGT2d.shape[0] * testGT2d.shape[1], 1)

df = pd.DataFrame(testGT2d)

df.to_csv('test_gt.csv')
np.save('test_gt.npy', testGT1d)

dataTest2d = np.load('C:/Users/gozud/Desktop/MLProject/ProjectFiles/DataNumpy/test.npy')
dataTest1d = dataTest2d.reshape(dataTest2d.shape[0] * dataTest2d.shape[1], -1)
np.save('test_all.npy', dataTest1d)
dfTest = pd.DataFrame(dataTest2d)
dfTest.columns=['Blue', 'Green', 'Red', 'NIR']
dfTest.to_csv('test.csv')


from sklearn.model_selection import train_test_split
X_Test, X_Val, y_test, y_val = train_test_split(dataTest1d, testGT1d, test_size=0.30,random_state=42)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Create the KNN classifier with k=1
clf = KNeighborsClassifier(n_neighbors=1)

# Use cross-validation to evaluate the model's accuracy
# scores = cross_val_score(clf, dataTraining1d, np.ravel(trainGT1d), cv=5)
# acc = scores.mean()


# Fit the classifier to the data
clf.fit(dataTraining1d, np.ravel(trainGT1d))

# Predict labels for new data
predictions = clf.predict(dataTest1d[:1000])

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(testGT1d[:1000], predictions)
print(f'Accuracy: {accuracy:.2%}')

# Compute the confusion matrix
labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice','Permanent water bodies', 'Herbaceous wetland']

cm = confusion_matrix(testGT1d[:1000], predictions)
print(classification_report(testGT1d[:1000], predictions,target_names=labels))
# print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot()

df = pd.DataFrame(predictions)
df.columns=['Code']
# Export the DataFrame as a CSV file
df.to_csv('submission.csv')


from sklearn.ensemble import RandomForestClassifier

# Create the random forest classifier
clf = RandomForestClassifier()

# Use cross-validation to evaluate the model's accuracy
# scores = cross_val_score(clf, dataTraining1d, np.ravel(trainGT1d), cv=5)
# acc = scores.mean()

# Fit the classifier to the data
clf.fit(dataTraining1d, np.ravel(trainGT1d))

# Predict labels for new data
predictions = clf.predict(dataTest1d)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(testGT1d, predictions)
print(f'Accuracy: {accuracy:.2%}')

# Compute the confusion matrix
labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice','Permanent water bodies', 'Herbaceous wetland']

cm = confusion_matrix(testGT1d, predictions)
print(classification_report(testGT1d, predictions,target_names=labels))
# print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot()

df = pd.DataFrame(predictions)
df.columns=['Code']
# Export the DataFrame as a CSV file
df.to_csv('C:/Users/gozud/Desktop/MLProject/ProjectFiles/submission.csv')

















