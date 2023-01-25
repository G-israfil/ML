# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:07:22 2023

@author: gozud
"""

# Import the necessary modules
from osgeo import gdal
import numpy as np
import pandas as pd

# Open the GeoTIFF files using GDAL
datasetTrainingGT = gdal.Open('C:/Users/gozud/Desktop/MLProject/ProjectFiles/S2A_MSIL1C_20220516_Train_GT.tif')

# Read the data from the first GeoTIFF file into a NumPy array
trainGT2d = datasetTrainingGT.ReadAsArray()
trainGT2d = np.swapaxes(trainGT2d, 0, 1)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
trainGT1d = trainGT2d.reshape(trainGT2d.shape[0] * trainGT2d.shape[1], 1)

# Convert the combined array into a Pandas DataFrame
dfTrainLabels = pd.DataFrame(trainGT1d)

# Export the DataFrame as a CSV file
# dfTrainLabels.to_csv('train.csv', index=False)
np.save('train_gt.npy', trainGT1d)

datasetTraining = gdal.Open('C:/Users/gozud/Desktop/MLProject/ProjectFiles/S2A_MSIL1C_20220516_TrainingData.tif')

# Read the data from the first GeoTIFF file into a NumPy array
dataTraing = datasetTraining.ReadAsArray()
dataTraing = np.swapaxes(dataTraing, 0, 2)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
dataTraining1d = dataTraing.reshape(dataTraing.shape[0] * dataTraing.shape[1], -1)
dfTrain = pd.DataFrame(dataTraining1d)

final_data = pd.concat([dfTrainLabels, dfTrain])

train_label_data = pd.concat([dfTrainLabels, dfTrain], axis=1)
train_label_data.columns=['Code', 'Blue', 'Green', 'Red', 'NIR']
train_label_data.to_csv('train.csv')

np.save('train.npy', dataTraining1d)

datasetTestGT = gdal.Open('C:/Users/gozud/Desktop/MLProject/ProjectFiles/S2B_MSIL1C_20220528_Test.tif')
testGT2d = datasetTestGT.ReadAsArray()
testGT2d = testGT2d[0, :, :]
#testGT2d = testGT2d[1:, :]
testGT2d = np.swapaxes(testGT2d, 0, 1)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
testGT1d = testGT2d.reshape(testGT2d.shape[0] * testGT2d.shape[1], 1)



# Convert the combined array into a Pandas DataFrame
df = pd.DataFrame(testGT1d)

# Export the DataFrame as a CSV file
df.to_csv('test_gt.csv')
np.save('test_gt.npy', testGT1d)

datasetTest = gdal.Open('C:/Users/gozud/Desktop/MLProject/ProjectFiles/S2B_MSIL1C_20220528_Test.tif')

# Read the data from the first GeoTIFF file into a NumPy array
dataTest2d = datasetTest.ReadAsArray()
dataTest2d = np.swapaxes(dataTest2d, 0, 2)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
dataTest1d = dataTest2d.reshape(dataTest2d.shape[0] * dataTest2d.shape[1], -1)
np.save('test_all.npy', dataTest1d)
# Convert the combined array into a Pandas DataFrame
dfTest = pd.DataFrame(dataTest1d)
dfTest.columns=['Blue', 'Green', 'Red', 'NIR']
# Export the DataFrame as a CSV file
dfTest.to_csv('test.csv')


#Bagging for improved accuracy
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
# Create the random forest classifier


bag_clf = BaggingClassifier(base_estimator=RandomForestClassifier(), 
                            n_estimators=10, 
                            bootstrap=True, 
                            bootstrap_features=False, 
                            oob_score=True)
bag_clf.fit(dataTraining1d, np.ravel(trainGT1d))
bag_predictions = bag_clf.predict(dataTest1d)
bag_accuracy = accuracy_score(testGT1d, bag_predictions)
print(f'Accuracy: {bag_accuracy:.2%}')

# Compute the confusion matrix
labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice','Permanent water bodies', 'Herbaceous wetland']

cm = confusion_matrix(testGT1d, bag_predictions)
print(classification_report(testGT1d, bag_predictions,target_names=labels))
# print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot()

df = pd.DataFrame(bag_predictions)
df.columns=['Code']
# Export the DataFrame as a CSV file
df.to_csv('C:/Users/gozud/Desktop/MLProject/ProjectFiles/submission.csv')





