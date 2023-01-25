# Import the necessary modules
from osgeo import gdal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
import time


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


datasetTest = gdal.Open('C:/Users/gozud/Desktop/MLProject/ProjectFiles/S2B_MSIL1C_20220528_Test.tif')
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

#------------------------- Normalization and removing zero items------------------------------------------------------
mask = dataTraining1d[:,3] != 0

# Use the mask to keep only the non-zero rows in the data and labels
dataTraining1d = dataTraining1d[mask]
trainGT1d = trainGT1d[mask]

#Normalize Data between 0 and 1 before using
dataTest1d = dataTest1d.astype(float) / 10000
dataTraining1d = dataTraining1d.astype(float) / 10000



X_Train = dataTraining1d
y_Train = trainGT1d
#--------------------------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(X_Train, y_Train, stratify = y_Train,test_size=0.010)


rf = RandomForestClassifier(n_jobs=-1)

start_time = time.time()

rf.fit(X_train, y_train.ravel())

elapsed_time = time.time() - start_time


y_pred = rf.predict(X_val)
y_pred = y_pred.reshape(-1, 1)
accuracy = accuracy_score(y_val, y_pred.ravel())
print(f'Accuracy: {accuracy:.2%}')


labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice','Permanent water bodies', 'Herbaceous wetland']

cm = confusion_matrix(y_val, y_pred.ravel())
print(classification_report(y_val, y_pred,target_names=labels))

# print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot()

predictions = rf.predict(dataTest1d)
df = pd.DataFrame(predictions)
df.columns=['Code']
df.insert(0, 'Id', range(1, len(df) + 1))

# Export the DataFrame as a CSV file
df.to_csv('C:/Users/gozud/Desktop/MLProject/ProjectFiles/DataCSV/submission.csv', index=False)

