# -*- coding: utf-8 -*-
"""ML Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yAlw0MMT3mlAiSGVr1BIjs9HMbKnzX1K

CombineScript
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#                                   ES335- Machine Learning- Assignment 1
#
# This script combines the data from the UCI HAR Dataset into a more usable format.
# The data is combined into a single csv file for each subject and activity.
# The data is then stored in the Combined folder.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Library imports
import pandas as pd
import numpy as np
import os

# Give the path of the test and train folder of UCI HAR Dataset
train_path = "C:\\Users\\Sneha Gautam\\Downloads\\human+activity+recognition+using+smartphones\\UCI HAR Dataset\\UCI HAR Dataset\\train"
test_path = "C:\\Users\\Sneha Gautam\\Downloads\\human+activity+recognition+using+smartphones\\UCI HAR Dataset\\UCI HAR Dataset\\test"

# Dictionary of activities. Provided by the dataset.
ACTIVITIES = {
    1: 'WALKING'            ,
    2: 'WALKING_UPSTAIRS'   ,
    3: 'WALKING_DOWNSTAIRS' ,
    4: 'SITTING'            ,
    5: 'STANDING'           ,
    6: 'LAYING'             ,
}

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                        # Combining Traing Data
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Load all the accelerometer data

# Assuming full paths to the three files
total_acc_x_path = "/content/total_acc_x_train.txt"
total_acc_y_path = "/content/total_acc_y_train.txt"
total_acc_z_path = "/content/total_acc_z_train.txt"


# Read the files directly
total_acc_x = pd.read_csv(total_acc_x_path, delim_whitespace=True, header=None)
total_acc_y = pd.read_csv(total_acc_y_path, delim_whitespace=True, header=None)
total_acc_z = pd.read_csv(total_acc_z_path, delim_whitespace=True, header=None)


# Read the subject IDs
subject_train = pd.read_csv("/content/subject_train.txt",delim_whitespace=True,header=None)

# Read the labels
y = pd.read_csv("/content/y_train.txt",delim_whitespace=True,header=None)


# Toggle through all the subjects.
for subject in np.unique(subject_train.values):

    sub_idxs = np.where( subject_train.iloc[:,0] == subject )[0]
    labels = y.loc[sub_idxs]

    # Toggle through all the labels.
    for label in np.unique(labels.values):

        # make the folder directory if it does not exist
        if not os.path.exists(os.path.join("Combined","Train",ACTIVITIES[label])):
            os.makedirs(os.path.join("Combined","Train",ACTIVITIES[label]))

        label_idxs = labels[labels.iloc[:,0] == label].index

        accx = []
        accy = []
        accz = []

        for idx in label_idxs:
            if accx is not None:
                accx = np.hstack((accx,total_acc_x.loc[idx][64:]))
                accy = np.hstack((accy,total_acc_y.loc[idx][64:]))
                accz = np.hstack((accz,total_acc_z.loc[idx][64:]))

            else:
                accx = total_acc_x.loc[idx]
                accy = total_acc_y.loc[idx]
                accz = total_acc_z.loc[idx]

        # saving the data into csv file
        data = pd.DataFrame({'accx':accx,'accy':accy,'accz':accz})
        save_path = os.path.join("Combined","Train",ACTIVITIES[label],f"Subject_{subject}.csv")
        data.to_csv(save_path,index=False)

print("Done Combining the training data")


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                        # Combining Test Data
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Load all the accelerometer data

total_acc_x_path = "/content/total_acc_x_test.txt"
total_acc_y_path = "/content/total_acc_y_test.txt"
total_acc_z_path = "/content/total_acc_z_test.txt"

total_acc_x = pd.read_csv(total_acc_x_path, delim_whitespace=True, header=None)
total_acc_y = pd.read_csv(total_acc_y_path, delim_whitespace=True, header=None)
total_acc_z = pd.read_csv(total_acc_z_path, delim_whitespace=True, header=None)

# Read the subject IDs
subject_test = pd.read_csv("/content/subject_test.txt",delim_whitespace=True,header=None)

# Read the labels
y = pd.read_csv("/content/y_test.txt",delim_whitespace=True,header=None)

# Toggle through all the subjects.
for subject in np.unique(subject_test.values):

        sub_idxs = np.where( subject_test.iloc[:,0] == subject )[0]
        labels = y.loc[sub_idxs]

        # Toggle through all the labels.
        for label in np.unique(labels.values):

            if not os.path.exists(os.path.join("Combined","Test",ACTIVITIES[label])):
                os.makedirs(os.path.join("Combined","Test",ACTIVITIES[label]))

            label_idxs = labels[labels.iloc[:,0] == label].index

            accx = []
            accy = []
            accz = []
            for idx in label_idxs:
                if accx is not None:
                    accx = np.hstack((accx,total_acc_x.loc[idx][64:]))
                    accy = np.hstack((accy,total_acc_y.loc[idx][64:]))
                    accz = np.hstack((accz,total_acc_z.loc[idx][64:]))

                else:
                    accx = total_acc_x.loc[idx]
                    accy = total_acc_y.loc[idx]
                    accz = total_acc_z.loc[idx]

            # saving the data into csv file
            data = pd.DataFrame({'accx':accx,'accy':accy,'accz':accz})
            save_path = os.path.join("Combined","Test",ACTIVITIES[label],f"Subject_{subject}.csv")
            data.to_csv(save_path,index=False)

print("Done Combining the testing data")
print("Done Combining the data")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

"""MakeDataset"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#                                   ES335- Machine Learning- Assignment 1
#
# This file is used to create the dataset for the mini-project. The dataset is created by reading the data from
# the Combined folder. The data is then split into training, testing, and validation sets. This split is supposed
# to be used for all the modeling purposes.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Library imports
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# Constants
time = 10
offset = 100
folders = ["LAYING","SITTING","STANDING","WALKING","WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]
classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}

combined_dir = os.path.join("Combined")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Train Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X_train=[]
y_train=[]
dataset_dir = os.path.join(combined_dir,"Train")

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))

    for file in files:

        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        df = df[offset:offset+time*50]
        X_train.append(df.values)
        y_train.append(classes[folder])

X_train = np.array(X_train)
y_train = np.array(y_train)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Test Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X_test=[]
y_test=[]
dataset_dir = os.path.join(combined_dir,"Test")

for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))
    for file in files:

        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        df = df[offset:offset+time*50]
        X_test.append(df.values)
        y_test.append(classes[folder])

X_test = np.array(X_test)
y_test = np.array(y_test)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Final Dataset
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# USE THE BELOW GIVEN DATA FOR TRAINING, TESTING, AND VALIDATION purposes

# concatenate the training and testing data
X = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))

# split the data into training,testing, and validation sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4,stratify=y)
X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=4,stratify=y_test)

print("Training data shape: ",X_train.shape)
print("Testing data shape: ",X_test.shape)
print("Validation data shape: ",X_val.shape)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

"""Q 3. Train Decision Tree using trainset and report Accuracy and confusion matrix using testset."""

# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ... (previous code)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Train Decision Tree Classifier
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Create a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=4)

# Flatten the features to fit the model (assuming X_train, X_test, X_val are 3D arrays)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Train the Decision Tree model
dt_model.fit(X_train_flat, y_train)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Evaluate Decision Tree Classifier
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Predict on the testing set
y_pred = dt_model.predict(X_test_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

"""Q 4. Train Decision Tree with varrying depths (2-8) using trainset and report accuracy and confusion matrix using Test set. Does the accuracy changes when the depth is increased? Plot the accuracies and reason why such a result has been obtained."""

# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ... (previous code)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Train Decision Trees with Varying Depths
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

depths = range(2, 9)
accuracies = []

for depth in depths:
    # Create a Decision Tree model with varying depth
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=4)

    # Flatten the features to fit the model
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Train the Decision Tree model
    dt_model.fit(X_train_flat, y_train)

    # Predict on the testing set
    y_pred = dt_model.predict(X_test_flat)

    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Print confusion matrix for depth 4 (you can comment this line for other depths)
    if depth == 4:
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix (Depth {depth}):")
        print(conf_matrix)

# Print accuracy for depth 4
depth_4_accuracy = accuracies[depths.index(4)]
print(f"Accuracy for Depth 4: {depth_4_accuracy}")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Plot Accuracies vs Depths
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

plt.plot(depths, accuracies, marker='o')
plt.title('Accuracy vs Depth for Decision Trees')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

"""Accuracy changes with depth because :
When the tree depth is too shallow, the model may be too simple and not capture the underlying patterns in the data.
 If the tree depth is too high, the model may become overly complex and start to memorize the training data rather than learning general patterns.

Q 5. Use PCA (Principal Component Analysis) on Total Acceleration
 to compress the acceleration timeseries into two features and plot a scatter plot to visualize different class of activities. Next, use TSFEL (a featurizer library) to create features (your choice which ones you feel are useful) and then perform PCA to obtain two features. Plot a scatter plot to visualize different class of activities. Are you able to see any difference?
"""

!pip install tsfresh

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming your X_train is a 3D array with shape (num_samples, num_timesteps, 3)
# Assuming your y_train is a 1D array with shape (num_samples,) containing activity labels
# Each sample in X_train corresponds to an activity label in y_train

# Generate synthetic data for testing
np.random.seed(42)
num_samples = 100
num_timesteps = 50
num_activities = 6
X_train = np.random.randn(num_samples, num_timesteps, 3)
y_train = np.random.randint(0, num_activities, size=num_samples)  # Replace with actual labels

# Reshape the data to have each sample as a single vector
X_train_reshaped = X_train.reshape(num_samples, -1)

# Calculate the total acceleration
total_acceleration = np.sum(X_train**2, axis=2)

# Apply PCA to compress the data into two features
pca = PCA(n_components=2)
compressed_data = pca.fit_transform(total_acceleration)

# Create a list of activity names (replace with your actual activity names)
activity_names = [f'Activity {i+1}' for i in range(num_activities)]

# Scatter plot to visualize different classes of activities with different colors
plt.figure(figsize=(10, 6))
for i in range(num_activities):
    indices = np.where(y_train == i)
    plt.scatter(compressed_data[indices, 0], compressed_data[indices, 1], label=activity_names[i])

plt.title('Scatter Plot of Compressed Acceleration Data using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Extract features using TSFRESH
X_features_magnitude = extract_features(df_magnitude, column_id='id', column_sort='time')

X_features_magnitude = pd.read_csv('/content/extracted_features.csv') #(df_magnitude, column_id='id', column_sort='time')

# X_features_magnitude.to_csv('extracted_features.csv', index=False)

# X_features_magnitude = pd.read_csv('/content/extracted_features.csv') #(df_magnitude, column_id='id', column_sort='time')


# Impute missing values using mean (you can choose a different strategy)
imputer = SimpleImputer(strategy='mean')
X_features_magnitude_imputed = imputer.fit_transform(X_features_magnitude)


# Standardize the features after imputation
scaler_magnitude = StandardScaler()
X_features_scaled_magnitude = scaler_magnitude.fit_transform(X_features_magnitude_imputed)


# Standardize the features before applying PCA
scaler_magnitude = StandardScaler()
X_features_scaled_magnitude = scaler_magnitude.fit_transform(X_features_magnitude)

if np.isnan(X_features_scaled_magnitude).any():
    # Handle NaN values (e.g., replace with a constant value or use another imputation strategy)
    X_features_scaled_magnitude = np.nan_to_num(X_features_scaled_magnitude)


imputer_scaled_magnitude = SimpleImputer(strategy='constant', fill_value=0)

# Apply PCA on extracted features
pca_features_magnitude = PCA(n_components=2)
X_pca_features_magnitude = pca_features_magnitude.fit_transform(X_features_scaled_magnitude)

# Plot scatter plot
plt.figure(figsize=(10, 8))
for label in np.unique(y_train):
    plt.scatter(X_pca_features_magnitude[y_train == label, 0], X_pca_features_magnitude[y_train == label, 1], label=f'Activity {label}')
plt.title('PCA on TSFRESH Extracted Features from Magnitude of Total Acceleration')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

"""Graphs of static activities such as standing, laying and sitting are much less scattered in the featured data

Q 6. Use the features obtained from TSFEL and train a Decision Tree. Report the accuracy and confusion matrix using test set. Does featurizing works better than using the raw data? Train Decision Tree with varrying depths (2-8) and compare the accuracies obtained in Q4 with the accuracies obtained using featured trainset. Plot the accuracies obtained in Q4 against the accuracies obtained in this question.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming you have a dataset X_features and y (features and labels)
# Split the data into training and testing sets
# X_features_magnitude_imputed.reset_index(drop=True, inplace=True)


print(X_features_magnitude_imputed.shape)
print(len(y))
y_array = np.array(y)
y_array = y_array[:X_features_magnitude_imputed.shape[0]]

X_train, X_test, y_train, y_test = train_test_split(X_features_magnitude_imputed, y_array, test_size=0.2, random_state=42)

# Train a Decision Tree classifier on the featured data
dt_classifier_featured = DecisionTreeClassifier(random_state=42)
dt_classifier_featured.fit(X_train, y_train)

# Make predictions on the test set
y_pred_featured = dt_classifier_featured.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy_featured = accuracy_score(y_test, y_pred_featured)
conf_matrix_featured = confusion_matrix(y_test, y_pred_featured)

# Print accuracy and confusion matrix
print("Accuracy (Featured Data):", accuracy_featured)
print("Confusion Matrix (Featured Data):\n", conf_matrix_featured)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming you have a dataset X_features and y (features and labels)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features_magnitude_imputed, y_array, test_size=0.2, random_state=42)

# Train Decision Trees with varying depths
for depth in range(2, 9):
    # Create a Decision Tree classifier with the current depth
    dt_classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # Train the classifier on the training data
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = dt_classifier.predict(X_test)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results for the current depth
    print(f"\nDecision Tree with max depth {depth}:")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming you have a dataset X_features and y (features and labels)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features_magnitude_imputed, y_array, test_size=0.2, random_state=42)

# Train Decision Trees with varying depths
for depth in range(2, 9):
    # Create a Decision Tree classifier with the current depth
    dt_classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # Train the classifier on the training data
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the training set
    y_pred_train = dt_classifier.predict(X_train)

    # Calculate accuracy on the training set
    accuracy_train = accuracy_score(y_train, y_pred_train)

    # Print results for the current depth
    print(f"\nDecision Tree with max depth {depth}:")
    print("Accuracy on Training Set:", accuracy_train)

    # (Optional) Print accuracy and confusion matrix on the test set
    y_pred_test = dt_classifier.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    print("Accuracy on Test Set:", accuracy_test)
    print("Confusion Matrix on Test Set:")
    print(conf_matrix_test)

# Plot the accuracies obtained in Question 4 against the accuracies obtained in the next code
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Assuming you have a dataset X_features and y (features and labels)
# Split the data into training and testing sets for raw features
# y_array = y_array[:X_features_magnitude_imputed.shape[0]]
y_array = y_array[:X_train.shape[0]]

# Reshape time series data to 2D
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_train_flattened, y_array, test_size=0.2, random_state=42)

# Train a Decision Tree classifier on the raw data
dt_classifier_raw = DecisionTreeClassifier(random_state=42)
dt_classifier_raw.fit(X_train_raw, y_train_raw)

# Make predictions on the test set for raw features
y_pred_raw = dt_classifier_raw.predict(X_test_raw)

# Calculate accuracy and confusion matrix for raw features
accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)
conf_matrix_raw = confusion_matrix(y_test_raw, y_pred_raw)

# Print accuracy and confusion matrix for raw features
print("Accuracy (Raw Data):", accuracy_raw)
print("Confusion Matrix (Raw Data):\n", conf_matrix_raw)

# Now, you can plot the accuracies obtained from both raw and featured data
plt.plot(depths, accuracies, marker='o', label='Decision Tree on Featured Data')
plt.axhline(y=accuracy_raw, color='r', linestyle='--', label='Decision Tree on Raw Data')
plt.title('Accuracy vs Depth for Decision Trees')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

"""Yes, featurizing gives better accuracy than the raw data

Q 7. Are there any participants/ activitivies where the Model performace is bad? If Yes, Why?
"""

from sklearn.metrics import classification_report

# Assuming you have y_test (true labels) and y_pred_featured (predicted labels)
# You can directly use your test data without splitting further for this analysis

# Generate the classification report
class_report = classification_report(y_test, y_pred_featured)

# Print the classification report
print("Classification Report (Featured Data):\n", class_report)

from sklearn.metrics import classification_report

class_report_dict = classification_report(y_test, y_pred_featured, output_dict=True)

# Extract precision, recall, F1-score, and support for each class
for activity in class_report_dict.keys():
    if activity not in ['accuracy', 'macro avg', 'weighted avg']:
        precision = class_report_dict[activity]['precision']
        recall = class_report_dict[activity]['recall']
        f1_score = class_report_dict[activity]['f1-score']
        support = class_report_dict[activity]['support']

        print(f"Activity: {activity}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, Support: {support}")
        print("\n")

"""Activity 1 (walking) and Activity 2 (walking_upstairs) have the least precision. It could be due to inconsistency while collecting the data"""
