# Import standard libraries
import os
import time
import shutil
import pathlib
import itertools

# Import data handling and visualization tools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Import machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Import deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('Modules loaded')
"""from os.path import exists

def is_file_imported_from_drive(file_name):
  
  Checks if a file is imported from Google Drive or not.

  Args:
    file_name: The name of the file to check.

  Returns:
    True if the file is imported from Google Drive, False otherwise.
  
  drive_mount_path = "/content/drive/My Drive"
  file_path = os.path.join(drive_mount_path, file_name)
  return exists(file_path)"""
import os
file_name = '/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set'
file_paths = []
labels = []


folds = os.listdir(file_name)
# print(folds)


for fold in folds :
    sub_fold_path = os.path.join(file_name , fold)
#     print(sub_fold_path)

    sub2folds= os.listdir(sub_fold_path)
#     print(sub2folds)

    for c_fold in sub2folds :
        c_sub_path = os.path.join(sub_fold_path,c_fold )
#         print(c_sub_path )

        subcfolds = os.listdir(c_sub_path)
#         print(subcfolds)

        for filepath in subcfolds :

            file_path = os.path.join(c_sub_path,filepath )
#             print(file_path)

            file_paths.append(file_path)
            labels.append(c_fold)

fseries=  pd.Series(file_paths, name = 'filepath')
lseries = pd.Series(labels, name= 'labels')

df= pd.concat([fseries, lseries ], axis = 1)

set(labels)

file_paths
 df

train_df, ts_df = train_test_split(df, test_size = 0.2 , random_state = 42, stratify = df['labels'])

valid_df, test_df = train_test_split(ts_df, test_size = 0.5 , random_state = 42, stratify  = ts_df['labels'])

train_df
'valid_df'
'test_df'
batch_size = 64
img_size =  (224,224)

geny = ImageDataGenerator()

train_geny = geny.flow_from_dataframe(train_df, x_col = 'filepath', y_col = 'labels', target_size= img_size , batch_size = batch_size, shuffle = True , class_mode = 'categorical', color_mode = 'rgb' )

valid_geny = geny.flow_from_dataframe(valid_df, x_col = 'filepath', y_col = 'labels', target_size= img_size , batch_size = batch_size, shuffle = True , class_mode = 'categorical', color_mode = 'rgb' )

test_geny = geny.flow_from_dataframe(test_df, x_col = 'filepath', y_col = 'labels', target_size= img_size , batch_size = batch_size, class_mode = 'categorical', color_mode = 'rgb' )
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adamax

model = Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(224, 224, 3)),
    Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),

    Flatten(),

    Dense(256, activation='relu'),
    Dense(128, activation='relu'),

    Dense(5, activation='softmax') # Adjusted to 5 output classes to match your target labels
])

# Compile the model
model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history=  model.fit(train_geny, epochs= 5,verbose= 1 ,validation_data = valid_geny, validation_steps= None, shuffle= False  )
# Assuming `model` is your trained model
y_pred = model.predict(test_geny)
y_pred_class = np.argmax(y_pred, axis=1)
# Calculate the number of steps for evaluation
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
test_steps = ts_length // test_batch_size

# Evaluate the model on the training, validation, and test datasets
train_score = model.evaluate(train_geny, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_geny, steps=test_steps, verbose=1)
test_score = model.evaluate(test_geny, steps=test_steps, verbose=1)

# Print the evaluation results
print("Train Loss: ", train_score[0] * 100)
print("Train Accuracy: ", train_score[1] * 100)
print('-' * 20)
print("Validation Loss: ", valid_score[0] * 100)
print("Validation Accuracy: ", valid_score[1] * 100)
print('-' * 20)
print("Test Loss: ", test_score[0] * 100)
print("Test Accuracy: ", test_score[1] * 100)