
import numpy as np
import pandas as pd
import os
import sys

from sktime.classification.hybrid import HIVECOTEV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import functions_slidingWindow as func
import warnings
import random
warnings.filterwarnings("ignore", category=FutureWarning)
# Prepare data_and_preprocessing
ROOT_PATH = "/Data/"

FEATURES = ['tiax', 'tiay', 'tiaz', 'tmax', 'tmay', 'tmaz', 'trax', 'tray', 'traz', 'tlax', 'tlay', 'tlaz']
RANGE = [0, 10]                # [0, 10]
STEP = 10                # step for incrementally increasing window size
TEST_PERCENTAGE = 0.2
Task = 'Task1'          # set Task

result_path = os.path.join(Task, 'drcif', 'probs')
os.makedirs(result_path, exist_ok=True)
print(f'{result_path} created.')


# set seed everywhere:
np.random.seed(0)
# torch.manual_seed(0)
random.seed(0)


df = pd.read_csv(os.path.join(ROOT_PATH, f'R2G_features_t_normalized_{Task}.csv'))
gb = df.groupby(['task', 'userID', 'object', 'side', 'action', 'trialID'])
groups = dict(list(gb))

for key, value in groups.items():
    value.reset_index(drop=True, inplace=True)

# ==============================read whole time series to find the maximum length:
data_whole = pd.DataFrame()

for f in FEATURES:
    temp_feature = []
    for key, value in groups.items():
        temp = value[f]
        temp_feature.append(temp)
    data_whole[f] = temp_feature

max_length = data_whole[FEATURES[0]].apply(len).max()     # find max length
print(f'max_length:{max_length}')

current_range = RANGE
for i in range(RANGE[1], max_length+STEP, STEP):
    current_range[1] = min(i, max_length)
    print(f'**************current_range:{current_range}******************************************')
    # step through all folders and load model and predict labels
    model_path = os.path.join(f'runs_{Task}_drcif', f'drcif_slidingWindow_currentrange_{current_range[1]}',
                              'Results_slidingWindow_DrCIF',
                              f'drcif_model_{Task}_{current_range[1]}.pkl')
    print(f'model_path:{model_path}')
    if os.path.exists(model_path):
        data, labels = func.read_data_within_window(groups, current_range, FEATURES)
        indices_all = [idx for idx in range(0, len(data))]

        x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
            data, labels, indices_all, test_size=TEST_PERCENTAGE, stratify=labels, random_state=0, shuffle=True
        )

        # Find the global min and max for the entire DataFrame
        global_min_train, global_max_train = func.get_global_min_max(x_train)
        # Normalize each time series in each cell based on the global min and max
        x_train = x_train.applymap(lambda series: func.normalize_series(series, global_min_train, global_max_train))
        x_test = x_test.applymap(lambda series: func.normalize_series(series, global_min_train, global_max_train))
        # x_val = x_val.applymap(lambda series: func.normalize_series(series, global_min_train, global_max_train))
        print(f'global_min_train:{global_min_train}, global_max_train:{global_max_train}')

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # np.save(os.path.join(result_path, f'global_train_min_{Task}_{current_range}.npy'), global_min_train)
        # np.save(os.path.join(result_path, f'global_train_max_{Task}_{current_range}.npy'), global_max_train)

        print(f'x_train shape is: {x_train.shape}')
        print(f'x_test shape is: {x_test.shape}')
        # print(f'x_val shape is: {x_val.shape}')
        print(f'*{x_train[FEATURES[0]].iloc[0].shape}')
        print(f'**{x_train[FEATURES[1]].iloc[0].shape}')
        print(f'***{x_train[FEATURES[2]].iloc[0].shape}')
        print(f'Number of classes in train_data: {np.unique(y_train, return_counts=True)}')
        print(f'Number of classes in test_data: {np.unique(y_test, return_counts=True)}')
        # print(f'Number of classes in val_data: {np.unique(y_val, return_counts=True)}')

        n_classes = len(np.unique(labels))

        start = time.time()

        #load model:
        # Load the model from the file
        print('start to load model')
        drcif = joblib.load(model_path)      # , mmap_mode='r'
        # Predict the labels for the test set
        y_pred = drcif.predict(x_test)
        probs = drcif.predict_proba(x_test)
        print(probs)
        print('start to train time in seconds:')
        end = time.time()
        print(end - start)
        np.save(os.path.join(result_path, f'probs_{current_range[1]}.npy'), probs)
        np.save(os.path.join(result_path, f'y_test_{current_range[1]}.npy'), y_test)
        np.save(os.path.join(result_path, f'y_pred_{current_range[1]}.npy'), y_pred)
    else:
        print(f'path does not exist')
    current_range[1] = current_range[1] + STEP

