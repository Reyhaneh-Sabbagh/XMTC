
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import random

from sktime.classification.interval_based import DrCIF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import functions_slidingWindow as func
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
print(sys.version)
# Prepare data
ROOT_PATH = 'Data/'

FEATURES = ['tiax', 'tiay', 'tiaz', 'tmax', 'tmay', 'tmaz', 'trax', 'tray', 'traz', 'tlax', 'tlay', 'tlaz']
RANGE = [0, 10]                # window size
STEP = 10                # step for incrementally increasing window size
TEST_PERCENTAGE = 0.2
Task = 'Task1'          # select Task (Task1, Task2, Task3)

result_path = 'Results_slidingWindow_DrCIF'
os.makedirs(result_path, exist_ok=True)
print(f'{result_path} created.')


# set seed everywhere:
np.random.seed(0)
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


# read data STEP-wise and move the window according to STEP:
test_acc_all_windows = []
train_acc_all_windows = []
val_acc_all_windows = []

current_range = RANGE
for i in range(RANGE[0], max_length, STEP):
    print(f'**************current_range:{current_range}******************************************')
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
    print(f'global_min_train:{global_min_train}, global_max_train:{global_max_train}')

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    np.save(os.path.join(result_path, f'global_train_min_{Task}_{current_range}.npy'), global_min_train)
    np.save(os.path.join(result_path, f'global_train_max_{Task}_{current_range}.npy'), global_max_train)

    print(f'x_train shape is: {x_train.shape}')
    print(f'x_test shape is: {x_test.shape}')
    print(f'*{x_train[FEATURES[0]].iloc[0].shape}')
    print(f'**{x_train[FEATURES[1]].iloc[0].shape}')
    print(f'***{x_train[FEATURES[2]].iloc[0].shape}')
    print(f'Number of classes in train_data: {np.unique(y_train, return_counts=True)}')
    print(f'Number of classes in test_data: {np.unique(y_test, return_counts=True)}')

    n_classes = len(np.unique(labels))

    drcif = DrCIF(random_state=0, time_limit_in_minutes=1)

    start = time.time()
    print('training is starting...')
    # Train the model
    drcif.fit(x_train, y_train)
    end = time.time()
    print('start to train time in seconds:')
    print(end - start)

    # load model:
    # print('start to load model')
    # drcif = joblib.load('drcif_model.pkl')

    # Predict the labels for the test set
    label_order = [0, 2, 4, 6, 1, 3, 5, 7]
    y_pred_test = drcif.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"test accuracy: {test_acc:.2f}")
    print(classification_report(y_test, y_pred_test))
    plt.figure()
    cm = confusion_matrix(y_test, y_pred_test)          ####### double check confusionmatrices for train and test
    # Reorder the rows and columns of the confusion matrix
    cm_reordered = cm[np.ix_(label_order, label_order)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_reordered, display_labels=label_order)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('test_data')
    plt.savefig(os.path.join(result_path, f'confusion_matrix_test_{Task}_{current_range}'), dpi=300)
    plt.figure()
    cm1 = confusion_matrix(y_test, y_pred_test, normalize='true')
    cm_reordered1 = (cm1[np.ix_(label_order, label_order)])*100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_reordered1, display_labels=label_order)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
    disp.plot()
    plt.title('test_data_percentage')
    plt.savefig(os.path.join(result_path, f'confusion_matrix_test_percentage_{Task}_{current_range}'), dpi=300)
    # plt.show()

    # Predict the labels for the train set
    y_pred_train = drcif.predict(x_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"train accuracy using model prediction for training data: {train_acc:.2f}")
    # Optional: Get detailed performance report
    print(classification_report(y_train, y_pred_train))
    plt.figure()
    cm2 = confusion_matrix(y_train, y_pred_train)
    cm_reordered2 = cm2[np.ix_(label_order, label_order)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_reordered2, display_labels=label_order)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
    disp.plot()
    plt.title('train_data')
    plt.savefig(os.path.join(result_path, f'confusion_matrix_train_{Task}_{current_range}'), dpi=300)
    plt.figure()
    cm3 = confusion_matrix(y_train, y_pred_train, normalize='true')
    cm_reordered3 = (cm3[np.ix_(label_order, label_order)])*100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_reordered3, display_labels=label_order)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm3)
    disp.plot()
    plt.title('train_data_percentage')
    plt.savefig(os.path.join(result_path, f'confusion_matrix_train_percentage_{Task}_{current_range}'), dpi=300)
    # plt.show()

    # # save model:
    joblib.dump(drcif, os.path.join(result_path,f'drcif_model_{Task}_{current_range}.pkl'))  # , compress=4

    test_acc_all_windows.append(test_acc)
    train_acc_all_windows.append(train_acc)

    plt.figure()
    plt.plot(train_acc_all_windows, label='Training accuracy')
    plt.plot(test_acc_all_windows, label='Test accuracy')
    plt.title(f'Accuracy_all_windows_{Task}_{RANGE}_{STEP}_currentrange:{current_range}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_path,f'accuracy_{Task}_{current_range}.png'), dpi=300)
    # plt.show()

    np.save(os.path.join(result_path, f'test_acc_all_windows_{Task}_{current_range}.npy'), test_acc_all_windows)
    np.save(os.path.join(result_path, f'train_acc_all_windows_{Task}_{current_range}.npy'), train_acc_all_windows)

    current_range = [RANGE[0], current_range[1]+STEP]           # update current_range
    print('Finish')



print('all runs finished')

