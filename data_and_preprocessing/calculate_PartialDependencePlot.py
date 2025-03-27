import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use "Agg" for non-GUI environments
import numpy as np
import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import joblib
import functions_slidingWindow as func
import warnings
import random
# https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay.from_estimator
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

warnings.filterwarnings("ignore", category=FutureWarning)
# Prepare data_and_preprocessing
ROOT_PATH = 'Data'

FEATURES = ['tiax', 'tiay', 'tiaz', 'tmax', 'tmay', 'tmaz', 'trax', 'tray', 'traz', 'tlax', 'tlay', 'tlaz']
RANGE = [0, 10]                # [0, 10]
STEP = 10                # step for incrementally increasing window size
TEST_PERCENTAGE = 0.2
Task = 'Task1'          # Specify task number

result_path = os.path.join(Task, 'drcif', 'PDP')
os.makedirs(result_path, exist_ok=True)
print(f'{result_path} created.')


# set seed everywhere:
np.random.seed(0)
# torch.manual_seed(0)
random.seed(0)

df = pd.read_csv(os.path.join(ROOT_PATH, f'R2G_features_t_normalized_{Task}.csv'))
gb = df.groupby(['task', 'userID', 'object', 'side', 'action', 'trialID'])
groups = dict(list(gb))


def calculate_partial_dependence(model, X, feature_idx, resolution=5):
# def calculate_partial_dependence(model, X, feature_idx, class_idx, resolution=30):

    """
    Calculate partial dependence for a single feature and class in multivariate time series.

    Parameters:
        model: Fitted DrCIF model.
        X: Input data (n_samples, n_features, series_length).
        feature_idx: Index of the feature for which PDP is computed.
        class_idx: Index of the class for which PDP is computed.
        resolution: Number of points to evaluate PDP at.

    Returns:
        feature_values: Array of feature values evaluated.
        pdp_values: Partial dependence values for 8 classes
    """
    n_samples, n_features, series_length = X.shape
    feature_values = np.linspace(X[:, feature_idx, :].min(), X[:, feature_idx, :].max(), resolution)
    pdp_values = []

    X_copy = X.copy()
    for value in feature_values:
        X_copy[:, feature_idx, :] = value
        predictions = model.predict_proba(X_copy)  # Predictions for all classes
        pdp_values.append(predictions.mean(axis=0))
        # for c_idx in range(8):              # Number of classes: 8
        #     pdp_values.append(predictions[:, c_idx].mean())  # Mean probability for the target class
        # pdp_values.append(predictions[:, class_idx].mean())  # Mean probability for the target class

    return feature_values, np.array(pdp_values)


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
for i in range(RANGE[0], max_length+STEP, STEP):
    # current_range[1] = min(i, max_length)
    print(f'**************current_range:{current_range}******************************************')
    # step through all folders and load model and predict labels
    model_path = os.path.join(f'runs', f'drcif_slidingWindow_currentrange_{current_range[1]}',
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
        # ==================Partial Dependence plot ===============
        #convert x_train to 3d numpy array:
        series_length = len(x_train.iloc[0, 0])  # Length of the first time series
        if not all(len(cell) == series_length for col in x_train for cell in x_train[col]):
            raise ValueError("All time series must have the same length.")

        # Create a 3D NumPy array
        n_samples = len(x_train)
        n_features = len(x_train.columns)
        x_train_np = np.zeros((n_samples, n_features, series_length))

        for i, row in enumerate(x_train.itertuples(index=False)):
            for j, cell in enumerate(row):
                x_train_np[i, j, :] = cell

        # convert x_test to 3d numpy array:
        series_length = len(x_test.iloc[0, 0])  # Length of the first time series
        if not all(len(cell) == series_length for col in x_test for cell in x_test[col]):
            raise ValueError("All time series must have the same length.")

        # Create a 3D NumPy array
        n_samples = len(x_test)
        n_features = len(x_test.columns)
        x_test_np = np.zeros((n_samples, n_features, series_length))

        for i, row in enumerate(x_test.itertuples(index=False)):
            for j, cell in enumerate(row):
                x_test_np[i, j, :] = cell

        n_features = x_test_np.shape[1]
        for feature_idx in range(n_features):
            print(f'feature:{feature_idx}')
            feature_values, pdp_values = calculate_partial_dependence(
                drcif, x_test_np, feature_idx
            )
            np.save(os.path.join(result_path, f'PDP_values_{Task}_{current_range}_feature{feature_idx}.npy'), pdp_values)
            np.save(os.path.join(result_path, f'PDP_feature_values_{Task}_{current_range}_feature{feature_idx}.npy'), feature_values)

        print('time for PDP calculation in seconds:')
        end = time.time()
        print(end - start)
    else:
        print(f'path does not exist')
    current_range[1] = current_range[1] + STEP



