
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd


# =====================================================new functions:
# Function to compute the global min and max for the entire DataFrame
def get_global_min_max(df):
    all_values = pd.concat([pd.concat(df[col].to_list()) for col in df.columns])  # Concatenate all series from all columns
    global_min = all_values.min()
    global_max = all_values.max()
    return global_min, global_max

# Function to normalize a series based on the global min and max
def normalize_series(series, global_min, global_max):
    return (series - global_min) / (global_max - global_min)


def read_data_within_window(groups, window_range, features):
    data = pd.DataFrame()
    labels = []

    for f in features:
        temp_feature = []
        for key, value in groups.items():
            temp = value[f]
            if len(value) < window_range[1]:            # this time series is shorter than window length
                original_indices = np.linspace(0, 1, len(value))
                target_indices = np.linspace(0, 1, window_range[1])
                # temp_df[col] = np.interp(target_indices, original_indices, value[col])
                temp_interpolated = np.interp(target_indices, original_indices, temp)
                temp_feature.append(pd.Series(temp_interpolated))
            else:
                temp_feature.append(temp[window_range[0]: window_range[1]])         # data per feature according to window range
        data[f] = temp_feature

    for key, value in groups.items():
        labels.append(value['label_obj_side_task_int'][0])

    return data, labels

