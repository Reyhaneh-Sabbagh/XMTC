
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import plotly
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
np.random.seed(0)
import random
random.seed(0)

# Prepare data
ROOT_PATH = 'Data'
Task = 'Task3'          # specify Task number
FEATURES = ['tiax', 'tiay', 'tiaz', 'tmax', 'tmay', 'tmaz', 'trax', 'tray', 'traz', 'tlax', 'tlay', 'tlaz']

df = pd.read_csv(os.path.join(ROOT_PATH, f'R2G_features_t_normalized_{Task}.csv'))
gb = df.groupby(['task', 'userID', 'object', 'side', 'action', 'trialID'])
groups = dict(list(gb))

for key, value in groups.items():
    value.reset_index(drop=True, inplace=True)

# ======== Histogram of all data: ==============
sequence_lengths = [len(value) for key, value in groups.items()]
path = os.path.join(Task, 'drcif', 'Histogram')
os.makedirs(path, exist_ok=True)
np.save(os.path.join(path, 'sequence_lengths.npy'), sequence_lengths)

print(f'sequence_lengths has saved in:{path}')

# fig = go.Figure(data=[go.Histogram(x=sequence_lengths, xbins=dict(start=0, end=max(sequence_lengths), size=10))])
# fig.show()

# # ==========Histogram of test dataset ==========

data = pd.DataFrame()
labels = []

for f in FEATURES:
    temp_feature = []
    for key, value in groups.items():
        temp = value[f]
        temp_feature.append(temp)         # data per feature according to window range
    data[f] = temp_feature

for key, value in groups.items():
    labels.append(value['label_obj_side_task_int'][0])

indices_all = [idx for idx in range(0, len(groups))]
x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
    data, labels, indices_all, test_size=0.2, stratify=labels, random_state=0, shuffle=True
)

max_length = data[FEATURES[0]].apply(len).max()     # find max length
print(f'max_length:{max_length}')

sequence_lengths_test_data = x_test[FEATURES[0]].apply(len)
path_test = os.path.join(Task, 'drcif', 'Histogram')
os.makedirs(path_test, exist_ok=True)
np.save(os.path.join(path_test, 'sequence_lengths_test_data.npy'), sequence_lengths_test_data)

print('finish')