import numpy as np
import os
import sys
import plotly.graph_objects as go
import re

Task = 'Task1'
PATH_DIR = os.path.join(Task,'drcif', 'probs')

num_test_data = np.load(os.path.join(PATH_DIR, 'probs_10.npy')).shape[0]
num_classes = np.load(os.path.join(PATH_DIR, 'probs_10.npy')).shape[1]
print(f'number of test_data is:{num_test_data}')
print(f'number of classes is:{num_classes}')

def numerical_sort(value):
    # Extract the numerical part from filenames matching the pattern 'probs_<number>.npy'
    return int(value.split('_')[1].split('.')[0])

# Regular expression to match files starting with 'probs_' followed by digits
pattern = re.compile(r'^probs_\d+\.npy$')

# Filter files that match the pattern
files = [f for f in os.listdir(PATH_DIR) if pattern.match(f)]

list_probs = []
listdir = sorted(files, key=numerical_sort)

for f in listdir:
    if f.startswith('probs_'):
        print(f)
        probs = np.load(os.path.join(PATH_DIR, f))
        list_probs.append(probs)
        print('finish')


list_probs_over_time = []
for idx in range(num_test_data):
    temp_idx=[]
    for c in range(num_classes):
        temp_class = []
        for prob in list_probs:
            temp = prob[idx][c]
            temp_class.append(temp)
        temp_idx.append(temp_class)
    list_probs_over_time.append(temp_idx)
    print(f'idx{idx} is over')

np.save(os.path.join(PATH_DIR, 'probs_over_time.npy'), list_probs_over_time)        # shape:(274, num_classes, num_timesteps)
print('finish')

