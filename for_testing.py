import os
import numpy as np

for f in range(12):  # 12 features: from 0 to 11
    row = int(f / 3) + 1
    col = (f % 3) + 1
    print(f'row={row}')
    print(f'col={col}')

pdp = np.load('D:\PINT\pint_tool_2024\pint_tool_2024_v1_drcif_T1_R2G_notSynch\data_and_preprocessing\Task1\drcif\\runs_PDP_Task1_drcif\PDP_currentrange_20\PDP_drcif\PDP_values_Task1_20_feature0.npy')
p = pdp[0, :]
pp = pdp[:, 0]
probs_over_time = np.load('data_and_preprocessing/Task1/drcif/probs/probs_over_time.npy')

for i in range(144):
    idx = (i+1)*10
    probs = np.load(f'data_and_preprocessing/Task1/drcif/probs/probs_{idx}.npy')

probs_10 = np.load('data_and_preprocessing/Task1/drcif/probs/probs_10.npy')
probs_120 = np.load('data_and_preprocessing/Task1/drcif/probs/probs_120.npy')
probs_130 = np.load('data_and_preprocessing/Task1/drcif/probs/probs_130.npy')
probs_140 = np.load('data_and_preprocessing/Task1/drcif/probs/probs_140.npy')


print('finish')