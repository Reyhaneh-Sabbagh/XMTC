
import numpy as np
import pandas as pd
import glob
import os

current_path = os.getcwd()
path = os.path.join(current_path, 'Data')
os.makedirs(path, exist_ok=True)
print(f'Path {path} created successfully.')

# Loop through Task1, Task2, Task3
for i in range(1, 4):
    csv_files = glob.glob(f'reaching_features/Task{i}_*.csv')

    if csv_files:  # Check if there are files
        # 1. Merging CSV files
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        print(f'Number of Task{i}_R2G_features files: {len(csv_files)}')
        # 2. Normalizing time to the range of [0,1]
        gb = df.groupby(['userID', 'object', 'side', 'action', 'trialID'])
        groups_org = dict(list(gb))  # dictionary that keys are groups_keys and value is the dataframe of that group
        groups = groups_org.copy()
        df_merge = pd.DataFrame()

        for key, value in groups.items():
            value.reset_index(drop=True, inplace=True)

        for key, value in groups_org.items():
            m = value['frameID'].max()
            normalized_time_data = [t / m for t in value['frameID']]
            groups[key].insert(7, 'frameID_normalized', normalized_time_data)

        df_merge = pd.concat([value for key, value in groups.items()])
        # 3. add class labels
        # add class label: object,side
        df_merge['label_obj_side_categorical'] = df_merge['object'] + '_' + df_merge['side']
        df_merge['label_obj_side_int'] = df_merge['label_obj_side_categorical'].astype('category').cat.codes

        # add class label: object,side,task
        df_merge['label_obj_side_task_categorical'] = df_merge['object'] + '_' + df_merge['side'] + '_' + df_merge['task']
        df_merge['label_obj_side_task_int'] = df_merge['label_obj_side_task_categorical'].astype('category').cat.codes

        df_merge.to_csv(os.path.join(path, f'R2G_features_t_normalized_Task{i}.csv'), index=False)
        print(f'R2G_features_t_normalized_Task{i}.csv is created in {path}.')
    else:
        print(f'No files found for Task{i}_R2G_features.')

print('finished.')
