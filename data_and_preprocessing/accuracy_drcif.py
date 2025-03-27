import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory pattern and the range of run numbers
base_dir = "runs_Task3_drcif/drcif_slidingWindow_currentrange_{}/Results_slidingWindow_DrCIF/test_acc_Task3_{}.npy"     # specify Task number
runnumbers = range(10, 1388, 10)  # from 0 to 1300 in increments of 10

# Initialize lists to store the results
run_numbers_list = []
test_acc_list = []
max_len = 1378
# Loop over each run number, load the .npy file, and store the data_and_preprocessing
for run in runnumbers:
    run = min(run, max_len)
    print('**************')
    print(run)
    file_path = base_dir.format(run, run)
    print(file_path)
    if os.path.exists(file_path):  # Check if the file exists
        test_acc = np.load(file_path)
        print(test_acc)
        run_numbers_list.append(run)
        test_acc_list.append(test_acc)
    else:
        print(f'file path:{file_path} doesnt exist')

# Convert to numpy arrays for easier plotting
run_numbers_array = np.array(run_numbers_list)
test_acc_array = np.array(test_acc_list)
np.save('run_numbers_array.npy', run_numbers_array)
np.save('test_acc_array.npy', test_acc_array)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(run_numbers_array, test_acc_array, marker='o', linestyle='-')
plt.xlabel("Run Number")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy over Run Number for DrCIF")
plt.grid(True)
plt.savefig('test_accuracy', dpi=300)
# plt.show()

print('finish')