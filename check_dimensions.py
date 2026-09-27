import numpy as np

# Load and check dimensions
test_data = np.load('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/test.npy')
train_data = np.load('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/train.npy')
labels = np.load('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/labels.npy')

print("Data dimensions:")
print(f"Test data shape: {test_data.shape}")
print(f"Train data shape: {train_data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Labels dtype: {labels.dtype}")

print("\nLabel statistics:")
print(f"Unique values in labels: {np.unique(labels)}")
print(f"Number of anomalies (1): {np.sum(labels == 1)}")
print(f"Number of normal (0): {np.sum(labels == 0)}")

# Check if labels need to be reshaped
if len(labels.shape) == 1:
    print(f"\nLabels are 1D with length: {len(labels)}")
    if len(labels) == test_data.shape[0]:
        print("Labels length matches test data length - this is correct")
    else:
        print("WARNING: Labels length does not match test data length")
else:
    print(f"\nLabels are {len(labels.shape)}D")