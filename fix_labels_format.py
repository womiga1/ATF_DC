import numpy as np

# Load current data
test_data = np.load('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/test.npy')
current_labels = np.load('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/labels.npy')

print(f"Test data shape: {test_data.shape}")
print(f"Current labels shape: {current_labels.shape}")

# The labels should have the same shape as test data
# Each time step and each feature should have a label
# For anomaly detection, typically all features at a time step have the same label

# Create 2D labels with shape (time_steps, features)
time_steps, features = test_data.shape
labels_2d = np.zeros((time_steps, features), dtype=int)

# Broadcast the 1D labels to all features
for i in range(features):
    labels_2d[:, i] = current_labels

print(f"New labels shape: {labels_2d.shape}")
print(f"Labels dtype: {labels_2d.dtype}")

# Verify the content
print(f"Number of anomaly time steps: {np.sum(current_labels == 1)}")
print(f"Number of anomaly labels in 2D: {np.sum(labels_2d == 1)}")
print(f"Expected anomaly labels in 2D: {np.sum(current_labels == 1) * features}")

# Save the corrected labels
output_path = '/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/labels.npy'
np.save(output_path, labels_2d)
print(f"Fixed labels saved to: {output_path}")

# Verify by loading again
verification_labels = np.load(output_path)
print(f"Verification - loaded labels shape: {verification_labels.shape}")
print(f"Verification - unique values: {np.unique(verification_labels)}")