import numpy as np

label = np.load('dataset/SMD/labels.npy')
print(label.shape)   # (124499,)
# find the index of the first 1
print(np.where(label == 1)[0][1])  # 1000
print(label[13813:13823])
