import glob
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
import matplotlib.pylab as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(curr_dir, "pca_model")

with open(pickle_path, "rb") as f:
    pca_loaded = pickle.load(f)

print(pca_loaded.components_)
plt.plot(pca_loaded.components_.T)

plt.show()

assert False

data = []
for file in glob.iglob("../images_data/*.npy"):
    img = np.load(file)[:,:,9:]
    H,W,C = img.shape;
    array = img.reshape((-1, C))
    data.append(array)

stack = np.vstack(data)[::100]
pca = PCA(n_components=3)
pca.fit(stack)

with open(pickle_path, "wb") as f:
    pickle.dump(pca, f)
