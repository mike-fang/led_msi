import numpy as np
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
import os.path
from tqdm import tqdm
from time import time
import cv2
import matplotlib.pylab as plt 
from camera_calibration import *
from scipy.interpolate import interp2d
from PIL import Image
from skimage.color import lab2rgb
from spec_recon import get_recon
from glob import glob
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def normalize_img(img, q=0, axis=None):
    m = np.quantile(img, q, axis=axis)
    M = np.quantile(img, 1-q, axis=axis)
    norm_img = (img - m) / (M - m)
    norm_img[norm_img > 1] = 1
    norm_img[norm_img < 0] = 0
    return norm_img
def unnormed_lab2rgb(ulab, q=0):
    lab = normalize_img(ulab, q=q, axis=(0, 1))

    AB_range = 100

    lab *= np.array([[ [100, 2*AB_range, 2*AB_range] ]])
    lab -= np.array([[ [0, AB_range, AB_range] ]])

    return lab2rgb(lab)

class HyperspectralImage:
    def __init__(self, subarr_size=(5, 5), raw_img=None, trim_img=True, corrected=False, interpolate=True):
        self.trim_img=trim_img
        self.subarr_size = subarr_size
        self.corrected = corrected
        self.interpolate = interpolate
        if raw_img is not None:
            self.raw_img = raw_img
    @property
    def raw_img(self):
        return self.raw_img__
    @raw_img.setter
    def raw_img(self, val):
        if self.trim_img:
            Y0, X0 = filter_offset
            HF, WF = filter_size
            Y1 = Y0 + HF
            X1 = X0 + WF
            val = val[Y0:Y1, X0:X1]
        self.raw_img__ = val
        self.img_cube = self.get_image_cube(val)
        self.flat_img = self.img_cube.reshape(-1, self.n_channels) 
        self.cube_shape = self.img_cube.shape
        self.raw_shape = val.shape
    @property
    def n_channels(self):
        return self.subarr_size[0] * self.subarr_size[1]
    def mean_lum(self):
        return self.img_cube.mean(axis=2)
    def centered_cube(self):
        return self.img_cube - self.mean_lum()[:, :, None]
    def get_image_cube(self, arr=None, corrected=True):
        if arr is None:
            arr = self.arr
        H, W = arr.shape
        sa_H, sa_W = self.subarr_size
        if not self.interpolate:
            ds_H = np.int(H / sa_H)
            ds_W = np.int(W / sa_W)
            img_cube = np.zeros((ds_H, ds_W, self.n_channels))
            for i in range(sa_H):
                for j in range(sa_W):
                    n = i * sa_W + j
                    img_cube[:, :, n] = self.raw_img[i::sa_H, j::sa_W]
        else:
            img_cube = np.zeros((H, W, self.n_channels))
            for i in range(sa_H):
                Y = (np.arange(i, H, 5))
                for j in tqdm(range(sa_W)):
                    n = i * sa_W + j
                    X = (np.arange(j, W, 5))
                    Z = self.raw_img[i::sa_H, j::sa_W]
                    f = interp2d(X, Y, Z)
                    img_cube[:, :, n] = f(np.arange(W), np.arange(H))

        if self.corrected:
            img_cube = img_cube @ correction_matrix
        return img_cube
    def get_projection(self, bases, centered=True, normalized=True):
        if isinstance(bases, str):
            if bases == 'pca':
                bases = self.get_pca()
        if bases.shape[0] != self.n_channels:
            if bases.shape[1] == self.n_channels:
                bases = bases.T
            else:
                raise Exception('The bases have incorrect dimensions')
        if normalized:
            norms = np.linalg.norm(bases, axis=0)
            bases /= norms

        img_cube = self.centered_cube() if centered else self.img_cube
        return img_cube @ bases
    def get_pca(self, return_pca=False, pca_args={}):
        pca = PCA(n_components=self.n_channels, **pca_args)
        pca_img = pca.fit_transform(self.flat_img)
        bases = pca.components_.T
        if return_pca:
            pca_img = pca_img.reshape(self.cube_shape)
            flip = (bases.mean(axis=0) > 0) * 2 - 1
            pca_img = pca_img * flip[None, None, :]
            return bases, pca_img, pca
        return bases
    def save_img(self, path=None):
        if path is None:
            path = f'saved_imgs/{int(time())}.npy'
        np.save(path, self.raw_img)
        print(f'Raw array saved to {path}')
    def pca_grid_img(self, whiten=True):
        bases, pca_img, _ = self.get_pca(return_pca=True, pca_args={'whiten':whiten})
        img_grid = []
        flip = (bases.mean(axis=0) > 0) * 2 - 1
        pca_img = pca_img * flip[None, None, :]
        for i in range(3):
            img_row = []
            for j in range(3):
                n = i * 3 + j
                if n == 0:
                    img = pca_img[:, :, 0][:, :, None] * np.ones(3)[None, None, :]
                else:
                    idx = slice(n*3 - 2, (n+1)*3 - 2)
                    img = pca_img[:, :, idx]
                img_row.append(img)
            img_row = np.concatenate(img_row, axis=1)
            img_grid.append(img_row)
        img_grid = np.concatenate(img_grid, axis=0)
        return img_grid
    def rgb_projection(self):
        rgb_bases = np.load('./rgb_bases.npy')
        return self.get_projection(rgb_bases)
    def false_color_lab(self, int_vals=False):
        bases, pca_img, _ = self.get_pca(return_pca=True, pca_args={'whiten':False})
        pca_img = pca_img[:, :, :3]
        lab_img = unnormed_lab2rgb(pca_img, q=.05)
        return lab_img
    def recon_spec(self, i, j, how='naive', return_wl=True):
        band_intensities = self.img_cube[i, j]
        if how == 'naive':
            psi = band_response.T * filter_response[:, None]
            recon = psi @ np.linalg.inv(psi.T @ psi) @ band_intensities
        elif how == 'lasso':
            recon =  get_recon(band_intensities, alpha=5e-4)
        elif how == 'raw':
            recon = band_intensities
            #filter_wavelengths = np.arange(len(recon))

        if return_wl:
            return filter_wavelengths, recon
        else:
            return recon

if __name__ == '__main__':
    for f in glob('./saved_imgs/march_11_2020/*.npy')[:1]:
        raw_img = np.load(f)
    himg = HyperspectralImage(raw_img=raw_img, corrected=False, interpolate=False, trim_img=False)
    cube = himg.img_cube.astype(np.uint8)
    np.save('./ex_hs_img.npy', cube)
    img = himg.false_color_lab()
    plt.title(img.shape)
    plt.imshow(img)

    R = img[:, :, 0].flatten()
    G = img[:, :, 1].flatten()
    B = img[:, :, 2].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(R, G, B, marker='o', s=2)
    plt.show()
