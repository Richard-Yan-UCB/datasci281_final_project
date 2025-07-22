from scipy import signal
import numpy as np
import cv2 as cv
from skimage.transform import rescale
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2 as cv
from skimage.feature import hog, blob_log, blob_dog, blob_doh, canny
from skimage import data, exposure
from skimage.filters import difference_of_gaussians
from skimage.filters import laplace
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, rotate
from skimage.color import rgb2gray
from sklearn.metrics import mean_squared_error
from scipy import stats
from skimage.feature import hessian_matrix, hessian_matrix_det
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
     
""" create a 2-D gaussian blurr filter for a given mean and std """
def create_2d_gaussian(size=9, std=1.5):
    gaussian_1d = signal.windows.gaussian(size,std=std)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d/(gaussian_2d.sum())
    return gaussian_2d


""" normalize teh image between 0 and 1 """
def normalize_img(img):
    normalized = (img - img.min())/(img.max() - img.min())   
    return normalized

""" 
convert to grayscale
normalize teh image between 0 and 1
resize image to im_size """

def preprocess_image(img_path, im_size=512):
    img = cv.imread(img_path)
    img_grayscale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    normalized = normalize_img(img_grayscale)
    rescaled = rescale_img(img=normalized, standard=im_size)
    return rescaled

def rescale_img(img, standard=256):
  # rescale short side to standard size, then crop center
  scale = standard / min(img.shape[:2])
  img = rescale(img, scale, anti_aliasing=True)
  img = img[int(img.shape[0]/2 - standard/2) : int(img.shape[0]/2 + standard/2),
            int(img.shape[1]/2 - standard/2) : int(img.shape[1]/2 + standard/2)]
  return img

def summarize_blobs(blobs, image_shape):
    if len(blobs) == 0:
        return np.zeros(16)
    
    y, x, r = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    h, w = image_shape

    # Normalize coordinates to [0, 1]
    y_norm = y / h
    x_norm = x / w
    r_norm = r / max(h, w)

    features = [
        len(blobs),                        # total number of blobs
        len(blobs) / (h * w),              # blob density
        np.mean(y_norm), np.std(y_norm), np.min(y_norm), np.max(y_norm),
        np.mean(x_norm), np.std(x_norm), np.min(x_norm), np.max(x_norm),
        np.mean(r_norm), np.std(r_norm), np.min(r_norm), np.max(r_norm),
        np.mean(r_norm) / np.mean([np.std(y_norm), np.std(x_norm)])  # compactness
    ]


    return np.array(features)


def get_features(in_imgs, feat_name='canny'):
    features = []
    if feat_name == 'canny':
        for i in range(in_imgs.shape[0]):
            # Convert to uint8 and apply Canny
            print("Canny Image: " + str(i))
            image = in_imgs[i]
            img_uint8 = (image * 255).astype(np.uint8)
            edges = cv.Canny(img_uint8, threshold1=50, threshold2=150)
            edges_norm = edges/255.0
            edges_norm_flatten = edges_norm.flatten()
            features.append(edges_norm_flatten)
        features = np.array(features)

        return features

    if feat_name == 'blob_dog':
        # stack extracted hog features into array
        # also save the first hog image for plotting
        max_features = 0
        for i in range(in_imgs.shape[0]):
            print("Blob DoG Image:" + str(i))
            image = in_imgs[i]
            mean_pixel_intensity = np.mean(image)
            brightness_adjusted_img = image.copy()

            # lower brights
            if (mean_pixel_intensity > 0.3):
                bright_mask = image > 0.6
                brightness_adjusted_img[bright_mask] = brightness_adjusted_img[bright_mask] * 0.2
            # brighten darks
            elif (mean_pixel_intensity < 0.15):
                dark_mask = image < 0.2
                brightness_adjusted_img[dark_mask] = brightness_adjusted_img[dark_mask] * 2.0

            # apply vignette on images so that edges are less emphasized for DoG computation
            h, w = image.shape
            center_x, center_y = w // 2, h // 2
            # Step 2: Create radial mask centered in image
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            mask = 1 - (dist / max_dist)
            mask = np.clip(mask, 0, 1)
            if (mean_pixel_intensity > 0.3):
                mask = mask**1.6  # steeper falloff
            elif (mean_pixel_intensity < 0.15):
                mask = mask**1.5
            # Step 3: Apply mask
            img_masked = brightness_adjusted_img * mask

            # Apply DoG
            large_low_sigma, large_high_sigma = 30, 40
            small_low_sigma, small_high_sigma = 20, 30
            # dog_image_large_blobs = difference_of_gaussians(img_masked, large_low_sigma, large_high_sigma)
            # dog_image_small_blobs = difference_of_gaussians(img_masked, small_low_sigma, small_high_sigma)
            # blobs_dog_small = blob_dog(img_masked, min_sigma=small_low_sigma, max_sigma=small_high_sigma, threshold=0.078)
            # blobs_dog_small[:, 2] = blobs_dog_small[:, 2] * np.sqrt(2)
            blobs_dog_large = blob_dog(img_masked, min_sigma=large_low_sigma, max_sigma=large_high_sigma, threshold=0.056)
            blobs_dog_large[:, 2] = blobs_dog_large[:, 2] * np.sqrt(2)


            # # Rescale for better display
            # dog_image_large_blobs_rescaled = exposure.rescale_intensity(dog_image_large_blobs, in_range=(0, 0.3))
            # dog_image_small_blobs_rescaled = exposure.rescale_intensity(dog_image_small_blobs, in_range=(0, 0.3))


            blob_dog_final = blobs_dog_large.flatten()
            if blob_dog_final.shape[0] > max_features:
                max_features = blob_dog_final.shape[0]
            # summary_feature = summarize_blobs(blobs_dog_large, image.shape)
            features.append(blob_dog_final)

        for blob_index in range(len(features)):
            print("Before: " + str(features[blob_index].shape))
            if features[blob_index].shape[0] < max_features:
                features[blob_index] = np.pad(features[blob_index], pad_width=(0,max_features-features[blob_index].shape[0]))
            print("After: " + str(features[blob_index].shape))
        features = np.array(features)
        return features
    
    if feat_name == 'blob_doh':
        max_features = 0
        for i in range(in_imgs.shape[0]):
            print("DoH Image: " + str(i))
            image = in_imgs[i]
            mean_pixel_intensity = np.mean(image)
            brightness_adjusted_img = image.copy()
            # lower brights
            if (mean_pixel_intensity > 0.3):
                bright_mask = image > 0.6
                brightness_adjusted_img[bright_mask] = brightness_adjusted_img[bright_mask] * 0.2
            # brighten darks
            elif (mean_pixel_intensity < 0.15):
                dark_mask = image < 0.2
                brightness_adjusted_img[dark_mask] = brightness_adjusted_img[dark_mask] * 2.0

            # apply vignette on images so that edges are less emphasized for DoG computation
            h, w = image.shape
            center_x, center_y = w // 2, h // 2
            # Step 2: Create radial mask centered in image
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            mask = 1 - (dist / max_dist)
            mask = np.clip(mask, 0, 1)
            if (mean_pixel_intensity > 0.3):
                mask = mask**1.6  # steeper falloff
            elif (mean_pixel_intensity < 0.15):
                mask = mask**1.5
            # Step 3: Apply mask
            img_masked = brightness_adjusted_img * mask

            # Apply DoH
            large_low_sigma, large_high_sigma = 25, 45
            # small_low_sigma, small_high_sigma = 10, 30
            # doh_image_large_blobs = hessian_matrix_det(img_masked, large_low_sigma)
            # doh_image_small_blobs = hessian_matrix_det(img_masked, small_low_sigma)
            # blobs_doh_small = blob_doh(img_masked, min_sigma=small_low_sigma, max_sigma=small_high_sigma, threshold=0.002)
            # blobs_doh_small[:, 2] = blobs_doh_small[:, 2] * np.sqrt(2)
            blobs_doh_large = blob_doh(img_masked, min_sigma=large_low_sigma, max_sigma=large_high_sigma, threshold=0.0015)
            # blobs_doh_large[:, 2] = blobs_doh_large[:, 2] * np.sqrt(2)
            # # Rescale for better display
            # doh_image_large_blobs_rescaled = exposure.rescale_intensity(doh_image_large_blobs, in_range=(0, 0.0001))
            # doh_image_small_blobs_rescaled = exposure.rescale_intensity(doh_image_small_blobs, in_range=(0, 0.0001))

            blobs_doh_final = blobs_doh_large.flatten()
            if blobs_doh_final.shape[0] > max_features:
                max_features = blobs_doh_final.shape[0]
            summary_feature = summarize_blobs(blobs_doh_large, image.shape)
            features.append(blobs_doh_final)
        
        for blob_index in range(len(features)):
            print("Before: " + str(features[blob_index].shape))
            if features[blob_index].shape[0] < max_features:
                features[blob_index] = np.pad(features[blob_index], pad_width=(0,max_features-features[blob_index].shape[0]))
            print("After: " + str(features[blob_index].shape))

        features = np.array(features)
        return features
    
    # if feat_name == "complex":
        

    return None


def get_PCA(X_list, n_components=[15,15,100]):
    pca_list = []
    xpca_list = []
    for index, X in enumerate(X_list):
        pca = PCA(n_components=n_components[index], svd_solver="randomized", whiten=True).fit(X)
        X_pca = pca.transform(X)
        pca_list.append(pca)
        xpca_list.append(X_pca)
    return pca_list, xpca_list

def plot_PCA(X_list, n_components=[15,15,100]):
    pca_list, xpca_list = get_PCA(X_list, n_components=n_components)

    plt.figure(figsize=(15,5))
    colors = ['r-', 'b-','g-']
    labels = ['dog features', 'doh features','canny_features']
    for i in range(len(X_list)):
        plt.plot(np.cumsum(pca_list[i].explained_variance_ratio_), colors[i], label=labels[i])
        plt.xticks(np.linspace(0, n_components[i]+1, 50))
        plt.yticks(np.linspace(0, 1.2, 8))
        plt.grid(True)
        plt.xlabel('Number of components')
        plt.ylabel('Explained Variances')
        plt.legend()
    
    plt.show()

def get_tsne(X_list, n_components=2):
  xtsne_list = []
  for X in X_list:
    tsne = TSNE(n_components=n_components, random_state=0)
    X_tsne = tsne.fit_transform(X)
    xtsne_list.append(X_tsne)
  return xtsne_list