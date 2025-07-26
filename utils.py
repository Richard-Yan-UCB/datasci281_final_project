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
from sklearn.metrics import mean_squared_error, roc_curve, auc
from scipy import stats
from skimage.feature import hessian_matrix, hessian_matrix_det
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize


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
        return np.zeros(14)
    
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
        np.mean(r_norm), np.std(r_norm), np.min(r_norm), np.max(r_norm)
    ]


    return np.array(features)


def get_features(in_imgs, feat_name='canny'):
    features = []
    if feat_name == 'canny':
        for i in tqdm(range(in_imgs.shape[0]), desc = 'Canny Edge Images'):
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
        for i in tqdm(range(in_imgs.shape[0]), desc = 'Blob Dog Images'):
            #print("Blob DoG Image:" + str(i))
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
            # small_low_sigma, small_high_sigma = 20, 30
            # dog_image_large_blobs = difference_of_gaussians(img_masked, large_low_sigma, large_high_sigma)
            # dog_image_small_blobs = difference_of_gaussians(img_masked, small_low_sigma, small_high_sigma)
            # blobs_dog_small = blob_dog(img_masked, min_sigma=small_low_sigma, max_sigma=small_high_sigma, threshold=0.078)
            # blobs_dog_small[:, 2] = blobs_dog_small[:, 2] * np.sqrt(2)
            blobs_dog_large = blob_dog(img_masked, min_sigma=large_low_sigma, max_sigma=large_high_sigma, threshold=0.056)
            blobs_dog_large[:, 2] = blobs_dog_large[:, 2] * np.sqrt(2)


            # # Rescale for better display
            # dog_image_large_blobs_rescaled = exposure.rescale_intensity(dog_image_large_blobs, in_range=(0, 0.3))
            # dog_image_small_blobs_rescaled = exposure.rescale_intensity(dog_image_small_blobs, in_range=(0, 0.3))


            # blob_dog_final = blobs_dog_large.flatten()
            # if blob_dog_final.shape[0] > max_features:
            #     max_features = blob_dog_final.shape[0]
            summary_feature = summarize_blobs(blobs_dog_large, image.shape)
            features.append(summary_feature)
            
        # pbar = tqdm(range(len(features)), desc="Padding Features")
        # for blob_index in pbar:
        #     if features[blob_index].shape[0] < max_features:
        #         features[blob_index] = np.pad(features[blob_index], pad_width=(0,max_features-features[blob_index].shape[0]))
        features = np.array(features)
        return features
    
    if feat_name == 'blob_doh':
        max_features = 0
        for i in tqdm(range(in_imgs.shape[0]), desc = 'Blob DoH images'):
            #print("DoH Image: " + str(i))
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
            blobs_doh_large[:, 2] = blobs_doh_large[:, 2] * np.sqrt(2)
            # # Rescale for better display
            # doh_image_large_blobs_rescaled = exposure.rescale_intensity(doh_image_large_blobs, in_range=(0, 0.0001))
            # doh_image_small_blobs_rescaled = exposure.rescale_intensity(doh_image_small_blobs, in_range=(0, 0.0001))

            # blobs_doh_final = blobs_doh_large.flatten()
            # if blobs_doh_final.shape[0] > max_features:
            #     max_features = blobs_doh_final.shape[0]
            summary_feature = summarize_blobs(blobs_doh_large, image.shape)
            features.append(summary_feature)
        
        # pbar = tqdm(range(len(features)), desc="Padding Features")
        # for blob_index in pbar:
        #     if features[blob_index].shape[0] < max_features:
        #         features[blob_index] = np.pad(features[blob_index], pad_width=(0,max_features-features[blob_index].shape[0]))

        features = np.array(features)
        return features
    
    if feat_name == "complex":
        if in_imgs.ndim == 3:  #(N, H, W)
            in_imgs = np.expand_dims(in_imgs, axis=-1)  # (N, H, W, 1)
            #Convert grayscale (single-channel) to RGB (3 channels)
        if in_imgs.shape[-1] == 1:
            in_imgs = np.repeat(in_imgs, 3, axis=-1)
        model_name = "facebook/dinov2-base"
        fp16 = True 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype = torch.float16 if fp16 else None
            ).eval().to(device)
        model_dtype = next(model.parameters()).dtype
        hidden_size = model.config.hidden_size
        
        N = len(in_imgs)
        features = torch.empty(N, hidden_size, dtype=torch.float32)
        
        step = 64 # btach_size
        
        with torch.no_grad(), torch.autocast('cuda', enabled=fp16):
            for start in tqdm(range(0, N, step), desc = 'Complex Features'):
                end = min(start + step, N)
                batch = in_imgs[start:end]
                inputs = processor(batch, return_tensors="pt")
                inputs = {k: v.to(device, dtype=model_dtype) for k, v in inputs.items()}
                feats = model(**inputs).last_hidden_state[:, 0] #before the last layer
                features[start:end] = feats.float().cpu()
                
                #print(f"\rExtracted: {end}/{N}", end="", flush=True)
                
            print(f"\nFeature tensor shape: {tuple(features.shape)}")
        
        return features

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
    colors = ['r-', 'b-','g-','p-']
    labels = ['dog features', 'doh features','canny_features','complex_features']
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


def train_model(X_train, y_train, classes, model_type='logistic', feature='canny'):

    if model_type =='logistic':
        model = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000)
        start_time = time.perf_counter()
        model.fit(X_train,y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.predict(X_train)
        y_model_pred_proba = model.predict_proba(X_train)

    elif model_type =='svm':
        svm_param_grid = {
            'C': [0.1],
            'gamma': [0.001,0.1,1],
            'kernel': ['rbf']
        }
        svc = svm.SVC(probability=True)
        model = GridSearchCV(svc, svm_param_grid, scoring='accuracy', cv=5, verbose=2)
        start_time = time.perf_counter()
        model.fit(X_train,y_train)
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

    elif model_type =='gbm':
        gbm_param_grid = {'loss':['log_loss'],
                'learning_rate':[0.1],
                'n_estimators':[80,100,120],
                'max_depth':[2,3,4]}
        
        gbm = GradientBoostingClassifier()
        model = GridSearchCV(gbm, gbm_param_grid, scoring='accuracy',cv=5, verbose=1)
        start_time = time.perf_counter()
        model.fit(X_train,y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

        print(str(model.best_params_))
        print(str(model.score(X_train,y_train)))
        
    elif model_type =='xgboost':
        params = {'learning_rate':[0.09, 0.1,0.11],
                'n_estimators':[80,90,100,110],
                'max_depth':[2,3,4]}

        xgb_model = XGBClassifier()

        # Fit model
        # Use GridSearch to find the best parameters
        model = GridSearchCV(xgb_model, params, cv=5, scoring='accuracy',verbose=1)
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

    # elif model_type == 'lda':


    # Generate Confusion Matrix for Logistic Regression
    confusion_matrix = metrics.confusion_matrix(y_train, y_model_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)
    cm_display.plot()
    plt.title('CM Feature:' + str(feature) + ' Model: ' + str(model_type))
    plt.show()

    # # Handle classifier output format
    # if isinstance(y_model_pred_proba, list):
    #     # Convert list of class-wise predictions to proper array
    y_train_binarized = label_binarize(y_train, classes=[0,1,2,3])

    if isinstance(y_model_pred_proba, list):
        y_model_pred_proba = np.stack([score[:, 1] for score in y_model_pred_proba], axis=1)
    print(y_model_pred_proba.shape)

    # Generate ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_train_binarized[:, i], y_model_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red','brown']
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Feature:' + str(feature) + ' Model: ' + str(model_type))
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    accuracy_score = metrics.accuracy_score(y_train, y_model_pred)
    macro_precision = metrics.precision_score(y_train, y_model_pred,average ='macro')
    macro_recall = metrics.recall_score(y_train, y_model_pred,average='macro')
    macro_f1 = metrics.f1_score(y_train, y_model_pred,average='macro')
    micro_precision = metrics.precision_score(y_train, y_model_pred,average='micro')
    micro_recall = metrics.recall_score(y_train, y_model_pred,average='micro')
    micro_f1 = metrics.f1_score(y_train, y_model_pred,average='micro')
    # fpr, tpr, thresholds = metrics.roc_curve(y_train, y_logistic_pred_dog)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    #                                   name='dog estimator')

    # print("==================" + str(feature) + " TRAINING METRICS ===================")
    # print("Accuracy Score: " + str(accuracy_score))
    # print("Macro Precision: " + str(macro_precision))
    # print("Macro Recall: " + str(macro_recall))
    # print("Macro F1: " + str(macro_f1))
    # print("Micro Precision: " + str(micro_precision))
    # print("Micro Recall: " + str(micro_recall))
    # print("Micro F1: " + str(micro_f1))

    results_dict = {}
    results_dict['feature'] = feature
    results_dict['model_type'] = model_type
    results_dict['accuracy_score'] = accuracy_score
    results_dict['macro_precision'] = macro_precision
    results_dict['macro_recall'] = macro_recall
    results_dict['macro_f1'] = macro_f1
    results_dict['micro_precision'] = micro_precision
    results_dict['micro_recall'] = micro_recall
    results_dict['micro_f1'] = micro_f1
    results_dict['training_time'] = elapsed_time


    if model_type == 'logistic':
        return model, results_dict
    elif model_type == 'svm' or model_type == 'gbm':
        return model.best_estimator_, results_dict