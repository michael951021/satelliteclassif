#!/usr/bin/env python
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import colorsys

from fontTools.misc.cython import returns
from sklearn import neighbors
from sklearn import svm
import os
import time

np.random.seed(7)
if not os.path.exists('samples'):
    os.makedirs('samples')
"""
TODO
Section 1: Data Loading
1. Load in the .mat file within the same directory of this python file.
2. Pull out the raw training/test data set components from .mat data.
3. Load in the annotations(label_dict) from .mat data.
"""
data_map = sio.loadmat("sat4")

mat = data_map.get("train_x")
train_x_raw = data_map.get("train_x")
trainfire_y = data_map.get("train_y")
test_x_raw = data_map.get("test_x")
testfire_y = data_map.get("test_y")
label_dict = data_map.get("annotations")


def stringlabel(encoded, labeldic):
    concatenated = np.array(["".join(encoded.astype(str))])
    for row in labeldic:
        if np.array_equal(concatenated, row[0]):
            return row[1][0]


def labeltoval(label):
    if label == "none":
        return 4
    elif label == "grassland":
        return 3
    elif label == "trees":
        return 2
    elif label == "barren land":
        return 1
    return 0


"""
Section 2: Data Processing
Process data to be inputted into models. Also, created a smaller
versions of datasets to debug model without significant overhead.
"""
# UNCOMMENT TO USE SHORTENED DATASETS FOR DEBUGGING
train_x_raw = train_x_raw[:, :, :, ::20]
test_x_raw = test_x_raw[:, :, :, ::20]
trainfire_y = trainfire_y[:, ::20]
testfire_y = testfire_y[:, ::20]

train_x = train_x_raw.astype(np.float64) / 255.
test_x = test_x_raw.astype(np.float64) / 255.
train_y = trainfire_y[0, :] + 2 * trainfire_y[1, :] + 3 * trainfire_y[2, :] + 4 * trainfire_y[3, :]
test_y = testfire_y[0, :] + 2 * testfire_y[1, :] + 3 * testfire_y[2, :] + 4 * testfire_y[3, :]
train_x_reshape = np.transpose(np.reshape(train_x, (-1, train_x.shape[-1])))
test_x_reshape = np.transpose(np.reshape(test_x, (-1, test_x.shape[-1])))
"""
TODO:
Section 3: Data Overview
1. Select 4 random points from training set and output image file
alongside its label (annotation)
2. Count number frequency of classes across the dataset
"""
# Create length 4 arrow of random training set points
indices = np.random.rand(4) * train_x_raw.shape[-1]
indices = indices.astype(int)
for i, indx in enumerate(indices):
    raw_img = train_x_raw[:, :, :, indx]
    rgb_data = raw_img[:, :, 0:3]
    nir_data = raw_img[:, :, 3]
    # Converting data into image
    sample = Image.fromarray(rgb_data.astype(np.uint8))
    sample_nir = Image.fromarray(nir_data.astype(np.uint8))
    sample.save('samples/sample_' + str(i + 1) + '.png')
    sample_nir.save('samples/sample_' + str(i + 1) + '_nir.png')

    image_enc = trainfire_y[:, indx]

    label = labeltoval(stringlabel(image_enc, label_dict)) - 1
    print("Image #" + str(indx))
    print('sample ' + str(i + 1) + ' is of class ' + label_dict[label, 1][0])
class_count = np.zeros((4))
for i in range(train_x_raw.shape[-1]):
    image_enc = trainfire_y[:, i]
    for j in range(4):
        class_count[j] += image_enc[j]
# Printing the frequecy outputs
for i in range(4):
    print('class ' + label_dict[i, 1][0] + ' has %i samples' % class_count[i])
"""
TODO:
Section 4: How does K affect accuracy?
"""
# Running knn on different neighbor sizes
start = time.time()
neighbours_range = np.arange(2, 61, 2)
train_acc_knn = np.zeros(neighbours_range.shape)
test_acc_knn = np.zeros(neighbours_range.shape)
for i, number in enumerate(neighbours_range):
    neigh = neighbors.KNeighborsClassifier(n_neighbors=number)
    neigh.fit(train_x_reshape, train_y)
    for num, val in enumerate(train_x_reshape):
        reshaped_row = np.reshape(val, (1, -1))
        predicted_label = neigh.predict(reshaped_row)
        expected_label = train_y[num]
        if predicted_label == expected_label:
            train_acc_knn[i] += 1
    for num, val in enumerate(test_x_reshape):
        reshaped_row = np.reshape(val, (1, -1))
        predicted_label = neigh.predict(reshaped_row)
        expected_label = test_y[num]
        if predicted_label == expected_label:
            test_acc_knn[i] += 1
train_acc_knn = train_acc_knn / (train_x_reshape.shape[0])
test_acc_knn = test_acc_knn / (test_x_reshape.shape[0])

print('k-NN on pixel values: done in %.2f secs' % (time.time() - start))
plt.figure(figsize=(12, 12))
plt.plot(neighbours_range, train_acc_knn, label='Training Accuracy')
plt.plot(neighbours_range, test_acc_knn, label='Test Accuracy')
plt.xlabel('Num. Neighbors', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('k-NN on pixel values', fontsize=16)
plt.grid()
plt.legend(fontsize=16)
plt.savefig('figure1.pdf')
plt.close()
# TODO: Find k for best training and test accuracy

# Training data pair of max accuracy and its associated # of neighbors
train_max = np.zeros(2)
for i, num in enumerate(train_acc_knn):
    if num > train_max[0]:
        train_max[0] = num
        train_max[1] = i
train_max[1] = neighbours_range[int(train_max[1])]

# Test data pair of max accuracy and its associated # of neighbors
test_max = np.zeros(2)
for i, num in enumerate(test_acc_knn):
    if num > test_max[0]:
        test_max[0] = num
        test_max[1] = i
test_max[1] = neighbours_range[int(test_max[1])]

best_k_train = train_max[1]
best_k_test = test_max[1]
print('k-NN on pixel values: neighbor size %i maximises training accuracy' %
      best_k_train)
print('k-NN on pixel values: neighbor size %i maximises test accuracy' %
      best_k_test)
# TODO: Load, fit and predict on testing data with best_k_test. knn_best_predict should be an
# array of predicted labels

knn_best_predict = None
# TODO: Finding wrongly classified samples for each class by comparing knn_best_predict and train_y
for i in range(4):
    pass
    # TODO: Find a correct label for label=i+1
    right_indx = None
    # TODO: Find a incorrect label for label=i+1
    wrong_indx = None
    # TODO: Load in data for right and wrong indices
    rgb_data_right = None
    nir_data_right = None
    rgb_data_wrong = None
    nir_data_wrong = None
    # Save data into images
    sample_right = Image.fromarray(rgb_data_right.astype(np.uint8))
    sample_right_nir = Image.fromarray(nir_data_right.astype(np.uint8))
    sample_wrong = Image.fromarray(rgb_data_wrong.astype(np.uint8))
    sample_wrong_nir = Image.fromarray(nir_data_wrong.astype(np.uint8))
    sample_right.save('samples/knn_class' + str(i) + '_right.png')
    sample_right_nir.save('samples/knn_class' + str(i) + '_right_nir.png')
    sample_wrong.save('samples/knn_class' + str(i) + '_wrong.png')
    sample_wrong_nir.save('samples/knn_class' + str(i) + '_wrong_nir.png')
"""
Section 5. SVM accuracy with Regularization
"""
# Running CVM for different regularizaion coefficients
C_list = np.arange(.5, 15.5, .5)
train_acc_svm = np.zeros(C_list.shape)
test_acc_svm = np.zeros(C_list.shape)
start = time.time()
for i, c in enumerate(C_list):
    # TODO: Train and fit SVM with gamma=c/10 and populate acc arrays
    pass
print('SVM on pixel values: done in %.2f secs' % (time.time() - start))
plt.figure(figsize=(12, 12))
plt.plot(C_list, train_acc_svm, label='Training Accuracy')
plt.plot(C_list, test_acc_svm, label='Test Accuracy')
plt.xlabel('C', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('SVM on pixel values', fontsize=16)
plt.grid()
plt.legend(fontsize=16)
plt.savefig('figure2.pdf')
plt.close()
"""
TODO: Section 6. Feature Reduction
"""
# Convert RGB to HSV
start = time.time()
train_x_hsv = np.zeros(train_x_raw.shape)
for i in np.arange(train_x_raw.shape[0]):
    for j in np.arange(train_x_raw.shape[1]):
        for k in np.arange(train_x_raw.shape[3]):
            train_x_hsv[i, j, :3, k] = colorsys.rgb_to_hsv(train_x_raw[i, j, 0, k] / 255.,
                                                           train_x_raw[i, j, 1, k] / 255.,
                                                           train_x_raw[i, j, 2, k] / 255.)
test_x_hsv = np.zeros(test_x_raw.shape)
for i in np.arange(test_x_raw.shape[0]):
    for j in np.arange(test_x_raw.shape[1]):
        for k in np.arange(test_x_raw.shape[3]):
            test_x_hsv[i, j, :3, k] = colorsys.rgb_to_hsv(test_x_raw[i, j, 0, k] / 255.,
                                                          test_x_raw[i, j, 1, k] / 255.,
                                                          test_x_raw[i, j, 2, k] / 255.)
# Retaining the NIR data
train_x_hsv[:, :, 3, :] = train_x[:, :, 3, :]
test_x_hsv[:, :, 3, :] = test_x[:, :, 3, :]
# For each HSV converted image, compute the mean and standard deviation and make new 8 feature dataset
train_x_hsv_mean = np.mean(train_x_hsv, axis=(0, 1))
train_x_hsv_std = np.std(train_x_hsv, axis=(0, 1))
train_x_hsv_feature = np.transpose(np.concatenate((train_x_hsv_mean, train_x_hsv_std), axis=0))
test_x_hsv_mean = np.mean(test_x_hsv, axis=(0, 1))
test_x_hsv_std = np.std(test_x_hsv, axis=(0, 1))
test_x_hsv_feature = np.transpose(np.concatenate((test_x_hsv_mean, test_x_hsv_std), axis=0))
# Normalize the new dataset with (x-min)/(max-min)
train_x_hsv_feature_max = np.amax(train_x_hsv_feature, axis=0)
train_x_hsv_feature_min = np.amin(train_x_hsv_feature, axis=0)
train_x_hsv_feature_norm = (train_x_hsv_feature - train_x_hsv_feature_min) / \
                           (train_x_hsv_feature_max - train_x_hsv_feature_min)
test_x_hsv_feature_max = np.amax(test_x_hsv_feature, axis=0)
test_x_hsv_feature_min = np.amin(test_x_hsv_feature, axis=0)
test_x_hsv_feature_norm = (test_x_hsv_feature - test_x_hsv_feature_min) / \
                          (test_x_hsv_feature_max - test_x_hsv_feature_min)
print('rgb to hsv transform done in %.2f secs' % (time.time() - start))
# Running knn on different neighbor sizes
train_acc_knn_feature = np.zeros(neighbours_range.shape)
test_acc_knn_feature = np.zeros(neighbours_range.shape)
start = time.time()
for i, number in enumerate(neighbours_range):
    # TODO: Train and fit knn with k=number and populate acc arrays
    pass
print('k-NN on feature space: done in %.2f secs' % (time.time() - start))
plt.figure(figsize=(12, 12))
plt.plot(neighbours_range, train_acc_knn_feature, label='Training Accuracy')
plt.plot(neighbours_range, test_acc_knn_feature, label='Test Accuracy')
plt.xlabel('Neighbor size', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('k-NN on feature space', fontsize=16)
plt.grid()
plt.legend(fontsize=16)
plt.savefig('figure3.pdf')
plt.close()
# TODO: Find k for best training and test accuracy
best_k_train = None
best_k_test = None
print('k-NN on features: neighbor size %i maximises training accuracy' %
      best_k_train)
print('k-NN on features: neighbor size %i maximises test accuracy' % best_k_test)
# TODO: Load, fit and predict on testing data with best_k_test. knn_best_predict should be an
# array of predicted labels
knn_best_predict_feature = None
# TODO: Find incorrect and correct samples
for i in range(4):
    pass
# TODO: Find a correct label for label=i+1
right_indx = None
# TODO: Find a incorrect label for label=i+1
wrong_indx = None
# TODO: Load in data for right and wrong indices
rgb_data_right = None
nir_data_right = None
rgb_data_wrong = None
nir_data_wrong = None
# Save data into images
sample_right = Image.fromarray(rgb_data_right.astype(np.uint8))
sample_right_nir = Image.fromarray(nir_data_right.astype(np.uint8))
sample_wrong = Image.fromarray(rgb_data_wrong.astype(np.uint8))
sample_wrong_nir = Image.fromarray(nir_data_wrong.astype(np.uint8))
sample_right.save('samples/knn_feature_class' + str(i) + '_right.png')
sample_right_nir.save('samples/knn_feature_class' + str(i) + '_right_nir.png')
sample_wrong.save('samples/knn_feature_class' + str(i) + '_wrong.png')
sample_wrong_nir.save('samples/knn_feature_class' + str(i) + '_wrong_nir.png')
C_list_feature = np.arange(31, 61)
train_acc_svm_feature = np.zeros(C_list_feature.shape)
test_acc_svm_feature = np.zeros(C_list_feature.shape)
start = time.time()
for i, c in enumerate(C_list_feature):
    # TODO: Train and fit svm with gamma=c/10 and populate acc arrays
    pass
print('SVM on feature space: done in %.2f secs' % (time.time() - start))
plt.figure(figsize=(12, 12))
plt.plot(C_list_feature, train_acc_svm_feature, label='Training Accuracy')
plt.plot(C_list_feature, test_acc_svm_feature, label='Test Accuracy')
plt.xlabel('C', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('SVM on feature space', fontsize=16)
plt.grid()
plt.legend(fontsize=16)
plt.savefig('figure4.pdf')
plt.close()
# TODO: Find C for best training and test accuracy
best_C_train = None
best_C_test = None
print('SVM on features: neighbor size %i maximises training accuracy' %
      best_C_train)
print('SVM on features: neighbor size %i maximises test accuracy' % best_C_test)
# TODO: Load, fit and predict on testing data with best_C_test. svm_best_predicti_feature should be an
# array of predicted labels
svm_best_predict_feature = None
# Finding incorrect and correct labels for svm_best_predict_feature
for i in range(4):
    pass
# TODO: Find a correct label for label=i+1
right_indx = None
# TODO: Find a incorrect label for label=i+1
wrong_indx = None
# TODO: Load in data for right and wrong indices
rgb_data_right = None
nir_data_right = None
rgb_data_wrong = None
nir_data_wrong = None
# Convert data into images
sample_right = Image.fromarray(rgb_data_right.astype(np.uint8))
sample_right_nir = Image.fromarray(nir_data_right.astype(np.uint8))
sample_wrong = Image.fromarray(rgb_data_wrong.astype(np.uint8))
sample_wrong_nir = Image.fromarray(nir_data_wrong.astype(np.uint8))
sample_right.save('samples/svm_feature_class' + str(i) + '_right.png')
sample_right_nir.save('samples/svm_feature_class' + str(i) + '_right_nir.png')
sample_wrong.save('samples/svm_feature_class' + str(i) + '_wrong.png')
sample_wrong_nir.save('samples/svm_feature_class' + str(i) + '_wrong_nir.png')
# TODO: get # of support vectors
SV_indx = None
print('there are %i support vectors' % SV_indx.shape[0])
# TODO: pick 4 random support vectors
SV_indx_4 = None
for i in range(4):
    indx = SV_indx_4[i]
raw_img = train_x_raw[:, :, :, indx]
# TODO: From raw_img get the rgb data and optionally nir data
rgb_data = None
nir_data = None
# Convert data into images
sample = Image.fromarray(rgb_data.astype(np.uint8))
sample_nir = Image.fromarray(nir_data.astype(np.uint8))
sample.save('samples/support_vector_' + str(i + 1) + '.png')
sample_nir.save('samples/support_vector_' + str(i + 1) + '_nir.png')
label = train_y[indx]
print('support vector ' + str(i + 1) + ' is of class ' + label_dict[label, 1][0])
