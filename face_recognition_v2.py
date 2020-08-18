import cv2
import glob
import random
from options import Options
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from PDBF import pdbf
import os
import pickle
import argparse
from sklearn import svm
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ratio_list = [0.5, 0.25, 0.125, 0.0625, 0.0375, 0.01875]

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result = cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def processLBP(img_path, radius = 15):
    img = cv2.imread(img_path)

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### uncomment for creating dataset for edgemap
    # im_gray = pdbf(input=im_gray, nbitplanes=3, beta=0, winsize=2, sigma=1, kernelsize=5, use_gaussian=False,
    #             decomp_method=0)*255

    # Number of points to be considered as neighbourers
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    return hist

# class extractor:
#     def __init__(self, n_features=20):
#         self.sift = cv2.xfeatures2d.(nfeatures=n_features)
#
#     def extract(self, image):
#         keypoints, descriptors = self.sift.detectAndCompute(image, None)
#         # build_histogram(descriptor, kmeans)
#         return keypoints, descriptors

def create_dataset(path, ratio=1.0):

    samples = glob.glob(os.path.join(path, "*/*/*.jpg"))
    samples = sorted(samples, key=lambda x: x.split('/')[-1])

    # samples = [sp for sp in samples if random.random() < ratio]

    for i, sample in enumerate(samples):
        l = processLBP(sample)

        #temp = sample.split('/')
        # pickle.dump(l, open("../data/temp/pickle_RGB_E/{}.pkl".format
        #                     (temp[-1].strip(".jpg") + '_' + temp[-2]), 'wb'))
        # print(i)
        # i += 1


def load_dataset(opt):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    samples = glob.glob(os.path.join(opt.path, "*.pkl"))

    for index, sample in enumerate(samples):
        f = pickle.load(open(sample, 'rb'))

        if int(sample.split('/')[-1].split('_')[3]) % 2 == 1:
            x_train.append(f)
            y_train.append(sample.split('_')[-1].strip('.pkl'))
        else:
            x_test.append(f)
            y_test.append(sample.split('_')[-1].strip('.pkl'))

    if opt.ratio <1:
        x_train_ratio = []
        y_train_ratio = []
        for index, sample in enumerate(x_train):
            if random.random() < opt.ratio:
                x_train_ratio.append(sample)
                y_train_ratio.append(y_train[index])
        x_train = x_train_ratio
        y_train = y_train_ratio

    return x_train, y_train, x_test, y_test

def preprocess(sequences):
    ret = []
    for i, sequence in enumerate(sequences):
        l = [processLBP(image_path) for image_path in sequence]
        # pickle.dump(l, open("{}.pkl".format(i), 'wb'))
        # print(i)
    return ret

def preprocess_hist(desc_sequences, km):
    ret = []
    for i, sequence in enumerate(desc_sequences):
        ret.append(build_histogram(sequence, km))
        if i == 100:
            break
        print(i)
    return ret


def run_BoW(opt):

    x_train, y_train, x_test, y_test = load_dataset(opt)

    ### if using edge map, uncomment the next 3 lines.
    # from keras.preprocessing.sequence import pad_sequences
    # x_train = pad_sequences(x_train, padding='post', maxlen=5, dtype='float32') # maxlen=[5-11]
    # x_test = pad_sequences(x_test, padding='post', maxlen=5, dtype='float32') # maxlen=[5-11]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    sumtime = 0
    error = 0

    for i, a_test in enumerate(x_test):
        st = time.time()
        y_predict = clf.predict(a_test.reshape(1,-1))

        if y_predict != y_test[i]:
            error += 1
            print("error", error)
            print("y_predict", y_predict)
            print("y_test[i]", y_test[i])

        ed = time.time()
        sumtime += ed - st
    # print("sumtime", sumtime)
    acurracy = 100 * (len(x_test) - error) / len(x_test)
    print("total error", error)
    print("no test samples", len(x_test))

    print('Accuracy on test data: {}% \n Error on test data: {}'.format(acurracy, 100 - acurracy))

if __name__ == '__main__':

    """
    Note: Since the results very not good, I did not spend time refarctoring the code and hard coded some part of the 
    code for now.
    There are 113 folders (Thus 113 classes). Each of them contain 5 images. 3 are used for training, and 2 for testing.
    When using pdbf, since the returned features of different images are not of the same size, They need to padded. 
    Thus in the method "run_BoW", uncomment the corresponding lines. Also I tried different max_length (5 - 11)
    """

    # create_dataset(path="../data/RGB_E", ratio=1.0)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', default="../data/pickle_RGB_E",
                        help='directory containing pickle files.')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of used train set')
    parser.add_argument('--epochs', type=int, default=30, help='# total training epoch')

    opt = parser.parse_args()

    run_BoW(opt)