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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['KMP_DUPLICATE_LIB_OK']='True'


BEHAVIORS = ['crush', 'grasp', 'lift_slow', 'shake', 'poke', 'push', 'tap', 'low_drop', 'hold']

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result = cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def processLBP(img_path, radius = 15):
    # final_path = utils.adress_file(img_path, "LBP")
    img = cv2.imread(img_path)

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    im_gray = pdbf(input=im_gray, nbitplanes=3, beta=0, winsize=2, sigma=1, kernelsize=5, use_gaussian=False,
                decomp_method=0)*255

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

    samples = glob.glob(os.path.join(path, "vision_data_*", "*", "*", "*"))
    samples = [sp for sp in samples if random.random() < ratio]

    print(samples)
    print(len(samples))

    i=0
    for sample in samples:
        for behavior in BEHAVIORS:
            sequence = sorted(glob.glob(os.path.join(sample, behavior, "*.jpg")),
                       key=lambda x:x.split('/')[-1].strip("vision_"))
            l = [processLBP(image_path) for image_path in sequence]
            pickle.dump(l, open("../data/pickle_ratio_{}/{}.pkl".format(ratio,'_'.join(sample.split('/')[-3:])+'_'+behavior), 'wb'))
            print(i)
            i+=1

def load_dataset(path):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    samples = glob.glob(os.path.join(path, "*.pkl"))
    for sample in samples:
        behavior = [b for b in BEHAVIORS if b in sample][0]
        f = pickle.load(open(sample, 'rb'))
        f = [np.pad(elem, (0, 122-elem.shape[0]),'constant', constant_values=0) for elem in f]
        if len(f) ==0:
            print(sample)
        if random.random()<0.2:
            x_test.append(f)
            y_test.append(BEHAVIORS.index(behavior))
        else:
            x_train.append(f)
            y_train.append(BEHAVIORS.index(behavior))
    return x_train, y_train, x_test, y_test

def preprocess(sequences):
    ret = []
    for i, sequence in enumerate(sequences):
        l = [processLBP(image_path) for image_path in sequence]
        pickle.dump(l, open("{}.pkl".format(i), 'wb'))
        print(i)
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

    # ratio_list = [0.5, 0.25, 0.125, 0.0625, 0.0375, 0.01875]

    print("ratio", opt.ratio)
    print("epochs", opt.epochs)

    x_train, y_train, x_test, y_test = load_dataset(opt.path)

    x_train_ratio = []
    y_train_ratio = []
    for index, sample in enumerate(x_train):
        if random.random() < opt.ratio:
            x_train_ratio.append(sample)
            y_train_ratio.append(y_train[index])

    x_train = x_train_ratio
    y_train = y_train_ratio

    # desc_list = sum(x_train, [])
    max_len = max([len(elem) for elem in x_train])
    pass
    import keras
    # x_train = [np.array(elem+[np.zeros(122) for _ in range(max_len-len(elem))]) for elem in x_train]
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, dtype='float32')
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, dtype='float32')
    # x_train = np.stack(x_train)
    model = Sequential()
    model.add(LSTM(32, input_shape=(max_len, 122)))
    # model.add(Dense(1))
    # model.add(Dropout(0.2))
    model.add(Dense(len(BEHAVIORS), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    model.fit(x_train, y_train, epochs=opt.epochs, batch_size=32, verbose=1) #epochs=20
    scores = model.evaluate(x_train, y_train, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))
    # print("shape of x_test", x_test.shape)
    import time
    sumtime = 0
    for i in range(100):
        st = time.time()
        model.predict(x_test[0:1])
        # scores = model.evaluate(x_test, y_test,verbose=0)
        # print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))
        ed = time.time()
        sumtime += ed - st
    print("sumtime / 100", sumtime / 100)

    # print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))



if __name__ == '__main__':
    # opt = Options().parse()
    # opt.baseline = True
    # from keras.datasets import imdb

    # create_dataset("../data/CY101", ratio=0.1)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', default="../data/pkl", help='directory containing pickle files.') #options:["../data/pkl", "../data/pkled"]
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of used train set')
    parser.add_argument('--epochs', type=int, default=30, help='# total training epoch')

    opt = parser.parse_args()

    run_BoW(opt)