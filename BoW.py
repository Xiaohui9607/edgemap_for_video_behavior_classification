import cv2
import glob
import os
import random
from options import Options
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from PDBF import pdbf

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout

import pickle

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

def create_dataset(path):
    samples = glob.glob(os.path.join(path, "vision_data_*", "*", "*", "*"))
    i=0
    for sample in samples:
        for behavior in BEHAVIORS:
            sequence = sorted(glob.glob(os.path.join(sample, behavior, "*.jpg")),
                       key=lambda x:x.split('/')[-1].strip("vision_"))
            l = [processLBP(image_path) for image_path in sequence]
            pickle.dump(l, open("{}.pkl".format('_'.join(sample.split('/')[-3:])+'_'+behavior), 'wb'))
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
        f = [np.pad(elem, (0,122-elem.shape[0]),'constant', constant_values=0) for elem in f]
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


if __name__ == '__main__':
    # opt = Options().parse()
    # opt.baseline = True
    # from keras.datasets import imdb

    create_dataset("/home/golf/code/data/CY101")
    # x_train, y_train, x_test, y_test = load_dataset("/home/golf/code/edgemap_for_video_behavior_classification/pkledge")
    #
    # # desc_list = sum(x_train, [])
    # max_len = max([len(elem) for elem in x_train])
    # import keras
    # # x_train = [np.array(elem+[np.zeros(122) for _ in range(max_len-len(elem))]) for elem in x_train]
    # x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, dtype='float32')
    # x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, dtype='float32')
    # # x_train = np.stack(x_train)
    # model = Sequential()
    #
    # model.add(LSTM(32, input_shape=(max_len, 122)))
    # # model.add(Dense(1))
    # # model.add(Dropout(0.2))
    # model.add(Dense(len(BEHAVIORS), activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=[keras.metrics.SparseCategoricalAccuracy()])
    # model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)
    # scores = model.evaluate(x_train, y_train,verbose=0)
    # print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))
    # scores = model.evaluate(x_test, y_test,verbose=0)
    # print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))
    # pass
