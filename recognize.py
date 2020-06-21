from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
import os
import random
import glob
from keras.preprocessing import sequence
from sequence_classifiers import CNNSequenceClassifier
import cv2
from PDBF import rgbpdbfs
from torchvision import transforms
from sklearn.svm import LinearSVC
from keras.datasets import imdb


BEHAVIORS = ['crush', 'grasp', 'lift_slow', 'shake', 'poke', 'push', 'tap', 'low_drop', 'hold']

def load_dataset(path):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    samples = glob.glob(os.path.join(path, "vision_data_*", "*", "*", "*"))
    for sample in samples:
        for behavior in BEHAVIORS:
            f = sorted(glob.glob(os.path.join(sample, behavior, "*.jpg")),
                       key=lambda x:(x.split('/')[-1].strip("vision_")))
            if random.random()<0.2:
                x_test.append(f)
                y_test.append(BEHAVIORS.index(behavior))
            else:
                x_train.append(f)
                y_train.append(BEHAVIORS.index(behavior))
    return x_train, y_train, x_test, y_test

if __name__=='__main__':

    desc = LocalBinaryPatterns(40, 20)
    hist_train_list, hist_train_labels_list, hist_test_list, hist_test_labels_list = [], [], [], []

    x_train, y_train, x_test, y_test = load_dataset("/Users/ramtin/PycharmProjects/data/CY101")

    for i, a_sequence in enumerate(x_train):
        print("i", i)
        for an_image_path in a_sequence:
            hist_train = desc.describe(an_image_path, flag='normal') #flag: normal or PDBF
            hist_train_list.append(hist_train)
            hist_train_labels_list.append(y_train[i])

    for j, a_sequence in enumerate(x_test):
        print("j", j)
        for an_image_path in a_sequence:
            hist_test = desc.describe(an_image_path, flag='normal') #flag: normal or PDBF
            hist_test_list.append(hist_test)
            hist_test_labels_list.append(y_test[j])

    hist_train_list = sequence.pad_sequences(hist_train_list, dtype=float)
    hist_test_list = sequence.pad_sequences(hist_test_list, dtype=float)

    clf = CNNSequenceClassifier(epochs=10)
    clf.fit(hist_train_list, hist_train_labels_list)
    print(clf.score(hist_test_list, hist_test_labels_list))