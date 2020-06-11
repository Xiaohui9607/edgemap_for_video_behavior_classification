import cv2
import glob
import os
import random
from options import Options
import numpy as np
from data import build_dataloader_CY101
from sklearn.cluster import KMeans


BEHAVIORS = ['crush', 'grasp', 'lift_slow', 'shake', 'poke', 'push', 'tap', 'low_drop', 'hold']

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


class extractor:
    def __init__(self, n_features=20):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)

    def extract(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        # build_histogram(descriptor, kmeans)
        return keypoints, descriptors

def load_dataset(path):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    samples = glob.glob(os.path.join(path, "vision_data_*", "*", "*", "*"))

    for sample in samples:
        for behavior in BEHAVIORS:
            f = sorted(glob.glob(os.path.join(sample, behavior, "*.jpg")),
                       key=lambda x:x.split('/')[-1].strip("vision_"))
            if random.random()<0.2:
                x_test.append(f)
                y_test.append(BEHAVIORS.index(behavior))
            else:
                x_train.append(f)
                y_train.append(BEHAVIORS.index(behavior))
    return x_train, y_train, x_test, y_test

def preprocess(sequences, ext):
    ret = []
    for i, sequence in enumerate(sequences):
        ret.append([list(ext.extract(cv2.imread(image_path))[1]) for image_path in sequence])
        if i == 100:
            break
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
    from keras.datasets import imdb

    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    ext = extractor(60)
    kmeans = KMeans(n_clusters=800)

    x_train, y_train, x_test, y_test = load_dataset("/Users/golf/code/data/CY101")
    x_train = preprocess(x_train, ext)
    desc_list = sum(sum(x_train, []),[])
    kmeans.fit(desc_list)
    x_train = preprocess_hist(x_train, kmeans)




    image = cv2.imread("/Users/golf/code/data/CY101/vision_data_part1/basket_green/trial_1/exec_2/grasp/vision_1301605035718448.jpg")
    k,p = ext.extract(image)
    pass
