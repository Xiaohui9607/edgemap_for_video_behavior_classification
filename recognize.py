from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
import cv2
import os
import random
import glob
from keras.preprocessing import sequence
from sequence_classifiers import CNNSequenceClassifier
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

    desc = LocalBinaryPatterns(24, 8)
    hist_train_list = []
    hist_train_labels_list= []
    hist_test_list = []
    hist_test_labels_list = []

    x_train, y_train, x_test, y_test = load_dataset("/Users/ramtin/PycharmProjects/data/CY101")

    for i, a_sequence in enumerate(x_train[:10]):
        print("i", i)
        for an_image_path in a_sequence:
            image = cv2.imread(an_image_path)
            #image = rgbpdbfs(image, nbitplanes=[3], decomp_method=0, p_code=-1, n_code=-1) * 255
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist_train = desc.describe(gray)
            hist_train_list.append(hist_train)
            hist_train_labels_list.append(y_train[i])

    # train a Linear SVM on the data
    # model = LinearSVC(C=100.0, random_state=42)
    # model.fit(data_train, y_train[:100])

    for j, a_sequence in enumerate(x_test[:5]):
        print("j", j)
        for an_image_path in a_sequence:
            image = cv2.imread(an_image_path)
            #image = rgbpdbfs(image, nbitplanes=[3], decomp_method=0,p_code=-1, n_code=-1) * 255
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist_test = desc.describe(gray)
            hist_test_list.append(hist_test)
            hist_test_labels_list.append(y_test[j])

        # prediction = model.predict(hist_test.reshape(1, -1))
        # print(prediction)

        '''
        # display the image and the prediction
        cv2.putText(image, str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        '''

    maxlen = 20
    hist_train_list = sequence.pad_sequences(hist_train_list, maxlen=maxlen, dtype=float)
    hist_test_list = sequence.pad_sequences(hist_test_list, maxlen=maxlen, dtype=float)

    clf = CNNSequenceClassifier(epochs=10)
    clf.fit(hist_train_list, hist_train_labels_list)
    print(clf.score(hist_test_list, hist_test_labels_list))