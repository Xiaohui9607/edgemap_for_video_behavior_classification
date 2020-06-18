from skimage import feature
import numpy as np
import cv2


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist




if __name__=='__main__':
    from PIL import Image
    img = Image.open('../ramtin.jpg').convert('LA')
    print(img)
    # img.save('greyscale.png')

    sample = LocalBinaryPatterns(numPoints=10, radius=20)
    hist = sample.describe(image= img)
    print(hist)





