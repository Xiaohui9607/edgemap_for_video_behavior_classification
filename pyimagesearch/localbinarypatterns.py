from skimage import feature, color, io
import numpy as np
from PDBF import rgbpdbfs
import cv2


def crop(im):
    height, width = im.shape[1:]
    width = max(height, width)
    im = im[:, :width, :width]
    return im

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def crop_image(self, image):
        height, width = image.shape[:2]
        width = max(height, width)
        image = image[:width, :width, :]
        return image

    def describe(self, image, eps=1e-7, flag='normal'): #flag: either normal or PDBF
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns

        image=io.imread(image)
        image = self.crop_image(image)

        if flag=='PDBF':
            image = rgbpdbfs(image, nbitplanes=[3], decomp_method=0, p_code=-1, n_code=-1) * 255

        image = color.rgb2gray(image)

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
    img = '../ramtin.jpg'

    sample = LocalBinaryPatterns(numPoints=10, radius=20)
    hist = sample.describe(image= img)
    print(hist)





