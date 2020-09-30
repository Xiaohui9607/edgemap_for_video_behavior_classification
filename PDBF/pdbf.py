import numpy as np
from ctypes import cdll, c_void_p , c_uint32, c_float
from PIL import Image
try:
    lib = cdll.LoadLibrary('./PDBF/libPDBF.so')
except:
    lib = cdll.LoadLibrary('../PDBF/libPDBF.dylib')

def pdbf(input, nbitplanes, beta, winsize, sigma, kernelsize, use_gaussian, decomp_method=0, p_code=-1, n_code=-1):
    use_gaussian = 1 if use_gaussian else 0
    res = np.zeros_like(input)

    lib.edgedetect(
        c_void_p(input.ctypes.data),  # input
        c_uint32(input.shape[0]),  # width
        c_uint32(input.shape[1]),  # height
        c_uint32(nbitplanes),  # nbitplanes
        c_float(beta),  # beta
        c_uint32(winsize),  # winsize
        c_float(sigma),  # sigma (smoothing)
        c_uint32(kernelsize),  # kernel size (smoothing)
        c_uint32(use_gaussian),  # use/not use gaussian
        c_void_p(res.ctypes.data),
        c_uint32(p_code),
        c_uint32(n_code),
        c_uint32(decomp_method),
    )
    return res


def rgbpdbfs(input, nbitplanes, use_gaussian=False, sigma=1.0, kernelsize=5, decomp_method=0, p_code=-1, n_code=-1):
    if isinstance(nbitplanes, int):
        nbitplanes = [nbitplanes]
    res = []
    input = np.array(input)
    for nbitplane in nbitplanes:
        res.extend([pdbf(input[..., i], nbitplanes=nbitplane, beta=0, winsize=2,sigma=sigma, kernelsize=kernelsize,
                         use_gaussian=use_gaussian, decomp_method=decomp_method, p_code=p_code, n_code=n_code)
                    for i in range(3)])
    res = np.stack(res, axis=-1)
    return res

def graypdbfs(input, nbitplanes, use_gaussian=False, sigma=1.0, kernelsize=5, decomp_method=0, p_code=-1, n_code=-1):
    if isinstance(nbitplanes, int):
        nbitplanes = [nbitplanes]
    res = []
    gray_input = input.convert('L')
    gray_input = np.array(gray_input)

    for nbitplane in nbitplanes:
        res.append(pdbf(gray_input, nbitplanes=nbitplane, beta=0, winsize=2, sigma=sigma,
                        kernelsize=kernelsize, use_gaussian=use_gaussian, decomp_method=decomp_method, p_code=p_code,
                        n_code=n_code))
    res = np.stack(res, axis=-1)
    return res

if __name__ == '__main__':
    import cv2
    image = cv2.imread("/Users/ramtin/Desktop/LinkedIn_Image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = pdbf(input=image, nbitplanes=3, beta=0, winsize=2, sigma=1, kernelsize=5, use_gaussian=False,
                decomp_method=1, p_code=2, n_code=8)
    cv2.imshow("the_img", edge*255)
    cv2.waitKey(0)
    pass
