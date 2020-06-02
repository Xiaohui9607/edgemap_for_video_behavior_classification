import numpy as np
from ctypes import cdll, c_void_p , c_uint32, c_float
from PIL import Image
try:
    lib = cdll.LoadLibrary('./PDBF/libPDBF.so')
except:
    lib = cdll.LoadLibrary('../PDBF/libPDBF.so')

def pdbf(input, nbitplanes, beta, winsize, sigma, kernelsize, use_gaussian):
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
        c_void_p(res.ctypes.data)
    )
    return res


def rgbpdbfs(input, nbitplanes, use_gaussian=False, sigma=1.0, kernelsize=5):
    if isinstance(nbitplanes, int):
        nbitplanes = [nbitplanes]
    res = []
    input = np.array(input)
    for nbitplane in nbitplanes:
        res.extend([pdbf(input[..., i], nbitplanes=nbitplane, beta=0, winsize=2,
                   sigma=sigma, kernelsize=kernelsize, use_gaussian=use_gaussian) for i in range(3)])
    res = np.stack(res, axis=-1)
    return res

def graypdbfs(input, nbitplanes, use_gaussian=False, sigma=1.0, kernelsize=5):
    if isinstance(nbitplanes, int):
        nbitplanes = [nbitplanes]
    res = []
    gray_input = input.convert('L')
    gray_input = np.array(gray_input)

    for nbitplane in nbitplanes:
        res.append(pdbf(gray_input, nbitplanes=nbitplane, beta=0, winsize=2,
                   sigma=sigma, kernelsize=kernelsize, use_gaussian=use_gaussian))
    res = np.stack(res, axis=-1)
    return res

if __name__ == '__main__':

    graypdbfs()