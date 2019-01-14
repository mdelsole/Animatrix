import numpy as np
import scipy.ndimage.filters as snf


""" 
Arguments:
-image: image data, as a 2D array
-v1Filters: the V1 filters, in the format produced by buildV1filters()

Each V1 cell is supposed to multiply its inputs by its filter, and then divide the result by the norm of its input 
(normalized dot-product, AKA cosine between inputs and filter). In effect this compares the input to the cell's filter.

We do this efficiently by convolving the image with each filter, then dividing the results pointwise with the square 
root of the convolution of the squared image with a uniform filter of adequate size (for each scale). 

Returns a list of 3D arrays (one per V1 scale). Each 3D array is a depth-stack of 4 2D maps, one per orientation.

This function uses Fourier-based convolutions, which help with very large filters. 
"""


def runV1cells(image, v1Filters):
    print("Running V1 Cells")

    # Convert image to float
    img = image.astype(float)
    output=[]
    imgsq = img ** 2
    cpt = 0

    # Each element in v1Filters is the set of filters (of various orientations) for a particular scale. We also use the
    # index of this scale for debugging purposes in an assertion.
    for scaleidx, fthisscale in enumerate(v1Filters):
        # We assume that at any given scale, all the filters have the same RF size, and so the RF size is simply the
        # x-size of the filter at the 1st orientation. Note that all RFs are assumed square.
        RFSIZE = fthisscale[0].shape[0]
        assert RFSIZE == range(7,30,2)[scaleidx]
        outputsAllOrient = []
        # The output of every S1 neuron is divided by the
        # Euclidan norm (root-sum-squares) of its inputs; also, we take the
        # absolute value.
        # As seen in J. Mutch's hmin and Riesenhuber-Serre-Bileschi code.
        # Perhaps a SIGMA in the denominator would be good here?...
        # Though it might need to be adjusted for filter size...
        tmp = snf.uniform_filter(imgsq, RFSIZE) * RFSIZE * RFSIZE
        tmp[tmp < 0] = 0.0
        normim = np.sqrt(tmp) + 1e-9 + opt.SIGMAS1
        assert np.min(normim > 0)
        for o in range(0, 4):
            # fft convolution; note that in the case of S1 filters, reversing
            # the filters seems to have no effect, so convolution =
            # cross-correlation (...?)
            tmp = np.fft.irfft2(np.fft.rfft2(img) * np.fft.rfft2(fthisscale[o], img.shape))
            # Using the fft convolution requires the following (fun fact: -N/2 != -(N/2) ...)
            tmp = np.roll(np.roll(tmp, -(RFSIZE / 2), axis=1), -(RFSIZE / 2), axis=0)
            # Normalization
            tmp /= (normim)
            fin = np.abs(tmp[RFSIZE / 2:-RFSIZE / 2, RFSIZE / 2:-RFSIZE / 2])
            assert np.max(fin) < 1
            outputsAllOrient.append(fin)
        # We stack together the orientation maps of all 4 orientations into one single
        # 3D array, for each scale/RF size.
        output.append(np.dstack(outputsAllOrient[:]));
        cpt += 1
    return output