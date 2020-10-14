import numpy as np
from scipy.interpolate import interp1d


def pad_patch_size(patch_size, reduced):
    """Pads the patch size to account for size change in conv.

    Args:
        patch_size (iterable[int]): The patch size to pad.
        reduced (int): The number of pixels to pad.

    Returns:
        list[int]: The padded patch size.

    """
    patch_size = list(patch_size)
    patch_size[0] = patch_size[0] + reduced
    patch_size[1] = patch_size[1] + reduced
    return patch_size


def calc_fwhm(kernel):
    """Calculates the full width at half maximum (FWHM) using linear interp.

    Args:
        kernel (numpy.ndarray): The kernel to calculat the FWHM from.

    Returns
    -------
    fwhm: float
        The calculated FWHM. It is equal to ``right - left``.
    left: float
        The position of the left of the FWHM.
    right: float
        The position of the right of the FWHM.

    """
    kernel = kernel.squeeze()
    half_max = float(np.max(kernel)) / 2
    indices = np.where(kernel > half_max)[0] 
    left = indices[0]
    if left > 0:
        interp = interp1d((kernel[left-1], kernel[left]), (left - 1, left))
        left = interp(half_max)
    right = indices[-1]
    if right < len(kernel) - 1:
        interp = interp1d((kernel[right+1], kernel[right]), (right + 1, right))
        right = interp(half_max)
    fwhm = right - left
    return fwhm, left, right
