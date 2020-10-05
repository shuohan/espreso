def pad_patch_size(patch_size, reduce):
    """Pads the patch size to account for size change in conv.

    Args:
        patch_size (iterable[int]): The patch size to pad.
        reduce (int): The number of pixels to pad.

    Returns:
        list[int]: The padded patch size.

    """
    patch_size = list(patch_size)
    patch_size[0] = patch_size[0] + reduce
    return patch_size
