def _gen_indices(i, k, s):
    """Generate start indices for patch extraction along one dimension.

    Args:
        i (int): Size of the dimension.
        k (int): Patch size.
        s (int): Stride.

    Yields:
        int: Start index for a patch.
    """
    assert i >= k, "Sample size has to be bigger than the patch size"
    for j in range(0, i - k + 1, s):
        yield j
    if j + k < i:
        yield i - k


def build_slices_3d(dataset, patch_shape, stride_shape):
    """Iterate over a 3D or 4D dataset patch-by-patch and build an array of slice positions.

    Args:
        dataset (np.ndarray): Input dataset, 3D or 4D.
        patch_shape (tuple): Shape of the patch (z, y, x).
        stride_shape (tuple): Stride for each dimension (z, y, x).

    Returns:
        list: List of tuples of slice objects for each patch.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
    """
    slices = []

    if dataset.ndim == 4:
        in_channels, i_z, i_y, i_x = dataset.shape
    else:
        i_z, i_y, i_x = dataset.shape

    k_z, k_y, k_x = patch_shape
    s_z, s_y, s_x = stride_shape
    z_steps = _gen_indices(i_z, k_z, s_z)
    for z in z_steps:
        y_steps = _gen_indices(i_y, k_y, s_y)
        for y in y_steps:
            x_steps = _gen_indices(i_x, k_x, s_x)
            for x in x_steps:
                slice_idx = (slice(z, z + k_z), slice(y, y + k_y), slice(x, x + k_x))
                if dataset.ndim == 4:
                    slice_idx = (slice(0, in_channels),) + slice_idx
                slices.append(slice_idx)
    return slices


def build_slices(dataset, patch_shape, stride_shape, use_shape_only=True):
    """Iterate over a dataset patch-by-patch with a given stride and build an array of slice positions.

    If use_shape_only is True, dataset should be a tuple or array containing the shape of the dataset rather than the data itself.

    Args:
        dataset (tuple or np.ndarray): Dataset shape or array.
        patch_shape (tuple): Shape of the patch (y, x).
        stride_shape (tuple): Stride for each dimension (y, x).
        use_shape_only (bool): Whether to use only the shape information.

    Returns:
        list: List of tuples of slice objects for each patch.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
    """
    slices = []
    if use_shape_only:
        assert type(dataset) is tuple, "When using use_shape_only parameter, dataset argument should just be the shape of the dataset as a tuple"
        if len(dataset) == 3:
            in_channels, i_y, i_x = dataset
        else:
            i_y, i_x = dataset

    else:

        if dataset.ndim == 3:
            in_channels, i_y, i_x = dataset.shape
        else:
            i_y, i_x = dataset.shape

    k_y, k_x = patch_shape
    s_y, s_x = stride_shape
    y_steps = _gen_indices(i_y, k_y, s_y)
    for y in y_steps:
        x_steps = _gen_indices(i_x, k_x, s_x)
        for x in x_steps:
            slice_idx = (slice(y, y + k_y), slice(x, x + k_x))
            if use_shape_only:
                if len(dataset) == 3:
                    slice_idx = (slice(0, in_channels),) + slice_idx
            else:
                if dataset.ndim == 3:
                    slice_idx = (slice(0, in_channels),) + slice_idx
            slices.append(slice_idx)
    return slices


def build_slices_zarr_2d(dataset, padding, patch_size, stride):
    """Build patch indices for 2D zarr datasets with padding.

    Args:
        dataset (np.ndarray): Input 2D dataset.
        padding (tuple): Padding for (top, bottom, left, right).
        patch_size (tuple): Size of the patch (height, width).
        stride (tuple): Stride for each dimension (height, width).

    Returns:
        list: List of (h, w) tuples for patch start indices.
    """
    patch_indices = []

    H, W = dataset.shape  # (height, width)
    padded_H = H + padding[0] + padding[1]  # Account for padding
    padded_W = W + padding[2] + padding[3]

    h_indices = range(0, padded_H - patch_size[0] + 1, stride[0])
    w_indices = range(0, padded_W - patch_size[1] + 1, stride[1])

    for h in h_indices:
        for w in w_indices:
            patch_indices.append((h, w))

    return patch_indices


def build_slices_zarr_3d(dataset, padding, patch_size, stride):
    """Build patch indices for 3D zarr datasets with padding.

    Args:
        dataset (np.ndarray): Input 3D dataset.
        padding (tuple): Padding for (front, back, top, bottom, left, right).
        patch_size (tuple): Size of the patch (depth, height, width).
        stride (tuple): Stride for each dimension (depth, height, width).

    Returns:
        list: List of (d, h, w) tuples for patch start indices.
    """
    patch_indices = []

    D, H, W = dataset.shape  # (depth, height, width)
    padded_D = D + padding[0] + padding[1]
    padded_H = H + padding[2] + padding[3]  # Account for padding
    padded_W = W + padding[4] + padding[5]

    d_indices = range(0, padded_D - patch_size[0] + 1, stride[0])
    h_indices = range(0, padded_H - patch_size[1] + 1, stride[1])
    w_indices = range(0, padded_W - patch_size[2] + 1, stride[2])

    for d in d_indices:
        for h in h_indices:
            for w in w_indices:
                patch_indices.append((d, h, w))

    return patch_indices
