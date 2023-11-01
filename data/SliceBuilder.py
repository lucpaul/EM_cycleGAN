
def build_slices(dataset, patch_shape, stride_shape):
    """Iterates over a given n-dim dataset patch-by-patch with a given stride
    and builds an array of slice positions.

    Returns:
        list of slices, i.e.
        [(slice, slice, slice, slice), ...] if len(shape) == 4
        [(slice, slice, slice), ...] if len(shape) == 3
    """
    slices = []
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
            slice_idx = (
                slice(y, y + k_y),
                slice(x, x + k_x)
            )
            if dataset.ndim == 3:
                slice_idx = (slice(0, in_channels),) + slice_idx
            slices.append(slice_idx)
    return slices

def _gen_indices(i, k, s):
    assert i >= k, 'Sample size has to be bigger than the patch size'
    for j in range(0, i - k + 1, s):
        yield j
    if j + k < i:
        yield i - k
