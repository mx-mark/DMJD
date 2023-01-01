# --------------------------------------------------------
# GPU-accelerate HOG implementation based on pytorch
# --------------------------------------------------------
import torch


def _hog_channel_gradient_torch(channel):
    """Compute unnormalized gradient image along `row` and `col` axes.

    Parameters
    ----------
    channel : (B, C, H, W) tensor(RGB image).

    Returns
    -------
    g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
    """
    g_row = torch.empty(channel.shape, device=channel.device, dtype=channel.dtype)
    g_row[:, :, 0, :] = 0
    g_row[:, :, -1, :] = 0
    g_row[:, :, 1:-1, :] = channel[:, :, 2:, :] - channel[:, :, :-2, :]
    g_col = torch.empty(channel.shape, device=channel.device, dtype=channel.dtype)
    g_col[:, :, :, 0] = 0
    g_col[:, :, :, -1] = 0
    g_col[:, :, :, 1:-1] = channel[:, :, :, 2:] - channel[:, :, :, :-2]

    return g_row, g_col


def _hog_normalize_block_torch(block, method, eps=1e-5):
    if method == 'L2':
        out = block / torch.sqrt(torch.sum(block**2, axis=-1, keepdims=True) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out


def patchify(imgs, cell_rows, cell_columns):
    """
    imgs: (B, C, H, W)
    x: (B, C, H//patch_size, W//patch_size, patch_size**2)
    """
    b, c = imgs.shape[0], imgs.shape[1]
    h, w = imgs.shape[-2] // cell_rows, imgs.shape[-1] // cell_columns
    x = imgs.reshape(shape=(b, c, h, cell_rows, w, cell_columns))
    x = torch.einsum('nchpwq->nchwpq', x)
    x = x.reshape(shape=(b, c, h, w, 1, cell_rows*cell_columns))
    return x


def _hog_histograms_torch(gradient_columns, gradient_rows, number_of_orientations, cell_rows, cell_columns):
    magnitude = patchify(torch.hypot(gradient_columns, gradient_rows), cell_rows, cell_columns).repeat(1, 1, 1, 1, number_of_orientations, 1)
    orientation = torch.rad2deg(torch.arctan2(gradient_rows, gradient_columns)) % 180
    orientation = torch.div(orientation, (180. / number_of_orientations), rounding_mode='floor')
    orientation = patchify(orientation, cell_rows, cell_columns).repeat(1, 1, 1, 1, number_of_orientations, 1)

    orientation_bin = torch.tensor(list(range(number_of_orientations)), device=orientation.device, dtype=orientation.dtype)
    orientation_bin = orientation_bin.reshape(shape=(1, 1, 1, 1, number_of_orientations, 1))
    mask = (orientation == orientation_bin).to(torch.int) # expand to same dimension and convert bool to int
    orientation_histogram = torch.sum(magnitude * mask, axis=-1) / (cell_rows * cell_columns)
    return orientation_histogram


def hog_torch(images, orientations=9, pixels_per_cell=(8, 8), block_norm='L2', *, transform_sqrt=False):
    images = images.to(torch.float)
    if transform_sqrt:
        images = torch.sqrt(images)

    # compute gradients of pixels
    g_row, g_col = _hog_channel_gradient_torch(images)
    c_row, c_col = pixels_per_cell
    n_cells_row = int(images.shape[-2] // c_row)  # number of cells along row-axis
    n_cells_col = int(images.shape[-1] // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = _hog_histograms_torch(g_col, g_row, orientations, c_row, c_col)
    
    return orientation_histogram