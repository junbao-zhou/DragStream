import math
from typing import List, Sequence, Tuple, Union
import torch
from torch import Tensor

from PIL import Image
import numpy as np
import os
from pathlib import Path

torch.set_printoptions(
    linewidth=10000,
)


def _get_center_distance(size: Tuple[int], device: str = "cpu") -> Tensor:
    """Compute the distance of each matrix element to the center.

    Args:
        size (Tuple[int]): [m, n].
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [m, n].
    """
    m, n = size
    i_ind = torch.tile(
        torch.tensor([[[i]] for i in range(m)], device=device), dims=[1, n, 1]
    ).float()  # [m, n, 1]
    j_ind = torch.tile(
        torch.tensor([[[i] for i in range(n)]], device=device), dims=[m, 1, 1]
    ).float()  # [m, n, 1]
    ij_ind = torch.cat([i_ind, j_ind], dim=-1)  # [m, n, 2]
    ij_ind = ij_ind.reshape([m * n, 1, 2])  # [m * n, 1, 2]
    center_ij = torch.tensor(((m - 1) / 2, (n - 1) / 2), device=device).reshape(1, 2)
    center_ij = torch.tile(center_ij, dims=[m * n, 1, 1])
    dist = torch.cdist(ij_ind, center_ij, p=2).reshape([m, n])
    return dist


def _get_ideal_weights(
    size: Tuple[int], D0: int, lowpass: bool = True, device: str = "cpu"
) -> Tensor:
    """Get H(u, v) of ideal bandpass filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (int): The cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """
    center_distance = _get_center_distance(size, device)
    center_distance[center_distance > D0] = -1
    center_distance[center_distance != -1] = 1
    if lowpass is True:
        center_distance[center_distance == -1] = 0
    else:
        center_distance[center_distance == 1] = 0
        center_distance[center_distance == -1] = 1
    return center_distance


def _to_freq(image: Tensor) -> Tensor:
    """Convert from spatial domain to frequency domain.

    Args:
        image (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W]
    """
    img_fft = torch.fft.fft2(image)
    img_fft_shift = torch.fft.fftshift(img_fft)
    return img_fft_shift


def _to_space(image_fft: Tensor) -> Tensor:
    """Convert from frequency domain to spatial domain.

    Args:
        image_fft (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W].
    """
    img_ifft_shift = torch.fft.ifftshift(image_fft)
    img_ifft = torch.fft.ifft2(img_ifft_shift)
    img = img_ifft.real.clamp(0, 1)
    return img


def ideal_bandpass(image: Tensor, D0: int, lowpass: bool = True) -> Tensor:
    """Low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.

    Returns:
        Tensor: [B, C, H, W].
    """
    img_fft = _to_freq(image)
    weights = _get_ideal_weights(img_fft.shape[-2:], D0=D0, lowpass=lowpass, device=image.device)
    img_fft = img_fft * weights
    img = _to_space(img_fft)
    return img


# Butterworth


def _get_butterworth_weights(size: Tuple[int], D0: int, n: int, device: str = "cpu") -> Tensor:
    """Get H(u, v) of Butterworth filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (int): The cutoff frequency.
        n (int): Order of Butterworth filters.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """
    center_distance = _get_center_distance(size=size, device=device)
    weights = 1 / (1 + torch.pow(center_distance / D0, 2 * n))
    return weights


def butterworth(image: Tensor, D0: int, n: int) -> Tensor:
    """Butterworth low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        n (int): Order of the Butterworth low-pass filter.

    Returns:
        Tensor: [B, C, H, W].
    """
    img_fft = _to_freq(image)
    weights = _get_butterworth_weights(image.shape[-2:], D0, n, device=image.device)
    img_fft = weights * img_fft
    img = _to_space(img_fft)
    return img


# def my_butterworth_low_pass_filter(
#     shape,
#     stop_freqs: List[float],
#     n=4,
# ):
#     assert len(shape) == len(stop_freqs)

#     grid = torch.meshgrid(
#         *[torch.arange(s, dtype=torch.float32) for s in shape],
#         indexing='ij',
#     )
#     # ( [shape[0], shape[1], ..., shape[N]] ) * len(shape)
#     indices = torch.stack(grid, dim=-1).float()
#     # print(f"{indices.shape = }")
#     # [shape[0], shape[1], ..., shape[N], len(shape)]

#     max_len = torch.tensor(shape).float()
#     max_len -= 1.0
#     max_len /= 2.0
#     # print(f"{max_len = }")
#     # print(f"{max_len.shape = }")
#     # [len(shape)]
#     max_len = max_len.view(*([1]*len(shape)), -1)
#     # print(f"{max_len.shape = }")
#     # [1, 1, ..., 1, len(shape)]

#     normalized_indices = indices / max_len
#     # [shape[0], shape[1], ..., shape[N], len(shape)]

#     normalized_indices_offset = normalized_indices - 1
#     # print(f"{normalized_indices_offset.shape = }")
#     # [shape[0], shape[1], ..., shape[N], len(shape)]


#     stop_freqs_torch = torch.tensor(stop_freqs).float().view(*([1]*len(shape)), -1)
#     # print(f"{stop_freqs_torch.shape = }")
#     # [1, 1, ..., 1, len(shape)]

#     scaled_normalized_indices_offset = normalized_indices_offset / stop_freqs_torch
#     # print(f"{scaled_normalized_indices_offset.shape = }")
#     # [shape[0], shape[1], ..., shape[N], len(shape)]

#     filter_ = 1.0 / (1.0 + torch.pow(scaled_normalized_indices_offset.norm(p=2, dim=-1), 2 * n))
#     return filter_


# def my_butterworth_low_pass_filter_non_center(
#     shape,
#     stop_freqs: List[float],
#     n=4,
# ):
#     new_shape = [
#         2*i-1
#         for i in shape
#     ]
#     filter_ = my_butterworth_low_pass_filter(
#         new_shape,
#         n=n,
#         stop_freqs=stop_freqs,
#     )

#     if len(shape) == 1:
#         crop_filter = filter_[-shape[0]:]
#     elif len(shape) == 2:
#         crop_filter = filter_[-shape[0]:, -shape[1]:]
#     elif len(shape) == 3:
#         crop_filter = filter_[-shape[0]:, -shape[1]:, -shape[2]:]
#     else:
#         raise ValueError("Shape must be 1D, 2D, or 3D.")
#     return crop_filter


# def my_butterworth_high_pass_filter(
#     shape,
#     stop_freqs: List[float],
#     n=4,
# ):
#     assert len(shape) == len(stop_freqs)

#     grid = torch.meshgrid(
#         *[torch.arange(s, dtype=torch.float32) for s in shape],
#         indexing='ij',
#     )
#     # ( [shape[0], shape[1], ..., shape[N]] ) * len(shape)
#     indices = torch.stack(grid, dim=-1).float()
#     # print(f"{indices.shape = }")
#     # [shape[0], shape[1], ..., shape[N], len(shape)]

#     max_len = torch.tensor(shape).float()
#     max_len -= 1.0
#     max_len /= 2.0
#     # print(f"{max_len = }")
#     # print(f"{max_len.shape = }")
#     # [len(shape)]
#     max_len = max_len.view(*([1]*len(shape)), -1)
#     # print(f"{max_len.shape = }")
#     # [1, 1, ..., 1, len(shape)]

#     normalized_indices = indices / max_len
#     # [shape[0], shape[1], ..., shape[N], len(shape)]

#     normalized_indices_offset = normalized_indices - 1
#     # print(f"{normalized_indices_offset.shape = }")
#     # [shape[0], shape[1], ..., shape[N], len(shape)]


#     stop_freqs_torch = torch.tensor(stop_freqs).float().view(*([1]*len(shape)), -1)
#     # print(f"{stop_freqs_torch.shape = }")
#     # [1, 1, ..., 1, len(shape)]

#     scaled_normalized_indices_offset = stop_freqs_torch / normalized_indices_offset
#     # print(f"{scaled_normalized_indices_offset.shape = }")
#     # [shape[0], shape[1], ..., shape[N], len(shape)]

#     filter_ = 1.0 / (1.0 + torch.pow(scaled_normalized_indices_offset.norm(p=2, dim=-1), 2 * n))
#     return filter_


# def my_butterworth_high_pass_filter_non_center(
#     shape,
#     stop_freqs: List[float],
#     n=4,
# ):
#     new_shape = [
#         2*i-1
#         for i in shape
#     ]
#     filter_ = my_butterworth_high_pass_filter(
#         new_shape,
#         n=n,
#         stop_freqs=stop_freqs,
#     )

#     if len(shape) == 1:
#         crop_filter = filter_[-shape[0]:]
#     elif len(shape) == 2:
#         crop_filter = filter_[-shape[0]:, -shape[1]:]
#     elif len(shape) == 3:
#         crop_filter = filter_[-shape[0]:, -shape[1]:, -shape[2]:]
#     else:
#         raise ValueError("Shape must be 1D, 2D, or 3D.")
#     return crop_filter


# ------------------------ Image loading ------------------------
def load_grayscale_image():
    # Try common sample images; fall back to skimage if available; else ask user to put an image in cwd
    candidates = ["onion.png", "cameraman.tif", "peppers.png", "lena.png", "camera.png"]
    for name in candidates:
        if os.path.exists(name):
            # img = Image.open(name).convert('L')
            img = Image.open(name).convert("RGB")
            image_np = np.asarray(img, dtype=np.float64)
            # print(f"{image_np = }")
            image_np = image_np / 255.0
            # print(f"{image_np = }")
            return image_np

    raise FileNotFoundError(
        "Could not find a local image. Place an image (e.g., cameraman.tif/peppers.png) in the working directory."
    )


# ------------------------ DCT implementations (orthonormal) ------------------------


def dct2_matrix_ortho(N, device="cpu", dtype=torch.float32):
    # T2[k, n] = sqrt(2/N) * beta(k) * cos(pi/N * (n + 0.5) * k), beta(0)=1/sqrt(2)
    n = torch.arange(N, device=device, dtype=dtype)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    W = torch.cos(math.pi / N * (n + 0.5) * k)  # [N, N]
    beta = torch.ones(N, device=device, dtype=dtype)
    beta[0] = 1 / math.sqrt(2.0)
    T = (math.sqrt(2.0 / N) * beta).unsqueeze(1) * W
    return T  # orthonormal; inverse is T.T


def dct1_matrix_ortho(N, device="cpu", dtype=torch.float32):
    # T1[k, n] = sqrt(2/(N-1)) * alpha(k) * alpha(n) * cos(pi/(N-1) * n*k)
    # alpha(0)=alpha(N-1)=1/sqrt(2), else 1. Self-inverse (orthonormal and symmetric).
    if N < 2:
        # N=1 trivial case
        return torch.ones((1, 1), device=device, dtype=dtype)
    n = torch.arange(N, device=device, dtype=dtype)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    C = torch.cos(math.pi / (N - 1) * (n * k))  # [N, N]
    alpha = torch.ones(N, device=device, dtype=dtype)
    alpha[0] = 1 / math.sqrt(2.0)
    alpha[-1] = 1 / math.sqrt(2.0)
    T = math.sqrt(2.0 / (N - 1)) * (alpha.unsqueeze(1) * C * alpha.unsqueeze(0))
    return T  # orthonormal, symmetric, self-inverse


def dct2_ortho(x, T2=None):
    # x: [N] float tensor. Returns DCT-II (orthonormal) [N].
    x = x.reshape(-1)
    N = x.numel()
    if T2 is None:
        T2 = dct2_matrix_ortho(N, device=x.device, dtype=x.dtype)
    return T2 @ x


def idct2_ortho(X, T2=None):
    # Inverse of DCT-II (orthonormal) is transpose
    X = X.reshape(-1)
    N = X.numel()
    if T2 is None:
        T2 = dct2_matrix_ortho(N, device=X.device, dtype=X.dtype)
    return T2.t() @ X


def dct1_ortho(x, T1=None):
    # x: [N] float tensor. Returns DCT-I (orthonormal) [N].
    x = x.reshape(-1)
    N = x.numel()
    if T1 is None:
        T1 = dct1_matrix_ortho(N, device=x.device, dtype=x.dtype)
    return T1 @ x


def idct1_ortho(X, T1=None):
    # DCT-I orthonormal is self-inverse
    X = X.reshape(-1)
    N = X.numel()
    if T1 is None:
        T1 = dct1_matrix_ortho(N, device=X.device, dtype=X.dtype)
    return T1 @ X


def _complex_dtype_from_real(real_dtype):
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise TypeError("Only float32/float64 supported.")


def dct2_fft(x, dim=-1, norm="ortho"):
    """
    DCT-II via even-symmetric 2N extension and torch.fft.rfft.
    x: real tensor (..., N)
    Returns: real tensor (..., N)
    norm: 'ortho' (orthonormal, like scipy.fft.dct(..., type=2, norm='ortho')) or None (unnormalized).
    """
    if not torch.is_floating_point(x):
        raise TypeError("x must be float tensor")
    N = x.shape[dim]
    if N < 1:
        return x.clone()

    # Even extension [x, flip(x)]
    x_flip = torch.flip(x, dims=(dim,))
    s = torch.cat([x, x_flip], dim=dim)  # (..., 2N)

    # RFFT over length 2N
    S = torch.fft.rfft(s, n=2 * N, dim=dim)  # (..., N+1)

    # k = 0..N-1
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    # exp(-j*pi*k/(2N))
    ctype = _complex_dtype_from_real(x.dtype)
    twiddle = torch.exp(-1j * math.pi * k / (2.0 * N)).to(dtype=ctype, device=x.device)
    for _ in range(dim, S.dim() - 1):
        twiddle = twiddle.unsqueeze(-1)

    # Take real part; factor 1/2 (see derivation)
    C = (S.narrow(dim, 0, N) * twiddle).real * 0.5  # (..., N)

    if norm == "ortho":
        # Orthonormal scaling: sqrt(2/N) * beta(k), beta(0)=1/sqrt(2)
        C = C * math.sqrt(2.0 / N)
        index0 = [slice(None)] * C.dim()
        index0[dim] = 0
        C[tuple(index0)] /= math.sqrt(2.0)
    elif norm is None:
        pass
    else:
        raise ValueError("norm must be 'ortho' or None")
    return C


def idct2_fft(C, dim=-1, norm="ortho"):
    """
    Inverse of dct2_fft (i.e., DCT-III) using torch.fft.irfft.
    C: real tensor (..., N) with same norm used in dct2_fft.
    Returns real tensor (..., N).
    """
    if not torch.is_floating_point(C):
        raise TypeError("C must be float tensor")
    N = C.shape[dim]
    if N < 1:
        return C.clone()

    # Undo orthonormal scaling to get "unnormalized" DCT-II coefficients
    Cun = C
    if norm == "ortho":
        Cun = C / math.sqrt(2.0 / N)
        index0 = [slice(None)] * Cun.dim()
        index0[dim] = 0
        Cun = Cun.clone()
        Cun[tuple(index0)] *= math.sqrt(2.0)
    elif norm is None:
        Cun = C
    else:
        raise ValueError("norm must be 'ortho' or None")

    # Build unique half-spectrum (length N+1) for the 2N-length irfft
    # S[k] = 2*Cun[k] * exp(+j*pi*k/(2N)), for k=0..N-1
    k = torch.arange(N, device=C.device, dtype=C.dtype)
    ctype = _complex_dtype_from_real(C.dtype)
    twiddle = torch.exp(+1j * math.pi * k / (2.0 * N)).to(dtype=ctype, device=C.device)
    for _ in range(dim, C.dim() - 1):
        twiddle = twiddle.unsqueeze(-1)

    # Allocate (..., N+1)
    new_shape = list(Cun.shape)
    new_shape[dim] = N + 1
    S_half = torch.zeros(*new_shape, dtype=ctype, device=C.device)

    # Fill 0..N-1
    # real times complex -> cast below
    S_part = (2.0 * Cun) * twiddle.real - 0j
    S_part = (2.0 * Cun).to(ctype) * twiddle
    S_half.narrow(dim, 0, N).copy_(S_part)

    # Nyquist (k=N) is zero for the chosen even-symmetric extension
    indexN = [slice(None)] * S_half.dim()
    indexN[dim] = N
    S_half[tuple(indexN)] = 0

    # irfft to length 2N, take first N samples
    s = torch.fft.irfft(S_half, n=2 * N, dim=dim)  # (..., 2N)

    # Slice first N along dim
    x = s.narrow(dim, 0, N)
    return x


# --------- N-D (multi-axis) DCT-II / IDCT-II built from the 1D versions ---------
def _normalize_dims(dims, ndim):
    if isinstance(dims, int):
        dims = (dims,)
    dims = tuple(d if d >= 0 else d + ndim for d in dims)
    if any(d < 0 or d >= ndim for d in dims):
        raise ValueError("dims out of range for input tensor.")
    # You can enforce uniqueness if desired:
    if len(set(dims)) != len(dims):
        raise ValueError("dims must be unique.")
    return dims


def dct2_nd_fft(x, dims, norm="ortho"):
    """
    N-D DCT-II applied along the specified dimensions.
    x: real tensor
    dims: tuple of axes (e.g., (-2,-1) for 2D, (-3,-2,-1) for 3D)
    norm: 'ortho' or None
    """
    dims = _normalize_dims(dims, x.ndim)
    y = x
    for d in dims:
        y = dct2_fft(y, dim=d, norm=norm)
    return y


def idct2_nd_fft(X, dims, norm="ortho"):
    """
    N-D inverse of DCT-II (DCT-III) along the specified dimensions.
    """
    dims = _normalize_dims(dims, X.ndim)
    y = X
    for d in dims:
        y = idct2_fft(y, dim=d, norm=norm)
    return y


def _to_device_dtype(x, device, dtype):
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
    if dtype is None:
        dtype = torch.float64  # match MATLAB double
    return device, dtype


def _omega_grid_1d(N, shifted, device, dtype):
    # Digital radian frequency samples on FFT bins.
    # unshifted: ω_k = 2π k / N, k=0..N-1 (DC at index 0)
    # shifted: fftshift layout (DC at center), monotonically increasing from negative to positive
    k = torch.arange(N, device=device, dtype=dtype)
    w = 2.0 * math.pi * k / N
    # [0, 2π)
    if shifted:
        w = torch.fft.fftshift(w)  # center DC
    return w


def _tan_half_abs(w, eps=1e-12):
    # Safe |tan(w/2)| to avoid overflow at w=π.
    half = 0.5 * w
    c = torch.cos(half)
    s = torch.sin(half)
    # Where cos is near zero, use a very large value (approach infinity)
    # large but not inf to avoid NaNs downstream
    large = torch.finfo(w.dtype).max ** 0.5
    t = torch.where(c.abs() < eps, torch.sign(s) * large, s / c)
    return t.abs()


def butterworth_mask_1d(
    N,
    fc,
    order,
    btype="low",
    shifted=False,
    device=None,
    dtype=None,
):
    """
    1D Butterworth frequency mask equivalent to MATLAB butter+freqz magnitude.
    - N: number of FFT bins
    - fc: normalized cutoff(s) in cycles/sample (relative to 1 sample) with 0 < fc < 0.5
      low/high: scalar; bandpass/stop: [f1, f2] with 0 < f1 < f2 < 0.5
        * fc is equivalent to Wn / 2 in MATLAB's butter function. e.g. butter(4, 0.25) is equivalent to fc=0.125 here.
    - order: integer >= 1
    - btype: 'low', 'high', 'bandpass', 'stop'
    - shifted: if True, return mask in fftshift layout (DC at center)
    """
    assert isinstance(N, int) and N >= 2
    assert isinstance(order, int) and order >= 1
    btype = btype.lower()
    if btype in ("low", "high"):
        fc = float(fc)
        assert 0.0 < fc < 0.5
    else:
        assert len(fc) == 2
        f1, f2 = float(fc[0]), float(fc[1])
        assert 0.0 < f1 < f2 < 0.5
        fc = (f1, f2)

    device, dtype = _to_device_dtype(torch.empty(0), device, dtype)
    w = _omega_grid_1d(N, shifted=shifted, device=device, dtype=dtype)  # 0..2π (or centered)
    # Bilinear mapping (prewarped): Ω = 2 * tan(ω/2)
    Om = 2.0 * _tan_half_abs(w)  # analog rad/sec (normalized T=1)

    if btype == "low":
        # Prewarp analog cutoff: Ωc = 2*tan(π*fc)
        Oc = 2.0 * math.tan(math.pi * fc)
        ratio = (Om / Oc).clamp_min(0)
        mag = 1.0 / torch.sqrt(1.0 + ratio.pow(2 * order))
    elif btype == "high":
        Oc = 2.0 * math.tan(math.pi * fc)
        # Handle Om=0 => magnitude=0
        ratio = torch.where(Om > 0, (Oc / Om), torch.full_like(Om, float("inf")))
        mag = 1.0 / torch.sqrt(1.0 + ratio.pow(2 * order))
    elif btype == "bandpass":
        f1, f2 = fc
        O1 = 2.0 * math.tan(math.pi * f1)
        O2 = 2.0 * math.tan(math.pi * f2)
        B = O2 - O1
        O0 = math.sqrt(O1 * O2)
        # D(Ω) = (Ω^2 - Ω0^2)/(B*Ω)
        denom = B * Om
        # denom=0 at Om=0 -> D=inf, magnitude=0
        D = torch.where(denom != 0, (Om.pow(2) - O0**2) / denom, torch.full_like(Om, float("inf")))
        mag = 1.0 / torch.sqrt(1.0 + D.abs().pow(2 * order))
    elif btype in ("stop", "bandstop", "bandreject"):
        f1, f2 = fc
        O1 = 2.0 * math.tan(math.pi * f1)
        O2 = 2.0 * math.tan(math.pi * f2)
        B = O2 - O1
        O0 = math.sqrt(O1 * O2)
        # D(Ω) = (B*Ω)/(Ω^2 - Ω0^2)
        denom = Om.pow(2) - O0**2
        # denom=0 at Om=O0 -> D=inf, magnitude=0
        D = torch.where(denom != 0, (B * Om) / denom, torch.full_like(Om, float("inf")))
        mag = 1.0 / torch.sqrt(1.0 + D.abs().pow(2 * order))
    else:
        raise ValueError("btype must be 'low', 'high', 'bandpass', or 'stop'.")

    return mag.to(dtype=dtype, device=device)


def butterworth_mask_2d_separable(
    shape,
    fc,
    order,
    btype="low",
    shifted=False,
    device=None,
    dtype=None,
):
    """
    2D separable Butterworth mask (rows × cols), equivalent to applying 1D Butterworth along rows and columns (zero-phase). Not an isotropic circular Butterworth.
    - shape: (M, N)
    - fc: scalar or 2-tuple for low/high; for band types, pass 2-tuples for each axis: ([f1y,f2y], [f1x,f2x]) You can also pass scalar or 2-tuple to apply same cutoffs on both axes.
    - order: integer or 2-tuple for (order_y, order_x)
    - btype: 'low', 'high', 'bandpass', 'stop'
    - shifted: if True, both axes are centered (fftshift layout)
    """
    M, N = int(shape[0]), int(shape[1])
    assert M >= 2 and N >= 2
    device, dtype = _to_device_dtype(torch.empty(0), device, dtype)

    # Normalize fc/order to per-axis tuples
    if btype in ("low", "high"):
        if not isinstance(fc, (list, tuple)):
            fcy = fcx = fc
        else:
            assert len(fc) == 2
            fcy, fcx = fc
    else:
        # band types
        if isinstance(fc[0], (list, tuple)) and isinstance(fc[1], (list, tuple)):
            fcy, fcx = fc
        else:
            # same band on both axes
            fcy = fcx = fc

    if isinstance(order, (list, tuple)):
        oy, ox = int(order[0]), int(order[1])
    else:
        oy = ox = int(order)

    Hy = butterworth_mask_1d(M, fcy, oy, btype=btype, shifted=shifted, device=device, dtype=dtype)
    Hx = butterworth_mask_1d(N, fcx, ox, btype=btype, shifted=shifted, device=device, dtype=dtype)

    # Outer product to build separable 2D mask
    H2 = Hy.reshape(M, 1) * Hx.reshape(1, N)
    return H2


def _freqvec_norm(
    N: int,
    shifted: bool,
    device=None,
    dtype=None,
):
    """
    Normalized frequency vector in [-0.5, 0.5), length N.
    - shifted=False: DC at index 0 (unshifted FFT layout)
    - shifted=True: DC at center (fftshift layout)
    """
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float64
    k = torch.arange(N, device=device, dtype=dtype)
    if shifted:
        f = (k - torch.floor(torch.tensor(N / 2, dtype=dtype, device=device))) / N
    else:
        f = k / N
    f = torch.where(f >= 0.5, f - 1.0, f)  # wrap into [-0.5, 0.5)
    return f  # [N]


def _radial_frequency_nd(
    shape: Sequence[int],
    shifted: bool,
    device=None,
    dtype=None,
):
    """
    Radial normalized frequency R in [-0.5,0.5) computed over all axes.
    Returns R with shape 'shape'.
    """
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float64
    grids = [_freqvec_norm(N, shifted=shifted, device=device, dtype=dtype) for N in shape]
    # list of tensors, each shape = shape
    meshes = torch.meshgrid(*grids, indexing="ij")
    R2 = torch.zeros(shape, dtype=dtype, device=device)
    for g in meshes:
        R2 = R2 + g**2
    R = torch.sqrt(R2)
    return R


def butterworth_nd(
    shape: Sequence[int],
    cutoff: Union[float, Tuple[float, float]],
    order: int,
    btype: str = "low",
    shifted: bool = False,
    device=None,
    dtype=None,
):
    """Isotropic N-D Butterworth mask (low/high/bandpass/bandstop).
    Args:
    shape: iterable of ints, e.g., (H, W) or (D, H, W) ...
    cutoff:
        - 'low'/'high': scalar D0 in (0, 0.5]
        - 'bandpass'/'bandstop': tuple (D1, D2) with 0 < D1 < D2 <= 0.5
    order: integer >= 1
    btype: 'low' | 'high' | 'bandpass' | 'bandstop' (alias 'stop')
    shifted: if True, mask is centered (fftshift layout); else unshifted
    device, dtype: optional torch device/dtype (defaults: CPU, float64)

    Returns:
    H: tensor with shape 'shape', values in [0, 1].
    """
    assert len(shape) >= 1 and all(int(s) >= 1 for s in shape), "Invalid shape."
    order = int(order)
    assert order >= 1, "order must be >= 1"
    btype = btype.lower()
    if btype in ("low", "high"):
        D0 = float(cutoff)
        # assert 0.0 < D0 <= 0.5, "cutoff must be in (0, 0.5]"
    else:
        D1, D2 = float(cutoff[0]), float(cutoff[1])
        # assert 0.0 < D1 < D2 <= 0.5, "for band types: 0 < D1 < D2 <= 0.5"
        B = D2 - D1
        D0 = math.sqrt(D1 * D2)

    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float64

    R = _radial_frequency_nd(
        tuple(int(s) for s in shape), shifted=shifted, device=device, dtype=dtype
    )
    eps = torch.finfo(dtype).eps
    # print(f"{R = }")

    if btype == "low":
        # H = 1 / (1 + (R/D0)^(2n))
        ratio = (R / D0).clamp_min(0)
        H = 1.0 / (1.0 + ratio.pow(2 * order))

    elif btype == "high":
        # H = 1 / (1 + (D0/R)^(2n)), H(DC)=0
        # avoid divide-by-zero at R=0
        safe_R = torch.where(R > 0, R, torch.tensor(1.0, device=device, dtype=dtype))  # dummy
        ratio = D0 / safe_R
        H = 1.0 / (1.0 + ratio.pow(2 * order))
        # enforce DC = 0
        H = torch.where(R > 0, H, torch.zeros_like(H))

    elif btype == "bandpass":
        # D = (R^2 - D0^2) / (B*R); H = 1 / (1 + |D|^(2n))
        # Handle R=0 -> D=inf -> H=0
        denom = B * R
        D = torch.where(denom != 0, (R.pow(2) - D0**2) / denom, torch.full_like(R, float("inf")))
        H = 1.0 / (1.0 + D.abs().pow(2 * order))

    elif btype in ("bandstop", "stop", "bandreject"):
        # D = (B*R) / (R^2 - D0^2); H = 1 / (1 + |D|^(2n))
        # Handle R^2 - D0^2 = 0 -> D=inf -> H=0 (deep notch at R=D0)
        denom = R.pow(2) - D0**2
        D = torch.where(denom != 0, (B * R) / denom, torch.full_like(R, float("inf")))
        H = 1.0 / (1.0 + D.abs().pow(2 * order))

    else:
        raise ValueError("btype must be 'low', 'high', 'bandpass', or 'bandstop'.")

    return H


def butterworth_low_pass_filter(
    tensor: torch.Tensor,
    dims: Sequence[int],
    cutoff: float,
    order: int,
    shifted: bool = False,
    device=None,
    dtype=None,
):
    """
    Applies a Butterworth low-pass filter to the input tensor.

    the dims specify which dim should be perform filtering

    return filtered tensor
    """
    if not isinstance(dims, (list, tuple)):
        dims = (dims,)
    ndims_total = tensor.ndim
    # Normalize dims (handle negatives)
    norm_dims = _normalize_dims(dims, ndim=ndims_total)

    original_dtype = tensor.dtype
    work_dtype = dtype or (tensor.dtype if torch.is_floating_point(tensor) else torch.float32)
    if work_dtype == torch.bfloat16 or work_dtype == torch.float16:
        work_dtype = torch.float32
    device = device or tensor.device

    # Prepare frequency-domain representation
    x = tensor.to(device=device, dtype=work_dtype)
    X = torch.fft.fftn(x, dim=norm_dims)
    if shifted:
        X = torch.fft.fftshift(X, dim=norm_dims)

    # Build isotropic Butterworth mask over the selected dims
    shape_subset = [x.shape[d] for d in norm_dims]
    H_small = butterworth_nd(
        shape=shape_subset,
        cutoff=cutoff,
        order=order,
        btype="low",
        shifted=shifted,
        device=device,
        dtype=work_dtype,
    )

    # Broadcast mask into full tensor shape
    mask_shape = [1] * ndims_total
    for i, d in enumerate(norm_dims):
        mask_shape[d] = shape_subset[i]
    H = H_small.view(*mask_shape)

    # Apply mask
    X_filtered = X * H

    # Inverse FFT
    if shifted:
        X_filtered = torch.fft.ifftshift(X_filtered, dim=norm_dims)
    x_filtered = torch.fft.ifftn(X_filtered, dim=norm_dims).real

    return x_filtered.to(dtype=original_dtype)


# def fft_denoise(tensor, dim, fft_ratio):
#     assert len(dim) == 2
#     original_dtype = tensor.dtype
#     tensor = tensor.to(torch.float32)
#     # Create low pass filter
#     LPF = butterworth_low_pass_filter(
#         (tensor.shape[dim[0]], tensor.shape[dim[1]]),
#         n=4,
#         d_s=fft_ratio,
#     )
#     LPF = LPF.to(dtype=tensor.dtype, device=tensor.device)
#     # print(f"{LPF = }")
#     # print(f"{LPF.shape = }")
#     for _ in range(dim[0]):
#         LPF = LPF.unsqueeze(0)
#     for _ in range(dim[1] + 1, len(tensor.shape)):
#         LPF = LPF.unsqueeze(-1)
#     # print(f"{LPF.shape = }")
#     # FFT
#     latents_freq_k = torch.fft.fftn(tensor, dim=dim)
#     # print(f"{latents_freq_k.shape = }")
#     latents_freq_k = torch.fft.fftshift(latents_freq_k, dim=dim)
#     # print(f"{latents_freq_k.shape = }")

#     new_freq_k = latents_freq_k * LPF

#     # IFFT
#     new_freq_k = torch.fft.ifftshift(new_freq_k, dim=dim)
#     denoised_k = torch.fft.ifftn(new_freq_k, dim=dim).real
#     denoised_k = denoised_k.to(original_dtype)
#     return denoised_k


if __name__ == "__main__":
    # x = torch.linspace(0, 2 * np.pi, 8)
    # y = torch.linspace(0, 2 * np.pi, 8)
    # X, Y = torch.meshgrid(x, y, indexing='ij')
    # latents = (
    #     torch.sin(2 * X + Y) +
    #     torch.sin(X + 3 * Y) +
    #     torch.sin(3 * X - 2 * Y)
    # ) + 1
    # latents += 0.01 * torch.randn_like(latents)  # Add Gaussian noise
    # # latents = torch.randn([8, 8])
    # print(f"latents = \n{latents}")

    # latents_freq = torch.fft.fftn(latents, dim=(-2, -1))
    # print(f"latents_freq = \n{torch.abs(latents_freq)}")

    # latents_freq_shift = torch.fft.fftshift(latents_freq, dim=(-2, -1))
    # print(f"latents_freq_shift = \n{torch.abs(latents_freq_shift)}")

    # latents_freq_dct = dct_2d(latents)
    # print(f"latents_freq_dct = \n{latents_freq_dct}")

    # LPF_1 = butterworth_low_pass_filter(latents=latents, d_s=-1.0)

    # print(f"LPF_1 = \n{LPF_1}")

    # LPF_2 = my_butterworth_low_pass_filter_non_center(
    #     shape=latents.shape,
    #     stop_freqs=[0.25, 0.25],
    #     n=4,
    # )
    # print(f"LPF_2 = \n{LPF_2}")

    # LPF_3 = my_butterworth_low_pass_filter(
    #     shape=latents.shape,
    #     stop_freqs=[0.25, 0.25],
    #     n=4,
    # )
    # print(f"LPF_3 = \n{LPF_3}")

    # img = load_grayscale_image()
    # # Extract middle column as 1-D signal
    # col = img.shape[1] // 2 - 1
    # print(f"{col = }")
    # x_np = img[:, col].astype(np.float32)  # [H]
    # # print(f"{x_np = }")
    # N = x_np.shape[0]
    # print(f"{N = }")

    # device = 'cpu'
    # dtype = torch.float64

    # x = torch.from_numpy(img).to(device=device, dtype=dtype)
    # print(f"{x = }")

    # # Transforms
    # Xf = torch.fft.fftn(x, dim=(-3, -2, -1), norm=None)            # complex64
    # print(f"{Xf = }")
    # x_reconstructed = torch.fft.ifftn(Xf, dim=(-3, -2, -1), norm=None)
    # print(f"{x_reconstructed = }")
    # print(f"{(x - x_reconstructed).abs().max() = }")

    # Xd2 = dct2_nd_fft(x, dims=(-3, -2, -1), norm="ortho")       # float
    # print(f"{Xd2 = }")
    # x_reconstructed = idct2_nd_fft(Xd2, dims=(-1, -2, -3), norm="ortho")
    # print(f"{x_reconstructed = }")
    # print(f"{(x - x_reconstructed).abs().max() = }")

    # H1 = butterworth_mask_1d(16, 0.125, 4, btype='low', shifted=True)
    # print(f"{H1 = }")

    H2 = butterworth_nd([30, 52], 1.0, 4, btype="low", shifted=True)
    print(f"{H2 = }")

    # ---- Planar wave demo with Butterworth low-pass filtering ----
    def demo_planar_wave():
        # Generate 2D planar wave: low-frequency + added high-frequency component
        H, W = 128, 128
        device = "cpu"
        y = torch.arange(H, device=device).view(H, 1)
        x = torch.arange(W, device=device).view(1, W)

        # Low-frequency component
        kx_low, ky_low = 2, 3
        low = torch.sin(2 * math.pi * (kx_low * x / W + ky_low * y / H))

        # High-frequency component
        kx_high, ky_high = 20, 24
        high = 0.5 * torch.sin(2 * math.pi * (kx_high * x / W + ky_high * y / H))

        signal = low + high

        # Apply Butterworth low-pass (cutoff chosen to keep low freq, attenuate high freq)
        cutoff = 0.12  # normalized radial cutoff (<=0.5)
        order = 4
        filtered = butterworth_low_pass_filter(
            signal, dims=(-2, -1), cutoff=cutoff, order=order, shifted=True
        )

        # Metrics
        mse_before = (signal - low).pow(2).mean()
        mse_after = (filtered - low).pow(2).mean()
        residual_energy_ratio = (filtered - low).pow(2).sum() / (signal - low).pow(2).sum()

        print("Planar wave demo:")
        print(f"mse_before={mse_before.item():.6e}")
        print(f"mse_after ={mse_after.item():.6e}")
        print(f"residual_energy_ratio={residual_energy_ratio.item():.4%}")
        # Quick sanity: high frequency suppression (should be << 1)
        assert (
            mse_after < mse_before
        ), "Filtering did not reduce error to low-frequency ground truth."

    demo_planar_wave()
