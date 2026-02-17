import numpy as np
from PIL import Image, ImageDraw, ImageColor
import scipy
import cv2
import torch
import kornia
import torch.nn.functional as F


def image_to_pil(image):
    """Convert a numpy array to a PIL Image."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise ValueError("Unsupported image type")


def image_to_np(image):
    """Convert a numpy array to a PIL Image."""
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise ValueError("Unsupported image type")


def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) // 2)
    center_y = int((y1 + y2) // 2)
    return (center_x, center_y)


def save_mask_to_file(
    mask,
    file_path,
):
    mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask *= 255
    elif mask.max() > 1:
        pass
    Image.fromarray(mask).save(file_path)


def read_mask_from_file(
    file_path,
):
    mask = Image.open(file_path).convert("L")
    mask = image_to_np(mask)
    return mask > 0


def bbox_from_mask(
    mask: np.ndarray | Image.Image | torch.Tensor,
):
    """
    Compute axis-aligned bounding box for a mask (numpy array, PIL.Image, or torch.Tensor).

    Returns:
        (min_x, min_y, max_x, max_y)  (inclusive coordinates)
        or None if mask has no positive / True pixels.

    Rules:
        - Non-zero (or True) pixels are foreground.
        - Supports 2D or (H,W,1) masks directly.
        - For multi-channel masks (H,W,C), foreground = any channel > 0.
        - For torch tensors, stays on device for reduction (fast), then moves only indices to CPU.
    """
    # Convert PIL to numpy
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Torch path
    if isinstance(mask, torch.Tensor):
        m = mask
        # Ensure at least 2D
        if m.ndim < 2:
            return None
        # If more than 2D, collapse channels/extra dims via any() over non-spatial dims
        # Assume last two dims are (H,W)
        if m.ndim > 2:
            # Move all non-spatial dims to a single dim then reduce
            # Example shapes:
            #   (H,W,1) -> squeeze
            #   (C,H,W) -> any over C
            #   (B,1,H,W) -> any over B & channel
            # Strategy: bring H,W to end and flatten others.
            # Easier: identify H,W as last two dims.
            spatial_h, spatial_w = m.shape[-2], m.shape[-1]
            if m.shape[:-2] != ():
                m = (m != 0).any(dim=tuple(range(0, m.ndim - 2)))
            m = m.to(torch.bool)
        else:
            m = m != 0

        if m.dtype != torch.bool:
            m = m != 0

        if not m.any():
            return None

        # Find rows / cols with any foreground
        rows = torch.any(m, dim=1)
        cols = torch.any(m, dim=0)
        y_idx = torch.nonzero(rows, as_tuple=False).squeeze(1)
        x_idx = torch.nonzero(cols, as_tuple=False).squeeze(1)
        y_min = int(y_idx[0].item())
        y_max = int(y_idx[-1].item())
        x_min = int(x_idx[0].item())
        x_max = int(x_idx[-1].item())
        return (x_min, y_min, x_max, y_max)

    # Numpy path
    mask_np = np.asarray(mask)
    if mask_np.ndim < 2:
        return None
    # Handle channels
    if mask_np.ndim == 3:
        if mask_np.shape[2] == 1:
            mask_np = mask_np[..., 0]
        else:
            mask_np = np.any(mask_np != 0, axis=2)

    fg = mask_np != 0
    if not fg.any():
        return None
    y_indices, x_indices = np.where(fg)
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    return (x_min, y_min, x_max, y_max)


def remove_small_components(mask, min_size=10):
    labeled, nlabels = scipy.ndimage.label(mask)
    for idx in range(1, nlabels + 1):
        if np.sum(labeled == idx) < min_size:
            labeled[labeled == idx] = 0
    return (labeled > 0).astype(np.uint8) * 255


def draw_bbox_on_image(
    image: np.ndarray | Image.Image,
    bbox,
    color="yellow",
    width=3,
):
    """Draw a bounding box on an image."""
    if image is None or bbox is None:
        return image
    image = image.copy()
    image = image_to_pil(image)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=width,
    )
    return image


def draw_mask_on_image(
    image: np.ndarray | Image.Image | None,
    mask: np.ndarray | Image.Image | None,
    mask_color: str | list[int] | tuple[int, int, int] = [30, 255, 144],
    alpha: float = 0.3,
):
    """
    Draw a binary mask overlay on an image.

    mask_color can be:
      - string (e.g. "red", "#ff0000", "#f00")
      - list/tuple/np.ndarray of 3 ints/floats in 0..255 (R,G,B)

    alpha: 0..1 overlay opacity.
    """
    if image is None or mask is None:
        return image

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")

    # Normalize mask_color to (R,G,B)
    if isinstance(mask_color, str):
        rgb = ImageColor.getrgb(mask_color)
    elif isinstance(mask_color, (list, tuple, np.ndarray)):
        if len(mask_color) != 3:
            raise ValueError("mask_color list/tuple must have length 3")
        rgb = tuple(int(round(float(c))) for c in mask_color)
    else:
        raise ValueError("Unsupported mask_color type")

    rgb = tuple(np.clip(rgb, 0, 255))

    image = image.copy()
    image = image_to_pil(image)
    mask = image_to_np(mask)

    # Binarize mask
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.ndim != 2:
        raise ValueError("mask must be 2D after binarization")

    h, w = mask_bin.shape
    # Build RGBA overlay
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = rgb[0]
    overlay[..., 1] = rgb[1]
    overlay[..., 2] = rgb[2]
    overlay[..., 3] = (
        (alpha * 255).astype(np.uint8) if isinstance(alpha, np.ndarray) else int(alpha * 255)
    )

    # Zero alpha where mask is 0
    overlay[mask_bin == 0, 3] = 0

    masked_image = Image.alpha_composite(
        image.convert("RGBA"),
        Image.fromarray(overlay),
    )
    return masked_image


def draw_mask_bbox_on_image(
    image,
    mask,
    mask_color: list[int] = [30, 255, 144],
    mask_alpha: float = 0.3,
    bbox_color="yellow",
    bbox_width=3,
):
    """Draw a mask and its bounding box on an image."""
    image = draw_mask_on_image(
        image,
        mask,
        mask_color=mask_color,
        alpha=mask_alpha,
    )
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return image, None
    image = draw_bbox_on_image(
        image,
        bbox,
        color=bbox_color,
        width=bbox_width,
    )
    return image, bbox


def draw_points_on_image(
    image,
    points: list[tuple],
    color="red",
    radius=5,
):
    image = image.copy()
    """Draw points on an image."""
    assert isinstance(points, list), "points must be a list of tuples"
    # if color is not a list, change it to a list with length of points
    if not isinstance(color, list):
        color = [color] * len(points)
    assert len(color) == len(points), "color must be a list of the same length as points"
    # if radius is not a list, change it to a list with length of points
    if not isinstance(radius, list):
        radius = [radius] * len(points)
    assert len(radius) == len(points), "radius must be a list of the same length as points"
    image = image_to_pil(image)
    draw = ImageDraw.Draw(image)

    # draw points, colors, and radius on the image
    for point, color, r in zip(points, color, radius):
        x, y = point
        draw.circle(
            (x, y),
            radius=r,
            fill=color,
            outline=color,
        )
    return image


def draw_lines_on_image(
    image,
    points: list[tuple],
    color="red",
    width=3,
):
    """
    Draw polyline on image.
    color can be:
      - single name / "#rrggbb" / "rrggbb"
      - list of such specs (length == len(points)-1)
    """
    if image is None:
        return image
    if not isinstance(points, list) or len(points) < 2:
        return image
    image = image.copy()
    image = image_to_pil(image)

    # Normalize color list
    if not isinstance(color, list):
        color_list = [color] * (len(points) - 1)
    else:
        if len(color) == len(points):
            color_list = color[:-1]
        else:
            color_list = color
        if len(color_list) != len(points) - 1:
            raise ValueError("color list length must be len(points)-1 or len(points)")

    def normalize(c):
        if isinstance(c, str):
            c = c.strip()
            if len(c) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in c):
                c = "#" + c
            return ImageColor.getrgb(c)
        return c  # assume tuple

    color_list = [normalize(c) for c in color_list]

    draw = ImageDraw.Draw(image)
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color_list[i], width=width)
    return image


def draw_arrow_on_image(
    image,
    start_point: tuple,
    end_point: tuple,
    color: str = "white",
    thickness: int = 5,
):
    image = image.copy()
    na = np.array(image)

    # Draw arrowed line, from start_point to end_point in color with thickness
    na = cv2.arrowedLine(na, start_point, end_point, color, thickness)

    return Image.fromarray(na)


def trajectory_interpolate_1d(
    trajectory: list[float],
    scale: int,
) -> list[float]:
    """
    Interpolate a 1D trajectory to a fixed number of points.

    Args:
        trajectory (List[float]): Sequence of scalar values (len >= 2).
        scale (int): Number of interpolated steps between original samples.

    Returns:
        List[float]: Interpolated 1D trajectory of length (L-1)*scale + 1.
    """
    assert isinstance(trajectory, list), "trajectory must be a list"
    assert len(trajectory) > 1, "trajectory must have at least 2 points"
    assert isinstance(scale, int), "scale must be an integer"
    assert scale > 0, "scale must be greater than 0"

    traj_np = np.asarray(trajectory, dtype=np.float32).reshape(-1)
    L = traj_np.shape[0]

    x = np.arange(L, dtype=np.float32)
    x_new = np.linspace(0, L - 1, (L - 1) * scale + 1, dtype=np.float32)
    y_new = np.interp(x_new, x, traj_np)  # linear 1D interpolation

    return y_new.tolist()


def trajectory_interpolate(
    trajectory: list[tuple],
    scale: int,
):
    """Interpolate a trajectory to a fixed number of points."""
    assert isinstance(trajectory, list), "trajectory must be a list of tuples"
    assert len(trajectory) > 1, "trajectory must have at least 2 points"
    assert isinstance(scale, int), "scale must be an integer"
    assert scale > 0, "scale must be greater than 0"

    original_trajectory_length = len(trajectory)

    # Convert trajectory to numpy array
    trajectory_np = np.array(trajectory)
    # print(f"{trajectory_np = }")

    trajectory_torch = torch.tensor(trajectory_np, dtype=torch.float32)

    trajectory_torch_interpolated = torch.nn.functional.interpolate(
        trajectory_torch.unsqueeze(0).unsqueeze(0),
        size=((original_trajectory_length - 1) * scale + 1, 2),
        mode="bilinear",
        align_corners=True,
    ).squeeze()

    # print(f"{trajectory_torch_interpolated = }")

    interpolated_trajectory = []
    for i in range(trajectory_torch_interpolated.shape[0]):
        x = int(trajectory_torch_interpolated[i, 0].item())
        y = int(trajectory_torch_interpolated[i, 1].item())
        interpolated_trajectory.append((x, y))

    # Return the interpolated trajectory
    return interpolated_trajectory


def dilate_mask(
    mask: np.ndarray | None,
    dilate_factor: int = 15,
):
    if mask is None:
        return None
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1)
    return mask


def dilate_masks(
    masks: list[np.ndarray],
    dilate_factor: int = 15,
):
    return [dilate_mask(mask, dilate_factor) for mask in masks]


def shift_masks(
    ref_mask,
    deltas: list[tuple[float, float]],
):
    ref_mask_indices = np.where(ref_mask > 0)
    # print(f"{ref_mask_indices = }")

    shifted_masks_indices = [
        (
            ref_mask_indices[0] + int(delta[0]),
            ref_mask_indices[1] + int(delta[1]),
        )
        for delta in deltas
    ]
    # print(f"{shifted_masks_indices = }")

    # filter out-of-bounds indices
    shifted_masks_indices = [
        (
            np.clip(shifted_mask_indexs[0], 0, ref_mask.shape[0] - 1),
            np.clip(shifted_mask_indexs[1], 0, ref_mask.shape[1] - 1),
        )
        for shifted_mask_indexs in shifted_masks_indices
    ]

    shifted_masks = []
    for i, shifted_mask_indexs in enumerate(shifted_masks_indices):
        shifted_mask = np.zeros_like(ref_mask, dtype=np.uint8)
        # shifted_mask_indexs = (
        #     np.clip(shifted_mask_indexs[0], 0, ref_mask.shape[0] - 1),
        #     np.clip(shifted_mask_indexs[1], 0, ref_mask.shape[1] - 1)
        # )
        shifted_mask[shifted_mask_indexs] = 1
        shifted_masks.append(shifted_mask)

    # for i, shifted_mask in enumerate(shifted_masks):
    #     Image.fromarray(shifted_mask * 255).save(f"shifted_mask_{i}.png")

    return shifted_masks, shifted_masks_indices


def rotate_points(points, angle, center=(0.0, 0.0), degrees=True):
    """
    Rotate 2D point(s) around a center by angle.

    points: array-like of shape (2,) or (N, 2) as [x, y]
    angle: rotation angle (degrees by default)
    center: rotation center [cx, cy]
    degrees: if True, angle is in degrees; otherwise radians
    """
    pts = np.asarray(points, dtype=float)
    ctr = np.asarray(center, dtype=float)

    theta = np.deg2rad(angle) if degrees else angle
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    shifted = pts - ctr
    rotated = shifted @ R.T
    return rotated + ctr


def calculate_angle(vector_1: torch.Tensor, vector_2: torch.Tensor):
    dot_product = torch.dot(vector_1, vector_2)
    magnitude_1 = torch.norm(vector_1)
    magnitude_2 = torch.norm(vector_2)

    if magnitude_1 == 0 or magnitude_2 == 0:
        raise ValueError("One of the vectors has zero magnitude, cannot calculate angle.")

    cos_theta = dot_product / (magnitude_1 * magnitude_2)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angle_rad = torch.acos(cos_theta)
    angle_deg = torch.rad2deg(angle_rad)

    cross_product = vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]
    if cross_product < 0:
        angle_deg = -angle_deg

    return angle_deg


def calculate_angle_from_points(
    center_points: torch.Tensor,
    handle_points: torch.Tensor,
    target_points: torch.Tensor,
):
    """
    center_points (x, y)
    """
    center_points = torch.Tensor(center_points)
    handle_points = torch.Tensor(handle_points)
    target_points = torch.Tensor(target_points)

    v1 = handle_points - center_points
    v2 = target_points - center_points
    return calculate_angle(v1, v2)


def tensor_2d_translation(
    tensor: torch.Tensor,
    translation: tuple[float, float] | torch.Tensor,
    mode: str = "bilinear",
):
    """
    Translate a 2D tensor by a given translation vector.
    Always performs the operation in float32 and casts back to the original tensor dtype.
    """
    # Record original dtype (before any conversion)
    original_dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else torch.float32

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    # Convert to float32 for kornia
    tensor = tensor.to(torch.float32)
    origin_shape = tensor.shape
    if len(origin_shape) == 2:
        tensor = tensor[None, None, ...]
    elif len(origin_shape) == 3:
        tensor = tensor[None, ...]

    if not isinstance(translation, torch.Tensor):
        translation = torch.tensor(translation, device=tensor.device)
    translation = translation.to(dtype=torch.float32, device=tensor.device)
    if translation.ndim == 1:
        translation = translation.unsqueeze(0)

    translated_tensor = kornia.geometry.transform.translate(
        tensor,
        translation=translation,
        mode=mode,
    )

    if len(origin_shape) == 2:
        translated_tensor = translated_tensor[0, 0, ...]
    elif len(origin_shape) == 3:
        translated_tensor = translated_tensor[0, ...]

    # Cast back to original dtype
    translated_tensor = translated_tensor.to(original_dtype)
    return translated_tensor


def tensor_2d_rotation(
    tensor: torch.Tensor,
    angle: float,
    center=None,
    mode: str = "bilinear",
):
    """
    Rotate a 2D tensor by a given angle (clockwise).
    Performs computations in float32; casts result back to original tensor dtype.
    angle and center are also promoted to float32 internally.
    """
    # Record original dtypes
    tensor_original_dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else torch.float32
    angle_original_dtype = angle.dtype if isinstance(angle, torch.Tensor) else None
    center_original_dtype = (
        (center.dtype if isinstance(center, torch.Tensor) else None) if center is not None else None
    )

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    tensor = tensor.to(torch.float32)

    origin_shape = tensor.shape
    if len(origin_shape) == 2:
        tensor = tensor[None, None, ...]
    elif len(origin_shape) == 3:
        tensor = tensor[None, ...]

    # Clockwise -> negate
    angle = -angle
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, device=tensor.device)
    angle = angle.to(dtype=torch.float32, device=tensor.device)
    if angle.ndim == 0:
        angle = angle.unsqueeze(0)

    if center is not None:
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, device=tensor.device)
        center = center.to(dtype=torch.float32, device=tensor.device)

    rotated_tensor = kornia.geometry.transform.rotate(
        tensor,
        angle,
        center=center,
        mode=mode,
    )

    if len(origin_shape) == 2:
        rotated_tensor = rotated_tensor[0, 0, ...]
    elif len(origin_shape) == 3:
        rotated_tensor = rotated_tensor[0, ...]

    # Cast result back
    rotated_tensor = rotated_tensor.to(tensor_original_dtype)
    return rotated_tensor


def resize_tensor(
    tensor: torch.Tensor,
    size: int | tuple[int, int] = None,
    scale_factor: float | tuple[float, float] = None,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize a 2D tensor to a given size.

    Args:
        tensor (torch.Tensor): The input tensor to be resized.
        size (Union[int, Tuple[int, int]]): The target size. If an int is provided, it will be used for both dimensions.
        scale_factor (Union[float, Tuple[float, float]]): The scale factor for resizing. If provided, it will override the size argument.

    Returns:
        torch.Tensor: The resized tensor.
    """
    # if not isinstance(tensor, torch.Tensor):
    #     tensor = torch.tensor(tensor, dtype=torch.float32)
    origin_shape = tensor.shape
    if len(origin_shape) == 2:
        tensor = tensor[None, None, ...]
    elif len(origin_shape) == 3:
        tensor = tensor[None, ...]

    resized_tensor = F.interpolate(
        tensor,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=(True if mode in ["linear", "bilinear", "bicubic", "trilinear"] else None),
    )

    if len(origin_shape) == 2:
        resized_tensor = resized_tensor[0, 0, ...]
    elif len(origin_shape) == 3:
        resized_tensor = resized_tensor[0, ...]

    return resized_tensor


def warp_tensor(
    tensor: torch.Tensor,
    is_rotation: bool,
    delta,
    rotation_center: tuple[float, float] | torch.Tensor | None = None,
    original_height: int | None = None,
    mode: str = "nearest",
) -> torch.Tensor:
    """
    Warp a tensor by translation or rotation based on a trajectory step.

    Args:
        tensor: Tensor to warp. Can be (H, W), (C, H, W), or (B, C, H, W).
        is_rotation: If True, warp by rotation; otherwise by translation.
        delta: The delta for this step. For rotation: scalar angle (degrees).
               For translation: (dx, dy) in original image pixel coordinates.
               Can be a torch.Tensor, tuple, list, or scalar.
        rotation_center: (x, y) center of rotation in original image pixel coordinates.
                         Required when is_rotation is True.
        original_height: The height of the original image at which delta was computed.
                         If provided and differs from tensor's spatial height, delta and
                         rotation_center are rescaled accordingly.
                         If None, no rescaling is applied.
        mode: Interpolation mode for warping.

    Returns:
        Warped tensor with the same shape as input.
    """
    tensor_height = tensor.shape[-2]

    if original_height is not None and original_height != tensor_height:
        scale = original_height / tensor_height
    else:
        scale = 1.0

    if is_rotation:
        if rotation_center is None:
            raise ValueError("rotation_center is required when is_rotation is True")
        if not isinstance(rotation_center, torch.Tensor):
            rotation_center = torch.tensor(
                rotation_center, dtype=tensor.dtype, device=tensor.device
            )
        center = rotation_center.to(dtype=tensor.dtype, device=tensor.device) / scale
        return tensor_2d_rotation(tensor, angle=delta, center=center, mode=mode)
    else:
        # delta can be a tuple/list/tensor; tensor_2d_translation handles conversion
        if isinstance(delta, torch.Tensor):
            return tensor_2d_translation(tensor, translation=delta / scale, mode=mode)
        else:
            # For tuple/list/scalar, scale manually before passing
            delta_scaled = tuple(d / scale for d in delta)
            return tensor_2d_translation(tensor, translation=delta_scaled, mode=mode)


def warp_tensor_sequence(
    tensor: torch.Tensor,
    is_rotation: bool,
    deltas: list,
    rotation_center: tuple[float, float] | torch.Tensor | None = None,
    original_height: int | None = None,
    mode: str = "nearest",
    cumulative: bool = False,
) -> list[torch.Tensor]:
    """
    Warp a tensor by a sequence of deltas, returning a list of warped tensors.

    Args:
        tensor: Tensor to warp. Can be (H, W), (C, H, W), or (B, C, H, W).
        is_rotation: If True, warp by rotation; otherwise by translation.
        deltas: List of deltas for each step. For rotation: each is a scalar angle (degrees).
                For translation: each is (dx, dy) in original image pixel coordinates.
                Each delta can be a torch.Tensor, tuple, list, or scalar.
        rotation_center: (x, y) center of rotation in original image pixel coordinates.
                         Required when is_rotation is True.
        original_height: The height of the original image at which deltas were computed.
                         If provided and differs from tensor's spatial height, deltas and
                         rotation_center are rescaled accordingly.
                         If None, no rescaling is applied.
        mode: Interpolation mode for warping.
        cumulative: If True, each warp is applied on top of the previous result
                    (i.e. sequential composition). If False, each delta is applied
                    independently to the original tensor.

    Returns:
        List of warped tensors, one per delta, each with the same shape as input.
    """
    warped_tensors = []
    current = tensor
    for delta in deltas:
        source = current if cumulative else tensor
        warped = warp_tensor(
            source,
            is_rotation=is_rotation,
            delta=delta,
            rotation_center=rotation_center,
            original_height=original_height,
            mode=mode,
        )
        warped_tensors.append(warped)
        if cumulative:
            current = warped
    return warped_tensors


def combine_masks_or(
    masks: list[torch.Tensor | np.ndarray],
) -> torch.Tensor | np.ndarray:
    """
    Combine a list of binary masks using logical OR (union).

    Each mask is assumed to be a 2D tensor/array with values in [0, 1].
    The result is clamped to [0, 1].

    Returns a tensor if any input is a tensor, otherwise a numpy array.
    """
    if len(masks) == 0:
        raise ValueError("masks list is empty")

    result = masks[0].clone() if isinstance(masks[0], torch.Tensor) else masks[0].copy()
    for m in masks[1:]:
        result = result + m

    if isinstance(result, torch.Tensor):
        result = torch.clamp(result, 0, 1)
    else:
        result = np.clip(result, 0, 1)

    return result


def record_tensor_statics(
    tensor: torch.Tensor,
    axis=None,
    keepdim=False,
):

    mean = tensor.detach().mean(axis, keepdim=keepdim)
    std = tensor.detach().std(axis, keepdim=keepdim)
    tensor_max = tensor.detach().amax(axis, keepdim=keepdim)
    tensor_min = tensor.detach().amin(axis, keepdim=keepdim)

    return mean, std, tensor_max, tensor_min


def normalize_tensor(
    tensor,
    dim,
    target_mean,
    target_std,
):
    """
    Normalize a tensor along a specified dimension.
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)

    assert mean.shape == target_mean.shape == std.shape == target_std.shape

    new_tensor = (tensor - mean) / std
    new_tensor = new_tensor * target_std + target_mean
    return new_tensor


def normalize_tensor_to_match_tensor(
    target_tensor,
    dim,
    reference_tensor,
):
    reference_mean, reference_std, reference_max, reference_min = record_tensor_statics(
        reference_tensor,
        axis=dim,
        keepdim=True,
    )

    return normalize_tensor(
        target_tensor,
        dim=dim,
        target_mean=reference_mean,
        target_std=reference_std,
    )


def build_gaussian_focus_map(
    h: int,
    w: int,
    center_y: float,
    center_x: float,
    radius: float,
    sigma: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a (h, w) gaussian focus map:
        - Inside circle (dist <= r): weight = 1
        - Outside: weight = exp(- ((dist - r)^2) / (2 * sigma^2))
    sigma defaults to radius / 2 if not provided.
    Returned shape: [1, 1, 1, h, w] ready for broadcasting over [B, F, C, h, w].
    """
    if sigma is None:
        sigma = max(1e-6, radius / 2.0)
    yy = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, w)
    dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    outside = (dist - radius).clamp_min(0.0)
    outside_weight = torch.exp(-(outside**2) / (2.0 * sigma**2))
    weight = torch.where(dist <= radius, torch.ones_like(dist), outside_weight)
    return weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,h,w]


def build_anisotropic_gaussian(
    H: int,
    W: int,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    # *,
    clamp: bool = True,
    normalize: bool = True,
    min_value: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Core builder: create anisotropic Gaussian over (H,W).
      G(y,x) = exp( - ( (x-cx)^2 / (2 sigma_x^2) + (y-cy)^2 / (2 sigma_y^2) ) )
    Returns shape [H,W].

    center_x, center_y: float (pixel coordinates)
    sigma_x, sigma_y: positive float
    """
    sigma_x = max(1e-6, float(sigma_x))
    sigma_y = max(1e-6, float(sigma_y))

    yy = torch.arange(H, device=device, dtype=dtype).view(H, 1)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, W)

    gx = (xx - center_x) ** 2 / (2.0 * sigma_x * sigma_x)
    gy = (yy - center_y) ** 2 / (2.0 * sigma_y * sigma_y)
    gauss = torch.exp(-(gx + gy))

    if normalize:
        m = gauss.max()
        if m > 0:
            gauss = gauss / m
    if clamp:
        gauss = gauss.clamp_(min_value, 1.0)

    return gauss


def build_anisotropic_gaussian_from_bbox(
    H: int,
    W: int,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    # *,
    padding_scale: float = 0.15,
    sigma_scale: float = 0.5,
    min_sigma: float = 1.0,
    clamp: bool = True,
    normalize: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute center & (sigma_x, sigma_y) from a bounding box, then call build_anisotropic_gaussian.

    sigma_x = ( (bbox_width  * (1+padding_scale))/2 ) * sigma_scale
    sigma_y = ( (bbox_height * (1+padding_scale))/2 ) * sigma_scale
    Both clamped by min_sigma.
    """
    # Center
    center_y = 0.5 * (y_min + y_max)
    center_x = 0.5 * (x_min + x_max)

    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1

    eff_h = bbox_h * (1.0 + padding_scale)
    eff_w = bbox_w * (1.0 + padding_scale)

    sigma_y = max(min_sigma, 0.5 * eff_h * sigma_scale)
    sigma_x = max(min_sigma, 0.5 * eff_w * sigma_scale)

    return build_anisotropic_gaussian(
        H=H,
        W=W,
        center_x=center_x,
        center_y=center_y,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        clamp=clamp,
        normalize=normalize,
        device=device,
        dtype=dtype,
    )


def build_anisotropic_gaussian_from_mask(
    mask: np.ndarray | Image.Image | torch.Tensor,
    # *,
    padding_scale: float = 0.15,
    sigma_scale: float = 0.5,
    min_sigma: float = 1.0,
    clamp: bool = True,
    normalize: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """
    Compute bounding box from mask, then call build_anisotropic_gaussian_from_bbox.
    Returns None if mask has no positive pixels.
    """
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    # print(f"{bbox = }")

    if isinstance(mask, torch.Tensor):
        H, W = mask.shape[-2], mask.shape[-1]
    else:
        mask_np = np.asarray(mask)
        H, W = mask_np.shape[-2], mask_np.shape[-1]

    return build_anisotropic_gaussian_from_bbox(
        H=H,
        W=W,
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        padding_scale=padding_scale,
        sigma_scale=sigma_scale,
        min_sigma=min_sigma,
        clamp=clamp,
        normalize=normalize,
        device=mask.device if isinstance(mask, torch.Tensor) else device,
        dtype=dtype,
    )


def combine_gaussian_maps(
    maps: list[torch.Tensor],
    mode: str = "prob_or",
    clamp: bool = True,
) -> torch.Tensor:
    """
    Combine multiple Gaussian (or weight) maps into one in [0,1].

    Args:
        maps: list of tensors with identical shape (e.g. [1,1,1,H,W] or [H,W]).
        mode:
          - "prob_or": 1 - prod(1 - g)   (smooth union, fast saturation)
          - "sum_clamp": clamp(sum(g), 0, 1)
          - "sum_norm": sum(g) / max(sum(g))
          - "max": elementwise max
        clamp: final clamp to [0,1] (except sum_norm which is already normalized).

    Returns:
        Combined tensor.
    """
    assert len(maps) > 0
    if len(maps) == 1:
        out = maps[0]
        return out.clamp_(0, 1) if clamp else out

    stacked = torch.stack(maps, dim=0)

    if mode == "prob_or":
        out = 1.0 - torch.prod(1.0 - stacked, dim=0)
    elif mode == "sum_clamp":
        out = stacked.sum(dim=0)
        if clamp:
            out = out.clamp_(0.0, 1.0)
    elif mode == "sum_norm":
        out = stacked.sum(dim=0)
        maxv = out.max()
        if maxv > 0:
            out = out / maxv
        if clamp:
            out = out.clamp_(0.0, 1.0)
    elif mode == "max":
        out, _ = stacked.max(dim=0)
        if clamp:
            out = out.clamp_(0.0, 1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out
