import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import torch
from torchvision.io import write_video
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from b_spline import build_clamped_bspline, equidistant_points_on_spline

torch.set_grad_enabled(False)

from palette import _palette

import gradio as gr
import numpy as np
from scipy import ndimage
from PIL import Image
import os
from pathlib import Path
import cv2

# from sam_segment import predict_masks_with_sam
from segment_anything import SamPredictor, sam_model_registry

from tensor_utils import (
    image_to_pil,
    image_to_np,
    bbox_from_mask,
    draw_bbox_on_image,
    draw_mask_on_image,
    draw_points_on_image,
    draw_lines_on_image,
    trajectory_interpolate,
    dilate_mask,
    dilate_masks,
)

from optimize_utils import (
    MultiTrajectory,
    Trajectory,
)

import sys

from utils.misc import set_seed

from stream_inference_wrapper import StreamInferenceWrapper
from stream_drag_inference_wrapper import StreamDragInferenceWrapper
from utils.dataset import TextDataset

from video_operations import generate_video, optimize_video

# from compute_objmc import visualize_ground_truth_from_trajectory_file


def extract_layer_as_mask(image_editor, layer_index=0):
    if len(image_editor["layers"]) > layer_index:
        layer = image_editor["layers"][layer_index]
        return image_to_np(layer.convert("L")) > 0
    return None


def apply_mask_to_image(
    mask: np.ndarray | None,
    image: np.ndarray | Image.Image,
    mask_color: list[int],
    alpha: float,
) -> None | Image.Image:
    if image is None:
        return None
    if mask is None:
        return image_to_pil(image)
    mask = np.array(mask)
    new_image = draw_mask_on_image(
        image,
        mask,
        mask_color=mask_color,
        alpha=alpha,
    )
    return new_image


def apply_movable_mask_to_image(
    mask: np.ndarray | None,
    image: np.ndarray | Image.Image,
):
    return apply_mask_to_image(
        mask=mask,
        image=image,
        mask_color=(255, 255, 255),
        alpha=0.35,
    )


def apply_target_mask_to_image(
    mask: np.ndarray | None,
    image: np.ndarray | Image.Image,
):
    return apply_mask_to_image(
        mask=mask,
        image=image,
        mask_color=(255, 64, 64),
        alpha=0.5,
    )


def get_video_last_frame(
    # video: Optional[torch.Tensor],  # None or (t, h, w, c)
    video_path: str,
):
    """
    Loads the last frame from a video.

    Returns:
        Image: The last frame as a PIL Image.
    """
    print(f"Getting last frame from video: {video_path = }")
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            print(f"Video has non-positive frame count: {frame_count}")
            cap.release()
            return None

        # Try direct seek to last frame
        target_index = frame_count - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ret, frame = cap.read()

        # Fallback: iterate to last frame if random access failed
        if (not ret) or frame is None:
            print("Direct seek failed, iterating through frames...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_valid = None
            while True:
                ret_i, frame_i = cap.read()
                if not ret_i:
                    break
                last_valid = frame_i
            frame = last_valid

        if frame is None:
            print("Could not retrieve last frame.")
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame_image = Image.fromarray(frame)
        return last_frame_image
    except Exception as e:
        print(f"Error extracting last frame: {e}")
        return None
    finally:
        cap.release()


def sam_predict_segmentation(
    sam_predictor: SamPredictor,
    origin_image: Image.Image | np.ndarray,
    restriction_mask: np.ndarray,  # (h, w), bool
    click_points: list[tuple[int, int]],
    previous_sam_logits: np.ndarray | None,  # (3, 256, 256)
):
    # print(f"{restriction_mask.shape = }")

    origin_image_np = image_to_np(origin_image)
    # print(f"{origin_image_np.shape = }")
    sam_predictor.set_image(origin_image_np)

    if previous_sam_logits is not None:
        print(f"{previous_sam_logits.shape = }")
    else:
        print(f"{previous_sam_logits = }")
    masks, scores, logits = sam_predictor.predict(
        point_coords=np.array(click_points),
        point_labels=np.ones((len(click_points),)),
        mask_input=(previous_sam_logits[0:1] if previous_sam_logits is not None else None),
        multimask_output=True,
    )
    # mask: np.ndarray
    # scores: np.ndarray
    # logits: np.ndarray
    # print(f"{masks.shape = }") # (3, 480, 832)
    # print(f"{logits.shape = }") # (3, 256, 256)

    mask = masks[0]
    mask *= restriction_mask

    logits *= cv2.resize(
        restriction_mask.astype(np.uint8),
        dsize=(256, 256),
        interpolation=cv2.INTER_LINEAR,
    )

    return mask, logits


def sam_predict_segmentation_wrapper(
    sam_predictor: SamPredictor,
    original_image: Image.Image | np.ndarray,
    restriction_mask: np.ndarray | None,
    previous_click_points: list[tuple[int, int]],
    previous_sam_logits: np.ndarray | None,
    bypass_sam_model: bool,
    evt: gr.SelectData,
):
    # print(f"{restriction_mask = }")
    original_image = image_to_pil(original_image).convert("RGB")

    if restriction_mask is None:
        labeled_restriction_mask = np.zeros(
            (original_image.height, original_image.width), dtype=np.int32
        )
    else:
        labeled_restriction_mask, _ = ndimage.label(restriction_mask, structure=np.ones((3, 3)))
    # print(f"{labeled_restriction_mask = }")
    current_click_label = labeled_restriction_mask[evt.index[1], evt.index[0]]
    # print(f"{current_click_label = }")

    if current_click_label == 0:
        selected_component_mask = np.zeros_like(labeled_restriction_mask, dtype=bool)
    else:
        selected_component_mask = labeled_restriction_mask == current_click_label
    # print(f"{selected_component_mask = }")

    if bypass_sam_model:
        click_points = [evt.index]
        mask = selected_component_mask
        logits = None
    else:
        click_points = previous_click_points + [evt.index]
        mask, logits = sam_predict_segmentation(
            sam_predictor=sam_predictor,
            origin_image=original_image,
            restriction_mask=selected_component_mask,
            click_points=click_points,
            previous_sam_logits=previous_sam_logits,
        )

    return mask, click_points, logits


def draw_all_sam_masks(image: Image.Image | None, mask_list: list[np.ndarray]):
    if image is None:
        return None
    if len(mask_list) == 0:
        pass
    else:
        for mask_idx, mask in enumerate(mask_list):
            image = draw_mask_on_image(
                image,
                mask,
                mask_color=tuple(_palette[mask_idx + 1]),
                alpha=0.65,
            )
    return image


def draw_sam_mask_wrapper(
    original_image,
    movable_mask,
    current_mask: np.ndarray | None,
    previous_masks: list[np.ndarray],
    click_points: list[tuple[int, int]],
):
    image = apply_movable_mask_to_image(
        image=original_image,
        mask=movable_mask,
    )
    if image is None:
        return None
    image = draw_all_sam_masks(
        image,
        previous_masks + ([current_mask] if current_mask is not None else []),
    )
    image = draw_points_on_image(
        image,
        click_points,
        color=[(0, 255, 0, 255) for l in click_points],
        radius=5,
    )
    return image


def save_sam_masks(
    current_mask: np.ndarray | None,
    previous_masks: list[np.ndarray],
):
    new_masks = previous_masks + ([current_mask] if current_mask is not None else [])
    return None, new_masks, [], None


def select_target_sam_mask(
    masks_list: list[np.ndarray],
    evt: gr.SelectData,
):
    is_match_mask = False
    for mask_index, sam_mask in enumerate(masks_list):
        # check if evt point in sam_mask
        if sam_mask[evt.index[1], evt.index[0]]:
            is_match_mask = True
            break

    if not is_match_mask:
        print(f"Mask not found for {evt.index = }")
        mask_index = -1
    return mask_index


def draw_rotation_trajectory(
    image,
    points,
):
    image = draw_points_on_image(
        image,
        [points[0]],
        color="green",
        radius=15,
    )
    if len(points) > 1:
        image = draw_points_on_image(
            image,
            points[1:],
            color=[
                (
                    255 - int(float(i) / len(points[1:]) * 255.0),
                    64,
                    int(float(i) / len(points[1:]) * 255.0),
                    255,
                )
                for i in range(len(points[1:]))
            ],
            radius=5,
        )
        for point in points[1:]:
            image = draw_lines_on_image(
                image,
                [points[0], point],
                color="green",
                width=3,
            )

    return image


def draw_translation_trajectory(
    image,
    points,
    control_points: list[tuple[int, int]] = [],
    is_draw_control_points: bool = True,
):
    if len(points) == 1:
        image = draw_points_on_image(
            image,
            points,
            color=[(255, 64, 0, 255)],
            radius=6,
        )
        return image
    if is_draw_control_points and (len(control_points) >= 2):
        image = draw_points_on_image(
            image,
            control_points,
            color=[(0, 255, 0, 255) for _ in control_points],
            radius=3,
        )
        image = draw_lines_on_image(
            image,
            control_points,
            color=[(0, 255, 0, 255) for _ in control_points],
            width=2,
        )
    image = draw_lines_on_image(
        image,
        points,
        color=[
            (
                255 - int(float(i) / len(points[1:]) * 255.0),
                64,
                int(float(i) / len(points[1:]) * 255.0),
                255,
            )
            for i in range(len(points))
        ],
        width=4,
    )
    image = draw_points_on_image(
        image,
        points,
        color=[
            (
                255 - int(float(i) / len(points[1:]) * 255.0),
                64,
                int(float(i) / len(points[1:]) * 255.0),
                255,
            )
            for i in range(len(points))
        ],
        radius=6,
    )

    return image


def draw_all_trajectories(
    image,
    trajectory: MultiTrajectory,
    is_draw_control_points: bool = True,
):
    print(
        f"""
draw_all_trajectories:
"""
    )
    if trajectory.trajectories is None:
        return image
    for traj in trajectory.trajectories:
        if traj.original_trajectory is None:
            continue
        original_traj = traj.original_trajectory
        if original_traj["is_rotation"]:
            image = draw_rotation_trajectory(image, original_traj["points"])
        else:
            image = draw_translation_trajectory(
                image,
                original_traj["points"],
                original_traj.get("control_points", []),
                is_draw_control_points=is_draw_control_points,
            )

    return image


def draw_trajectory_image(
    original_image,
    movable_mask,
    mask_index,
    masks_list: list[np.ndarray],
    trajectory: MultiTrajectory,
    is_draw_bbox: bool = True,
    is_draw_control_points: bool = True,
):
    print(
        f"""
draw_trajectory_image:
    {mask_index = }
"""
    )
    image = apply_movable_mask_to_image(
        mask=movable_mask,
        image=original_image,
    )
    image = draw_all_sam_masks(image, masks_list)
    if (
        (mask_index is not None)
        and (mask_index >= 0)
        and (mask_index < len(masks_list))
        and is_draw_bbox
    ):
        image = draw_bbox_on_image(image, bbox_from_mask(masks_list[mask_index]))
    image = draw_all_trajectories(
        image,
        trajectory,
        is_draw_control_points=is_draw_control_points,
    )
    return image


def update_trajectory(
    trajectory: MultiTrajectory,
    mask_index: int,
    drag_animation_select: str,
    translate_rotate_select: str,
    evt: gr.SelectData,
):
    print(f"update_trajectory")

    # Work on a deep copy so Gradio sees a new object
    trajectory = copy.deepcopy(trajectory)

    if mask_index < 0:
        print(f"Invalid mask_index: {mask_index}")
        return trajectory

    # print(f"{evt.index = }")
    x_center, y_center = evt.index  # evt.value is (x, y)

    clicked_point = (x_center, y_center)
    print(f"{clicked_point = }")

    # Ensure trajectories list is large enough
    while len(trajectory.trajectories) <= mask_index:
        trajectory.trajectories.append(Trajectory())

    existing_traj_obj = trajectory.trajectories[mask_index]
    if existing_traj_obj.original_trajectory is not None:
        current_trajectory = dict(existing_traj_obj.original_trajectory)
    else:
        current_trajectory = {}

    if translate_rotate_select == "Translation":
        current_trajectory["is_rotation"] = False

        # Append clicked control point
        control_points = current_trajectory.get("control_points", [])
        control_points = control_points + [clicked_point]

        # Drag vs Animation behavior
        if drag_animation_select == "Drag":
            # Restrict to last two control points, sample exactly 2 points
            if len(control_points) > 2:
                control_points = [clicked_point]
            num_traj_points = 2
        elif drag_animation_select == "Animation":
            # No restriction on control points, sample N = 1 + 3 * block_number
            num_traj_points = 1 + 3 * int(trajectory.block_number)
        else:
            raise ValueError(f"Invalid drag_animation_select: {drag_animation_select}")

        current_trajectory["control_points"] = control_points

        # Compute trajectory points along BSpline (or pad if not enough controls)
        if len(control_points) < 2:
            sampled_pts = [control_points[0]] * num_traj_points
        else:
            spline = build_clamped_bspline(control_points, degree=3)
            pts = equidistant_points_on_spline(spline, num_points=num_traj_points, grid=8000)
            sampled_pts = [(int(round(px)), int(round(py))) for px, py in pts]

        current_trajectory["points"] = sampled_pts

    elif translate_rotate_select == "Rotation":
        current_trajectory["is_rotation"] = True

        # Initialize if missing, else apply 3-point logic
        if "points" not in current_trajectory or current_trajectory["points"] is None:
            current_trajectory["points"] = [clicked_point]
        else:
            pts = current_trajectory["points"] + [clicked_point]

            # If about to exceed 3, reset to the new point
            if len(pts) > 3:
                current_trajectory["points"] = [clicked_point]
            # If less than 3, just append
            elif len(pts) < 3:
                current_trajectory["points"] = pts
            else:
                # len(pts) == 3: pts[0] is rotation center
                if drag_animation_select == "Animation":
                    first = trajectory_interpolate(pts[1:], scale=int(trajectory.block_number))
                    second = trajectory_interpolate(first, scale=3)
                    current_trajectory["points"] = pts[0:1] + second
                else:
                    # Drag: do not interpolate
                    current_trajectory["points"] = pts
    else:
        raise ValueError("Invalid translation/rotation selection")

    # Update the Trajectory object in-place (recomputes block_trajectories)
    existing_traj_obj.set_original_trajectory(current_trajectory)
    # print(f"{trajectory = }")

    return trajectory


def save_trajectory(
    save_dir: Path,
    saved_trajectory: MultiTrajectory,
    original_image: Image.Image,
    current_block_index: int,
    masks: list[np.ndarray],
):
    print(f"save_trajectory")
    print(f"{save_dir = }")
    print(f"{saved_trajectory = }")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    drag_animation_select = saved_trajectory.drag_or_animation_select or "Drag"
    save_prefix = f"block_{current_block_index}_{drag_animation_select}"

    # Use MultiTrajectory's save method
    saved_trajectory.save(
        save_dir=save_dir,
        prefix=save_prefix,
    )

    # Save the trajectory image
    trajectory_image = draw_trajectory_image(
        original_image=original_image,
        movable_mask=saved_trajectory.movable_mask,
        mask_index=None,
        masks_list=masks,
        trajectory=saved_trajectory,
        is_draw_bbox=False,
        is_draw_control_points=False,
    )
    trajectory_image.save(save_dir / f"{save_prefix}_trajectory.png")


def clear_current_trajectory(
    idx: int,
    trajectory: MultiTrajectory,
):
    trajectory = copy.deepcopy(trajectory)
    """Clear the trajectory at the given mask index."""
    try:
        idx_int = int(idx)
    except Exception:
        return trajectory

    if not trajectory.trajectories:
        return trajectory

    if idx_int < 0 or idx_int >= len(trajectory.trajectories):
        return trajectory

    # Reset this trajectory (keep the mask)
    mask = trajectory.trajectories[idx_int].mask
    trajectory.trajectories[idx_int] = Trajectory(mask=mask)

    return trajectory


def clear_all_trajectories(
    trajectory: MultiTrajectory,
):
    trajectory = copy.deepcopy(trajectory)
    """Clear all trajectories but keep the masks."""
    if trajectory.trajectories is not None:
        for i in range(len(trajectory.trajectories)):
            mask = trajectory.trajectories[i].mask
            trajectory.trajectories[i] = Trajectory(mask=mask)
    return trajectory


def sync_trajectory_masks(saved_trajectory: MultiTrajectory, dilated_masks: list[np.ndarray]):
    """Resize saved_trajectory.trajectories to match the number of dilated masks,
    and update each Trajectory.mask with the corresponding dilated mask."""
    saved_trajectory = copy.deepcopy(saved_trajectory)

    current_len = len(saved_trajectory.trajectories)
    target_len = len(dilated_masks) if dilated_masks else 0

    if target_len > current_len:
        # Expand: append new empty Trajectory objects
        for _ in range(target_len - current_len):
            saved_trajectory.trajectories.append(Trajectory())
    elif target_len < current_len:
        # Shrink: truncate
        saved_trajectory.trajectories = saved_trajectory.trajectories[:target_len]

    # Update each Trajectory.mask
    for i, mask in enumerate(dilated_masks):
        saved_trajectory.trajectories[i].mask = mask

    return saved_trajectory


def add_listeners_to_trajectory(
    saved_trajectory: MultiTrajectory,
    prompt_box: gr.Textbox,
    trajectory_block_number_slider: gr.Slider,
    drag_animation_select: gr.Dropdown,
    movable_area_mask: gr.State,
    dilated_saved_sam_predicted_masks: gr.State,
):
    # Sync prompt into saved_trajectory when prompt_box changes

    def sync_trajectory_prompt(saved_trajectory: MultiTrajectory, prompt: str):
        saved_trajectory.prompt = prompt
        return saved_trajectory

    prompt_box.change(
        fn=sync_trajectory_prompt,
        inputs=[saved_trajectory, prompt_box],
        outputs=saved_trajectory,
        trigger_mode="always_last",
    )

    # Sync block_number into saved_trajectory when trajectory_block_number_slider changes
    def sync_trajectory_block_number(saved_trajectory: MultiTrajectory, block_number: int):
        saved_trajectory.block_number = block_number
        return saved_trajectory

    trajectory_block_number_slider.change(
        fn=sync_trajectory_block_number,
        inputs=[saved_trajectory, trajectory_block_number_slider],
        outputs=saved_trajectory,
        trigger_mode="always_last",
    )

    # Sync drag_or_animation_select into saved_trajectory when drag_animation_select changes
    def sync_trajectory_drag_animation(
        saved_trajectory: MultiTrajectory, drag_animation_select: str
    ):
        saved_trajectory.drag_or_animation_select = drag_animation_select
        return saved_trajectory

    drag_animation_select.change(
        fn=sync_trajectory_drag_animation,
        inputs=[saved_trajectory, drag_animation_select],
        outputs=saved_trajectory,
        trigger_mode="always_last",
    )

    # Sync movable_area_mask into saved_trajectory when it changes
    def sync_trajectory_movable_mask(saved_trajectory: MultiTrajectory, movable_mask):
        saved_trajectory.movable_mask = movable_mask
        return saved_trajectory

    movable_area_mask.change(
        fn=sync_trajectory_movable_mask,
        inputs=[saved_trajectory, movable_area_mask],
        outputs=saved_trajectory,
        trigger_mode="always_last",
    )

    # Sync dilated_saved_sam_predicted_masks into saved_trajectory when it changes
    dilated_saved_sam_predicted_masks.change(
        fn=sync_trajectory_masks,
        inputs=[saved_trajectory, dilated_saved_sam_predicted_masks],
        outputs=saved_trajectory,
        trigger_mode="always_last",
    )


def create_generate_video_ui(
    label_root: str | Path,
    text_dataset: Dataset,
    video_path: gr.State,
    stream_drag_inference: StreamDragInferenceWrapper,
    output_dir: str | Path,
    original_image: gr.State,
):
    with gr.Row():
        prompt_index_number = gr.Number(
            label="Step 1: Select Prompt Index Here",
            interactive=True,
            scale=1,
        )
        prompt_box = gr.Textbox(
            label="Prompt",
            interactive=True,
            scale=3,
        )
        save_dir_text_box = gr.Textbox(
            label="Save Directory",
            interactive=False,
            scale=1,
        )
    prompt_index_number.change(
        fn=lambda prompt_index_number: text_dataset[prompt_index_number]["prompts"],
        inputs=prompt_index_number,
        outputs=[
            prompt_box,
        ],
    )
    gr.on(
        triggers=[
            prompt_box.change,
        ],
        fn=lambda prompt_index_number, prompt: str(
            label_root / f"{prompt_index_number:04d}-{prompt[:50].replace(' ', '_')}"
        ),
        inputs=[prompt_index_number, prompt_box],
        outputs=save_dir_text_box,
        trigger_mode="always_last",
    )
    with gr.Row():
        current_block_index_slider = gr.Slider(
            label="Current Start Block Index",
            minimum=0,
            maximum=50,
            value=0,
            step=1,
        )
        generate_block_number_slider = gr.Slider(
            label="Step 2: Select Number of Blocks to Generate",
            minimum=1,
            maximum=50,
            value=2,
            step=1,
        )
    with gr.Row():
        begin_generate_button = gr.Button(
            value="Step 3: Click Here to Begin Generation",
        )
        refresh_video_display_button = gr.Button(value="Refresh Video Display")

    with gr.Row():
        video_display = gr.Video()

    begin_generate_button.click(
        fn=lambda pi, p, sbi, bn: generate_video(
            stream_inference_model=stream_drag_inference,
            prompt_index=pi,
            prompt=p,
            start_block_index=sbi,
            block_number=bn,
            output_dir=output_dir,
        ),
        inputs=[
            prompt_index_number,
            prompt_box,
            current_block_index_slider,
            generate_block_number_slider,
        ],
        outputs=[video_path, current_block_index_slider],
    )
    gr.on(
        triggers=[
            refresh_video_display_button.click,
            video_path.change,
        ],
        fn=lambda video_path: video_path,
        inputs=video_path,
        outputs=video_display,
        trigger_mode="always_last",
    )

    with gr.Row():
        get_last_frame_button = gr.Button(
            value="Get Last Frame (Normally No Need to Click This, In Case the Last Frame Fails to Update due to Gradio Bug)",
        )
    gr.on(
        triggers=[
            video_path.change,
            get_last_frame_button.click,
        ],
        fn=get_video_last_frame,
        inputs=video_path,
        outputs=original_image,
    )

    return (
        prompt_index_number,
        save_dir_text_box,
        prompt_box,
        current_block_index_slider,
        generate_block_number_slider,
    )


def create_movable_area_ui(
    movable_area_mask: gr.State,
    original_image: gr.State,
):

    with gr.Row():
        movable_area_image_editor = gr.ImageEditor(
            label="Step 4: This is Last Frame of Video, Draw Editable Area Here. (Normally This Should Be Large and Cover all Possible Area Where the Object You Want to Move/Animate to)",
            type="pil",
            interactive=True,
            brush=gr.Brush(
                default_size=100,
                colors=[
                    "rgba(0, 0, 255, 0.5)",
                ],
                default_color="auto",
                color_mode="defaults",
            ),
        )
    movable_area_image_editor.change(
        fn=extract_layer_as_mask,
        inputs=movable_area_image_editor,
        outputs=movable_area_mask,
        trigger_mode="always_last",
    )
    original_image.change(
        fn=lambda image: image,
        inputs=original_image,
        outputs=movable_area_image_editor,
        trigger_mode="always_last",
    )
    with gr.Row():
        refresh_movable_area_button = gr.Button(
            value="Refresh Movable Area (Normally No Need to Click This, In Case the Mask Fails to Update due to Gradio Bug)"
        )
    refresh_movable_area_button.click(
        fn=extract_layer_as_mask,
        inputs=movable_area_image_editor,
        outputs=movable_area_mask,
        trigger_mode="always_last",
    )


def create_target_area_ui(
    target_area_mask: gr.State,
    original_image: gr.State,
    movable_area_mask: gr.State,
):
    with gr.Row():
        target_area_image_editor = gr.ImageEditor(
            label="Step 5: Draw Target Area on the Object You Want to Move/Animate (Normally This Should Be a Subset of Editable Area) (Normally This Mask should be Bigger than the Desired Object)",
            type="pil",
            interactive=True,
            brush=gr.Brush(
                default_size=50,
                colors=[
                    "rgba(255, 0, 0, 0.5)",
                ],
                default_color="auto",
                color_mode="defaults",
            ),
        )
    target_area_image_editor.change(
        fn=extract_layer_as_mask,
        inputs=target_area_image_editor,
        outputs=target_area_mask,
        trigger_mode="always_last",
    )
    gr.on(
        triggers=[
            original_image.change,
            movable_area_mask.change,
        ],
        fn=apply_movable_mask_to_image,
        inputs=[
            movable_area_mask,
            original_image,
        ],
        outputs=target_area_image_editor,
        trigger_mode="always_last",
    )

    with gr.Row():
        refresh_target_area_button = gr.Button(
            value="Refresh Target Area (Normally No Need to Click This, In Case the Mask Fails to Update due to Gradio Bug)"
        )
    refresh_target_area_button.click(
        fn=extract_layer_as_mask,
        inputs=target_area_image_editor,
        outputs=target_area_mask,
        trigger_mode="always_last",
    )


def create_sam_segmentation_ui(
    original_image: gr.State,
    movable_area_mask: gr.State,
    target_area_mask: gr.State,
    sam_predictor: SamPredictor,
    sam_click_points: gr.State,
    sam_saved_logits: gr.State,
    current_sam_predicted_mask: gr.State,
    saved_sam_predicted_masks: gr.State,
    dilated_current_sam_predicted_mask: gr.State,
    dilated_saved_sam_predicted_masks: gr.State,
):
    with gr.Row():
        refresh_sam_segment_click_image_button = gr.Button(
            value="Refresh Target Area Mask Display (Normally No Need to Click This, In Case the Mask Fails to Update due to Gradio Bug)"
        )
    with gr.Row():
        sam_segment_click_image = gr.Image(
            label="Step 6: Click to Perform SAM Segment on Target Area, Segment the Object You Want to Move/Animate. The SAM Mask is Restricted within the Target Area Mask",
            type="pil",
            interactive=True,
        )
    gr.on(
        triggers=[
            original_image.change,
            movable_area_mask.change,
            target_area_mask.change,
            refresh_sam_segment_click_image_button.click,
        ],
        fn=lambda movable_mask, target_mask, image: apply_target_mask_to_image(
            target_mask,
            apply_movable_mask_to_image(
                movable_mask,
                image,
            ),
        ),
        inputs=[
            movable_area_mask,
            target_area_mask,
            original_image,
        ],
        outputs=sam_segment_click_image,
        trigger_mode="always_last",
    )

    with gr.Row():
        dilate_mask_slider = gr.Slider(
            label="Dilate Mask Pixel",
            minimum=0,
            maximum=50,
            value=15,
            step=1,
        )
        bypass_sam_model_check_box = gr.Checkbox(
            label="Bypass SAM Model",
            value=False,
        )

    def sam_predict_segmentation_wrapper_wrapper(
        oi,
        rm,
        pcp,
        psl,
        bs,
        evt: gr.SelectData,
    ):
        return sam_predict_segmentation_wrapper(
            sam_predictor=sam_predictor,
            original_image=oi,
            restriction_mask=rm,
            previous_click_points=pcp,
            previous_sam_logits=psl,
            bypass_sam_model=bs,
            evt=evt,
        )

    sam_segment_click_image.select(
        fn=sam_predict_segmentation_wrapper_wrapper,
        inputs=[
            original_image,
            target_area_mask,
            sam_click_points,
            sam_saved_logits,
            bypass_sam_model_check_box,
        ],
        outputs=[
            current_sam_predicted_mask,
            sam_click_points,
            sam_saved_logits,
        ],
        trigger_mode="always_last",
    )
    gr.on(
        triggers=[
            current_sam_predicted_mask.change,
            dilate_mask_slider.change,
        ],
        fn=dilate_mask,
        inputs=[
            current_sam_predicted_mask,
            dilate_mask_slider,
        ],
        outputs=dilated_current_sam_predicted_mask,
        trigger_mode="always_last",
    )
    gr.on(
        triggers=[
            saved_sam_predicted_masks.change,
            dilate_mask_slider.change,
        ],
        fn=dilate_masks,
        inputs=[
            saved_sam_predicted_masks,
            dilate_mask_slider,
        ],
        outputs=dilated_saved_sam_predicted_masks,
        trigger_mode="always_last",
    )


def create_sam_mask_management_ui(
    original_image: gr.State,
    movable_area_mask: gr.State,
    dilated_current_sam_predicted_mask: gr.State,
    dilated_saved_sam_predicted_masks: gr.State,
    sam_click_points: gr.State,
    current_sam_predicted_mask: gr.State,
    saved_sam_predicted_masks: gr.State,
    sam_saved_logits: gr.State,
):
    with gr.Row():
        save_sam_masks_button = gr.Button(
            value="Step 7: Save the Current SAM Mask",
        )
        cancel_sam_mask_button = gr.Button(value="Cancel Current SAM Mask")
        delete_sam_mask_button = gr.Button(value="Delete All SAM Masks")
    save_sam_masks_button.click(
        fn=save_sam_masks,
        inputs=[
            current_sam_predicted_mask,
            saved_sam_predicted_masks,
        ],
        outputs=[
            current_sam_predicted_mask,
            saved_sam_predicted_masks,
            sam_click_points,
            sam_saved_logits,
        ],
        trigger_mode="always_last",
    )
    with gr.Row():
        sam_segment_display_image = gr.Image(
            label="Step 8: Display the SAM Segmentation, Click to Select Target Object to Create Trajectory",
            type="pil",
            interactive=True,
        )
    gr.on(
        triggers=[
            original_image.change,
            movable_area_mask.change,
            dilated_current_sam_predicted_mask.change,
            dilated_saved_sam_predicted_masks.change,
            sam_click_points.change,
        ],
        fn=draw_sam_mask_wrapper,
        inputs=[
            original_image,
            movable_area_mask,
            dilated_current_sam_predicted_mask,
            dilated_saved_sam_predicted_masks,
            sam_click_points,
        ],
        outputs=sam_segment_display_image,
        trigger_mode="always_last",
    )
    cancel_sam_mask_button.click(
        fn=lambda: (None, [], None),
        outputs=[
            current_sam_predicted_mask,
            sam_click_points,
            sam_saved_logits,
        ],
        trigger_mode="always_last",
    )
    gr.on(
        triggers=[
            # target_area_mask.change,
            delete_sam_mask_button.click,
        ],
        fn=lambda: (None, [], [], None),
        outputs=[
            current_sam_predicted_mask,
            saved_sam_predicted_masks,
            sam_click_points,
            sam_saved_logits,
        ],
        trigger_mode="always_last",
    )
    with gr.Row():
        current_selected_mask_index_number = gr.Number(
            label="Current Selected Mask Index",
            interactive=False,
        )

    sam_segment_display_image.select(
        fn=select_target_sam_mask,
        inputs=[
            saved_sam_predicted_masks,
        ],
        outputs=[
            current_selected_mask_index_number,
        ],
        trigger_mode="always_last",
    )

    return current_selected_mask_index_number


def create_trajectory_display_ui(
    original_image: gr.State,
    movable_area_mask: gr.State,
    dilated_saved_sam_predicted_masks: gr.State,
    saved_trajectory: gr.State,
    current_selected_mask_index_number: gr.State,
):
    with gr.Row():
        trajectory_block_number_slider = gr.Slider(
            label="Step 9: Select Number of Trajectory Blocks (For Animation Only, More Blocks Means Longer Animation, For Drag, This Should be 1)",
            minimum=1,
            maximum=10,
            value=1,
            step=1,
        )
    with gr.Row():
        drag_animation_select = gr.Dropdown(
            choices=["Drag", "Animation"],
            label="Step 10: Select Drag or Animation",
        )
        translate_rotate_select = gr.Dropdown(
            choices=["Translation", "Rotation"],
            label="Step 11: Select Translation or Rotation",
        )

    with gr.Row():
        trajectory_display_image = gr.Image(
            label="Step 12: Click on the Object in the Image to Create Trajectory. The Translation Trajectory is Controlled by Bspline Interpolation. The Rotation Trajectory is Controlled by 3 Points",
            type="pil",
            interactive=False,
        )
    gr.on(
        triggers=[
            original_image.change,
            movable_area_mask.change,
            current_selected_mask_index_number.change,
            dilated_saved_sam_predicted_masks.change,
            saved_trajectory.change,
        ],
        fn=draw_trajectory_image,
        inputs=[
            original_image,
            movable_area_mask,
            current_selected_mask_index_number,
            dilated_saved_sam_predicted_masks,
            saved_trajectory,
        ],
        outputs=trajectory_display_image,
        trigger_mode="always_last",
    )

    trajectory_display_image.select(
        fn=update_trajectory,
        inputs=[
            saved_trajectory,
            current_selected_mask_index_number,
            drag_animation_select,
            translate_rotate_select,
        ],
        outputs=saved_trajectory,
    )

    return drag_animation_select, trajectory_block_number_slider


def create_trajectory_management_ui(
    save_dir_text_box: gr.Textbox,
    original_image: gr.State,
    current_block_index_slider: gr.Slider,
    saved_trajectory: gr.State,
    dilated_saved_sam_predicted_masks: gr.State,
    current_selected_mask_index_number: gr.Number,
):
    with gr.Row():
        save_trajectory_button = gr.Button(
            value="Step 13: Save Trajectory",
        )
        delete_current_trajectory_button = gr.Button(value="Delete Current Trajectory")
        delete_all_trajectory_button = gr.Button(value="Delete All Trajectories")
    save_trajectory_button.click(
        fn=save_trajectory,
        inputs=[
            save_dir_text_box,
            saved_trajectory,
            original_image,
            current_block_index_slider,
            dilated_saved_sam_predicted_masks,
        ],
    )
    delete_current_trajectory_button.click(
        fn=clear_current_trajectory,
        inputs=[current_selected_mask_index_number, saved_trajectory],
        outputs=[saved_trajectory],
    )
    delete_all_trajectory_button.click(
        fn=clear_all_trajectories,
        inputs=[saved_trajectory],
        outputs=[saved_trajectory],
    )


def create_ui(
    text_dataset: Dataset,
    label_root: str | Path,
    output_dir: str | Path,
    sam_predictor: SamPredictor,
    stream_drag_inference: StreamDragInferenceWrapper,
):
    with gr.Blocks() as demo:
        video_path = gr.State(value=None)
        original_image = gr.State(value=None)
        movable_area_mask = gr.State(value=None)
        target_area_mask = gr.State(value=None)

        sam_click_points = gr.State(value=[])
        sam_saved_logits = gr.State(value=None)
        saved_sam_predicted_masks = gr.State(value=[])
        current_sam_predicted_mask = gr.State(value=None)

        dilated_current_sam_predicted_mask = gr.State(value=None)
        dilated_saved_sam_predicted_masks = gr.State(value=[])

        saved_trajectory = gr.State(value=MultiTrajectory())

        (
            prompt_index_number,
            save_dir_text_box,
            prompt_box,
            current_block_index_slider,
            generate_block_number_slider,
        ) = create_generate_video_ui(
            label_root=label_root,
            text_dataset=text_dataset,
            video_path=video_path,
            stream_drag_inference=stream_drag_inference,
            output_dir=output_dir,
            original_image=original_image,
        )

        create_movable_area_ui(movable_area_mask, original_image)
        create_target_area_ui(target_area_mask, original_image, movable_area_mask)
        create_sam_segmentation_ui(
            original_image=original_image,
            movable_area_mask=movable_area_mask,
            target_area_mask=target_area_mask,
            sam_predictor=sam_predictor,
            sam_click_points=sam_click_points,
            sam_saved_logits=sam_saved_logits,
            current_sam_predicted_mask=current_sam_predicted_mask,
            saved_sam_predicted_masks=saved_sam_predicted_masks,
            dilated_current_sam_predicted_mask=dilated_current_sam_predicted_mask,
            dilated_saved_sam_predicted_masks=dilated_saved_sam_predicted_masks,
        )

        current_selected_mask_index_number = create_sam_mask_management_ui(
            original_image=original_image,
            movable_area_mask=movable_area_mask,
            dilated_current_sam_predicted_mask=dilated_current_sam_predicted_mask,
            dilated_saved_sam_predicted_masks=dilated_saved_sam_predicted_masks,
            sam_click_points=sam_click_points,
            current_sam_predicted_mask=current_sam_predicted_mask,
            saved_sam_predicted_masks=saved_sam_predicted_masks,
            sam_saved_logits=sam_saved_logits,
        )

        drag_animation_select, trajectory_block_number_slider = create_trajectory_display_ui(
            original_image=original_image,
            movable_area_mask=movable_area_mask,
            dilated_saved_sam_predicted_masks=dilated_saved_sam_predicted_masks,
            saved_trajectory=saved_trajectory,
            current_selected_mask_index_number=current_selected_mask_index_number,
        )
        create_trajectory_management_ui(
            save_dir_text_box=save_dir_text_box,
            original_image=original_image,
            current_block_index_slider=current_block_index_slider,
            saved_trajectory=saved_trajectory,
            dilated_saved_sam_predicted_masks=dilated_saved_sam_predicted_masks,
            current_selected_mask_index_number=current_selected_mask_index_number,
        )

        add_listeners_to_trajectory(
            saved_trajectory=saved_trajectory,
            prompt_box=prompt_box,
            trajectory_block_number_slider=trajectory_block_number_slider,
            drag_animation_select=drag_animation_select,
            movable_area_mask=movable_area_mask,
            dilated_saved_sam_predicted_masks=dilated_saved_sam_predicted_masks,
        )

        with gr.Row():
            begin_optimize_button = gr.Button(
                value="Step 14: Click Here to Begin Optimize, Wait for a Moment and the Dragged/Animated Video will be Displayed Above",
            )
        begin_optimize_button.click(
            fn=lambda pi, sbi, st: optimize_video(
                stream_drag_inference_model=stream_drag_inference,
                output_dir=output_dir,
                prompt_index=pi,
                start_block_index=sbi,
                multi_trajectory=st,
            ),
            inputs=[
                prompt_index_number,
                current_block_index_slider,
                saved_trajectory,
            ],
            outputs=[
                video_path,
                current_block_index_slider,
            ],
        )
        with gr.Row():
            clear_all_button = gr.Button(
                value="Step 15: Remember to Click Here to Clear All Before Generation/Editing on Next Video, Otherwise the Previous KV Cache will Affect the Generation/Editing of Next Video",
            )

        def clear_all():
            stream_drag_inference.reset()

            return (
                0,
                None,
                None,
                None,
                None,
                [],
                None,
                [],
                None,
                MultiTrajectory(),
            )

        clear_all_button.click(
            fn=clear_all,
            outputs=[
                current_block_index_slider,
                video_path,
                original_image,
                movable_area_mask,
                target_area_mask,
                sam_click_points,
                sam_saved_logits,
                saved_sam_predicted_masks,
                current_sam_predicted_mask,
                saved_trajectory,
            ],
        )

    return demo


def main():
    sam_model = sam_model_registry["vit_h"](checkpoint="../segment-anything/sam_vit_h_4b8939.pth")
    sam_model.to(device="cuda")
    sam_predictor = SamPredictor(sam_model)

    SEED = 42

    text_dataset = TextDataset(prompt_path="prompts/MovieGenVideoBench_extended.txt")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir = "configs"
    stream_config_name = "self_forcing_dmd_vsink_stream_drag"
    with initialize(version_base=None, config_path=config_dir):
        stream_config = compose(config_name=stream_config_name)
    print(f"{stream_config = }")

    stream_drag_inference = StreamDragInferenceWrapper(
        stream_model_config=stream_config,
        checkpoint_path="./checkpoints/self_forcing_dmd.pt",
        total_generate_block_number=36,
        use_ema=True,
        seed=SEED,
    )
    label_save_dir = Path("./saved_labels")
    label_save_dir = label_save_dir / f"{stream_config_name}-seed{SEED}"
    label_save_dir.mkdir(parents=True, exist_ok=True)

    output_save_dir = Path("outputs-editing")
    output_save_dir = output_save_dir / f"{stream_config_name}-seed{SEED}"
    output_save_dir.mkdir(parents=True, exist_ok=True)

    demo = create_ui(
        text_dataset=text_dataset,
        label_root=label_save_dir,
        output_dir=output_save_dir,
        sam_predictor=sam_predictor,
        stream_drag_inference=stream_drag_inference,
    )
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
