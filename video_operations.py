from pathlib import Path

import torch
from torchvision.io import write_video

from optimize_utils import MultiTrajectory
from stream_drag_inference_wrapper import StreamDragInferenceWrapper
from stream_inference_wrapper import StreamInferenceWrapper
from utils.misc import set_seed


def run_inference(
    model: StreamDragInferenceWrapper,
    start_block_index: int,
    end_block_index: int,
    prompt: str,
    multiple_trajectory: MultiTrajectory | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single inference call (shared by animation, drag, and generation).
    """
    with torch.no_grad():
        all_video, current_video = model.inference(
            start_block_index=start_block_index,
            end_block_index=end_block_index,
            prompt=prompt,
            multiple_trajectory=multiple_trajectory,
        )
    return all_video, current_video


def run_optimization(
    model: StreamDragInferenceWrapper,
    trajectory: MultiTrajectory,
    start_block_index: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Run drag or animation optimization and return (all_video, current_video, end_block_index).
    """
    mode = trajectory.drag_or_animation_select

    if mode == "Animation":
        end_block_index = start_block_index + int(trajectory.block_number)
        all_video, current_video = run_inference(
            model=model,
            start_block_index=start_block_index,
            end_block_index=end_block_index,
            prompt=trajectory.prompt,
            multiple_trajectory=trajectory,
        )
        return all_video, current_video, end_block_index

    if mode == "Drag":
        end_block_index = start_block_index
        all_video, current_video = run_inference(
            model=model,
            start_block_index=start_block_index - 1,
            end_block_index=start_block_index,
            prompt=trajectory.prompt,
            multiple_trajectory=trajectory,
        )
        return all_video, current_video, end_block_index

    raise ValueError(f"Unknown mode: {mode!r}. Expected 'Animation' or 'Drag'.")


def save_videos(
    all_video: torch.Tensor,
    current_video: torch.Tensor,
    output_dir: Path | str,
    prompt_index: int,
    prompt: str,
    start_block_index: int,
    end_block_index: int,
    mode: str | None = None,
    fps: int = 8,
) -> tuple[str, str]:
    """
    Save current and (optionally) full video.

    Returns:
        (full_video_path, current_video_path).
        When start_block_index == 0, full_video_path equals current_video_path.
    """
    safe_prompt = (prompt or "no_prompt")[:50].replace(" ", "_")
    save_dir = Path(output_dir) / f"{prompt_index:04d}-{safe_prompt}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if mode is not None:
        save_prefix = f"block_{start_block_index}_{mode}_{end_block_index}"
    else:
        save_prefix = f"block_{start_block_index}_{end_block_index}"

    current_video_path = str(save_dir / f"{save_prefix}.mp4")
    write_video(current_video_path, current_video, fps=fps)

    if start_block_index > 0:
        if mode is not None:
            full_prefix = f"block_0_{start_block_index}_{mode}_{end_block_index}"
        else:
            full_prefix = f"block_0_{end_block_index}"
        full_video_path = str(save_dir / f"{full_prefix}.mp4")
        write_video(full_video_path, all_video, fps=fps)
    else:
        full_video_path = current_video_path

    return full_video_path, current_video_path


def generate_video(
    stream_inference_model: StreamInferenceWrapper,
    prompt_index: int,
    prompt: str,
    start_block_index: int,
    block_number: int,
    output_dir: str | Path,
) -> tuple[str, int]:
    """
    Generate video blocks without drag/animation optimization.
    """
    if start_block_index == 0:
        set_seed(stream_inference_model.seed)

    end_block_index = start_block_index + block_number
    with torch.no_grad():
        all_video, current_video = stream_inference_model.inference(
            start_block_index=start_block_index,
            end_block_index=end_block_index,
            prompt=prompt,
        )

    full_video_path, current_video_path = save_videos(
        all_video=all_video,
        current_video=current_video,
        output_dir=output_dir,
        prompt_index=prompt_index,
        prompt=prompt,
        start_block_index=start_block_index,
        end_block_index=end_block_index,
        mode=None,
        fps=8,
    )
    return full_video_path, end_block_index


def optimize_video(
    stream_drag_inference_model: StreamDragInferenceWrapper,
    output_dir: str | Path,
    prompt_index: int,
    start_block_index: int,
    multi_trajectory: MultiTrajectory,
) -> tuple[str, int]:
    """
    Run drag/animation optimization and save the resulting videos.
    """
    print(
        f"""
optimize_video
    {multi_trajectory = }
"""
    )

    all_video, current_video, end_block_index = run_optimization(
        model=stream_drag_inference_model,
        trajectory=multi_trajectory,
        start_block_index=start_block_index,
    )

    full_video_path, current_video_path = save_videos(
        all_video=all_video,
        current_video=current_video,
        output_dir=output_dir,
        prompt_index=prompt_index,
        prompt=multi_trajectory.prompt,
        start_block_index=start_block_index,
        end_block_index=end_block_index,
        mode=multi_trajectory.drag_or_animation_select,
        fps=8,
    )
    return full_video_path, end_block_index
