import argparse
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from torchvision.io import write_video

from optimize_utils import MultiTrajectory
from stream_drag_inference_wrapper import StreamDragInferenceWrapper
from utils.misc import set_seed


def build_stream_drag_inference(
    config_dir: str,
    config_name: str,
    checkpoint_path: str,
    total_generate_block_number: int,
    use_ema: bool,
    seed: int,
) -> StreamDragInferenceWrapper:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path=config_dir):
        stream_config = compose(config_name=config_name)

    return StreamDragInferenceWrapper(
        stream_model_config=stream_config,
        checkpoint_path=checkpoint_path,
        total_generate_block_number=total_generate_block_number,
        use_ema=use_ema,
        seed=seed,
    )


def run_optimization(
    model: StreamDragInferenceWrapper,
    trajectory: MultiTrajectory,
    start_block_index: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    mode = trajectory.drag_or_animation_select

    if mode == "Animation":
        end_block_index = start_block_index + int(trajectory.block_number)
        with torch.no_grad():
            all_video, current_video = model.inference(
                start_block_index=start_block_index,
                end_block_index=end_block_index,
                prompt=trajectory.prompt,
                multiple_trajectory=trajectory,
            )
        return all_video, current_video, end_block_index

    if start_block_index <= 0:
        raise ValueError("start_block_index must be > 0 for drag mode.")

    end_block_index = start_block_index
    with torch.no_grad():
        all_video, current_video = model.inference(
            start_block_index=start_block_index - 1,
            end_block_index=start_block_index,
            prompt=trajectory.prompt,
            multiple_trajectory=trajectory,
        )
    return all_video, current_video, end_block_index


def save_videos(
    all_video: torch.Tensor,
    current_video: torch.Tensor,
    output_dir: Path,
    prompt_index: int,
    prompt: str,
    start_block_index: int,
    end_block_index: int,
    mode: str,
    fps: int,
) -> Path:
    safe_prompt = (prompt or "no_prompt")[:50].replace(" ", "_")
    save_dir = output_dir / f"{prompt_index:04d}-{safe_prompt}"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_prefix = f"block_{start_block_index}_{mode}_{end_block_index}"
    current_video_path = save_dir / f"{save_prefix}.mp4"
    write_video(str(current_video_path), current_video, fps=fps)

    if start_block_index > 0:
        full_prefix = (
            f"block_0_{start_block_index}_{mode}_{end_block_index}"
        )
        full_video_path = save_dir / f"{full_prefix}.mp4"
        write_video(str(full_video_path), all_video, fps=fps)
        return full_video_path

    return current_video_path


def main() -> None:
    prompt_index = 4
    trajectory_dir = "./saved_labels/self_forcing_dmd_vsink_stream_drag-seed42/0004-A_close-up_3D_animated_scene_of_a_short,_fluffy_mo"
    start_block_index = 3
    trajectory_prefix = "block_3_Animation"
    config_dir = "configs"
    config_name = "self_forcing_dmd_vsink_stream_drag"
    checkpoint_path = "./checkpoints/self_forcing_dmd.pt"
    total_generate_block_number = 36
    seed = 42
    fps = 8
    output_dir = "outputs-editing/self_forcing_dmd_vsink_stream_drag-seed42"
    use_ema = True

    torch.set_grad_enabled(False)

    trajectory = MultiTrajectory.load(
        save_dir=trajectory_dir,
        prefix=trajectory_prefix,
    )

    model = build_stream_drag_inference(
        config_dir=config_dir,
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        total_generate_block_number=total_generate_block_number,
        use_ema=use_ema,
        seed=seed,
    )

    set_seed(seed)
    model.reset()
    model.inference(
        start_block_index=0,
        end_block_index=start_block_index,
        prompt=trajectory.prompt,
    )

    all_video, current_video, end_block_index = run_optimization(
        model=model,
        trajectory=trajectory,
        start_block_index=start_block_index,
    )

    saved_path = save_videos(
        all_video=all_video,
        current_video=current_video,
        output_dir=Path(output_dir),
        prompt_index=prompt_index,
        prompt=trajectory.prompt,
        start_block_index=start_block_index,
        end_block_index=end_block_index,
        mode=trajectory.drag_or_animation_select,
        fps=fps,
    )
    print(str(saved_path))


if __name__ == "__main__":
    main()
