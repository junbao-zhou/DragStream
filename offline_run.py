import argparse
from pathlib import Path
from PIL import Image

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from torchvision.io import write_video
from torchvision import transforms

from optimize_utils import MultiTrajectory
from stream_drag_inference_wrapper import StreamDragInferenceWrapper
from utils.misc import set_seed
from video_operations import run_optimization, save_videos


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


def main() -> None:
    prompt_index = 4
    trajectory_dir = "./saved_labels/self_forcing_dmd_vsink_stream_drag-seed42/0004-A_close-up_3D_animated_scene_of_a_short,_fluffy_mo"
    first_frame_path = "1st_frames_dataset/832-480/4-A_close-up_3D_animated_scene_of_a_short,_fluffy_mo-0_ema.png"
    start_block_index = 3 if first_frame_path is None else 1
    trajectory_prefix = "block_3_Animation"
    config_dir = "configs"
    config_name = "self_forcing_dmd_vsink_stream_drag"
    checkpoint_path = "./checkpoints/self_forcing_dmd.pt"
    total_generate_block_number = 36
    seed = 42
    fps = 8
    output_dir = f"outputs-editing-{'i2v' if first_frame_path else ''}/self_forcing_dmd_vsink_stream_drag-seed42"
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
    if first_frame_path is not None:
        transform = transforms.Compose(
            [
                transforms.Resize((480, 832)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image = Image.open(first_frame_path).convert("RGB")
        image = transform(image)
        model.encode_image_and_update_recorded_latents(
            image=image,
            text_prompt=trajectory.prompt,
        )
    else:
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
