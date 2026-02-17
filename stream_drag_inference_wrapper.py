import torch
import numpy as np
from omegaconf import DictConfig
from optimize_utils import MultiTrajectory
from stream_inference_wrapper import StreamInferenceWrapper


def _extract_block_trajectories(
    multi_traj: MultiTrajectory,
) -> tuple[
    list[list[dict[str, bool | list[torch.Tensor]]]],
    list[np.ndarray],
    np.ndarray | None,
]:
    """Extract block_trajectories from a MultiTrajectory in the format expected by begin_optimize.

    Returns:
        block_trajectories: block_num x N x trajectory dict
            Each trajectory dict has keys 'is_rotation', 'deltas', 'start_point',
            and optionally 'rotation_center'.
        masks: list of N masks corresponding to each trajectory
        movable_mask: the movable area mask for the whole image
    """
    if multi_traj.trajectories is None or len(multi_traj.trajectories) == 0:
        return [], [], None

    movable_mask = multi_traj.movable_mask

    # Collect per-trajectory masks
    masks = [traj.mask for traj in multi_traj.trajectories]

    # Find the maximum number of blocks across all trajectories
    max_blocks = (
        max(
            len(traj.block_trajectories)
            for traj in multi_traj.trajectories
            if traj.block_trajectories
        )
        if any(traj.block_trajectories for traj in multi_traj.trajectories)
        else 0
    )

    if max_blocks == 0:
        return [], masks, movable_mask

    block_trajectories = []
    for block_idx in range(max_blocks):
        block = []
        for traj in multi_traj.trajectories:
            if traj.block_trajectories and block_idx < len(traj.block_trajectories):
                block.append(traj.block_trajectories[block_idx])
            else:
                # Provide an empty placeholder
                block.append(
                    {
                        "is_rotation": False,
                        "deltas": [],
                        "start_point": (0, 0),
                    }
                )
        block_trajectories.append(block)

    # Assert: the N of each block in block_trajectories should equal the length of masks
    for block_idx, block in enumerate(block_trajectories):
        assert len(block) == len(masks), (
            f"Block {block_idx} has {len(block)} trajectories, " f"but there are {len(masks)} masks"
        )

    assert ((len(block_trajectories) == 0) and (movable_mask is None)) or (
        (len(block_trajectories) > 0) and (movable_mask is not None)
    ), "block_trajectories and movable_mask must both be present or both be absent"

    return block_trajectories, masks, movable_mask


class StreamDragInferenceWrapper(StreamInferenceWrapper):
    def __init__(
        self,
        stream_model_config: DictConfig,
        checkpoint_path: str,
        total_generate_block_number: int,
        use_ema: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            stream_model_config=stream_model_config,
            checkpoint_path=checkpoint_path,
            total_generate_block_number=total_generate_block_number,
            use_ema=use_ema,
            seed=seed,
        )
        self.previous_record_feature_list = None

    def inference(
        self,
        start_block_index: int,
        end_block_index: int,
        prompt: str,
        # below are for drag optimization
        multiple_trajectory: MultiTrajectory = None,
    ):
        assert start_block_index >= 0
        assert end_block_index > start_block_index
        print(f"""
{self.__class__.__name__}.inference():
    {start_block_index = }  |  {end_block_index = }
""")
        sampled_noise = self.get_sampled_noise(start_block_index, end_block_index)
        prompts = [prompt]

        # Extract block_trajectories, masks, and movable_mask from multiple_trajectory
        drag_optimize_target_latent_index = -1
        if multiple_trajectory is not None:
            block_trajectories, masks, movable_mask = _extract_block_trajectories(
                multiple_trajectory
            )
            assert multiple_trajectory.drag_or_animation_select in [
                "Drag",
                "Animation",
            ]
            if multiple_trajectory.drag_or_animation_select == "Drag":
                drag_optimize_target_latent_index = 2
        else:
            block_trajectories, masks, movable_mask = [], [], None

        if len(block_trajectories) > 0:
            is_drag_optimize = True
        else:
            is_drag_optimize = False

        initial_latents = self.get_initial_latents(
            start_block_index,
        )
        if initial_latents is not None:
            print(f"{initial_latents.shape = }")

        print(f"{block_trajectories = }")
        print(f"{len(masks) = }")
        latents_result = self.pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latents,
            do_not_decode_video=True,
            do_not_recompute_initial_latents=True,
            # below are for drag optimization
            model_config=self.stream_model_config,
            previous_record_feature_list=self.previous_record_feature_list,
            is_drag_optimize=is_drag_optimize,
            block_trajectories=block_trajectories,
            masks=masks,
            movable_mask=movable_mask,
            drag_optimize_target_latent_index=drag_optimize_target_latent_index,
        )
        if self.stream_model_config.drag_optim_config.record_feature_block_indexes:
            latents, record_attention_values_list = latents_result
        else:
            latents = latents_result
            record_attention_values_list = None
        if self.recorded_latents is None:
            self.recorded_latents = latents
        else:
            self.recorded_latents = torch.concat(
                [
                    self.recorded_latents[:, :0],
                    latents,
                ],
                dim=1,
            )

        if record_attention_values_list is not None:

            def dict_first_value(d: dict):
                return next(iter(d.values()))

            print(f"{record_attention_values_list.keys() = }")  # denoising timesteps
            print(
                f"{dict_first_value(record_attention_values_list).keys() = }"
            )  # attention block layers
            print(
                f"{dict_first_value(dict_first_value(record_attention_values_list)).keys() = }"
            )  # attention types name
            print(
                f"{dict_first_value(dict_first_value(dict_first_value(record_attention_values_list))).shape = }"
            )  # [1, 3, 30, 52, 1536]
        else:
            print(f"{record_attention_values_list = }")
        self.previous_record_feature_list = record_attention_values_list

        self.decode_and_update_video(start_block_index, end_block_index)

        return (
            self.video,
            self.video[self.block_to_video_index(start_block_index) :],
        )

    def reset(
        self,
    ):
        super().reset()
        self.previous_record_feature_list = None
