import gc
import random
import time
from typing import List, Optional
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from optimize_utils import transpose_dict_2d
from tensor_utils import (
    build_anisotropic_gaussian_from_mask,
    combine_gaussian_maps,
    combine_masks_or,
    normalize_tensor_to_match_tensor,
    resize_tensor,
    warp_tensor,
    warp_tensor_sequence,
)
from utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)

from demo_utils.memory import (
    gpu,
    get_cuda_free_memory_gb,
    DynamicSwapInstaller,
    move_model_to_device_with_memory_preservation,
)

IMAGE_HEIGHT = 480.0


def split_trajectories_segments(
    trajectories: list[dict[str, bool | list[torch.Tensor]]],
    translation_step: float,
    rotation_step: float,
) -> List[list[dict[str, bool | list[torch.Tensor]]]]:
    """
    Split drag trajectories into evenly spaced intermediate segments for
    progressive (coarse-to-fine) optimization.

    Given N trajectories (each with per-frame deltas), this function:
      1. Determines the maximum number of segments needed across all
         trajectories based on the magnitude of their deltas and the
         provided step sizes.
      2. Divides every trajectory's deltas uniformly into that many segments.
      3. Produces a list of cumulative intermediate trajectory snapshots,
         where segment k contains deltas scaled by (k / max_segments).

    :param trajectories:
        N x trajectory dicts.
        Each dict has keys:
          - 'is_rotation' (bool): Whether this trajectory is a rotation.
          - 'deltas' (list): Per-frame displacement values.
              For translation: each delta is a 2D vector (dx, dy).
              For rotation: each delta is a scalar angle.
          - 'start_point': The starting pixel coordinate of the drag.
          - 'rotation_center' (only if is_rotation): The center of rotation.

    :param translation_step:
        The pixel distance that defines one segment for translation
        trajectories. Larger values produce fewer segments.

    :param rotation_step:
        The angle (in the same units as deltas) that defines one segment
        for rotation trajectories. Larger values produce fewer segments.

    :returns:
        segment_num x N x trajectory dicts.
        A list of length `max_segment_number`, where each element is a list
        of N trajectory dicts. The k-th element (1-indexed) contains
        trajectories whose deltas are scaled to (k / max_segment_number)
        of the original deltas — i.e., cumulative intermediate waypoints.
    """
    # -------------------------------------------------------------------------
    # Phase 1: Convert raw deltas to torch tensors (ensure uniform type)
    # -------------------------------------------------------------------------
    for trajectory in trajectories:
        trajectory["deltas"] = [
            torch.tensor(delta, device="cpu") for delta in trajectory["deltas"]
        ]

    # -------------------------------------------------------------------------
    # Phase 2: Determine the maximum number of segments across all trajectories.
    #   - For rotations: segment count = |angle_delta| // rotation_step
    #   - For translations: segment count = ||displacement_delta||₂ // translation_step
    #   - We take the global maximum so every trajectory is split into the
    #     same number of segments (ensuring synchronized progressive steps).
    # -------------------------------------------------------------------------
    max_segment_number = 1  # at least one segment
    for trajectory in trajectories:
        print(f"{trajectory['is_rotation'] = }")
        for delta in trajectory["deltas"]:
            if trajectory["is_rotation"]:
                magnitude = abs(delta)
                step = rotation_step
            else:
                magnitude = abs(torch.norm(delta))
                step = translation_step
            segment_number = int(magnitude // step)
            print(f"{delta = } {magnitude = } {segment_number = }")
            max_segment_number = max(max_segment_number, segment_number)
    print(f"{max_segment_number = }")

    # -------------------------------------------------------------------------
    # Phase 3: Compute per-segment step sizes for each trajectory.
    #   Each trajectory's deltas are divided by max_segment_number to get
    #   the uniform per-segment increment.
    # -------------------------------------------------------------------------
    split_trajectory_steps = []
    for trajectory in trajectories:
        print(f"{trajectory['is_rotation'] = }")
        # Divide each frame's delta by the total number of segments
        trajectory_steps = [
            delta / float(max_segment_number) for delta in trajectory["deltas"]
        ]
        print(f"{trajectory_steps = }")
        # Build the per-trajectory step metadata
        split_trajectory_step = {
            "is_rotation": trajectory["is_rotation"],
            "steps": trajectory_steps,  # per-segment increment per frame
            "start_point": trajectory["start_point"],
        }
        if trajectory["is_rotation"]:
            split_trajectory_step["rotation_center"] = trajectory[
                "rotation_center"
            ]
        split_trajectory_steps.append(split_trajectory_step)

    # -------------------------------------------------------------------------
    # Phase 4: Build cumulative intermediate trajectory lists.
    #   For segment_index k (1-indexed from 1 to max_segment_number):
    #     delta_k = step * k
    #   This produces progressively larger displacements, enabling the
    #   optimizer to move features gradually toward the final target.
    # -------------------------------------------------------------------------
    new_trajectories_list = []
    for segment_index in range(max_segment_number):
        segment_index += 1  # 1-indexed: cumulative scale factor
        new_trajectories = []
        for trajectory_step in split_trajectory_steps:
            new_trajectory = {
                "is_rotation": trajectory_step["is_rotation"],
                "deltas": [
                    step * segment_index for step in trajectory_step["steps"]
                ],
                "start_point": trajectory_step["start_point"],
            }
            if trajectory_step["is_rotation"]:
                new_trajectory["rotation_center"] = trajectory_step[
                    "rotation_center"
                ]
            new_trajectories.append(new_trajectory)
        print(f"{new_trajectories = }")
        new_trajectories_list.append(new_trajectories)

    # Return: list of length max_segment_number, each containing N trajectory dicts
    # with cumulatively scaled deltas (segment 1 = smallest, last = full original delta)
    return new_trajectories_list


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = (
            WanDiffusionWrapper(
                **getattr(args, "model_kwargs", {}),
                is_causal=True,
            )
            if generator is None
            else generator
        )
        self.text_encoder = (
            WanTextEncoder() if text_encoder is None else text_encoder
        )
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long
        )
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[
                1000 - self.denoising_step_list
            ]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def stack_features(
        self,
        record_features: dict[int, torch.Tensor],
        # dict[block_index] -> Tensor[1, 3, 30, 52, 1536]
    ):
        """
        Reorder and stack per-block attention features into one tensor.

        Input:
            record_features: Mapping `block_index -> Tensor` with shape
                `[1, 3, 30, 52, 1536]`.

        Returns:
            Tensor with shape `[1536 * L, 3, 30, 52]`,
            where `L = len(record_features)`.
        """
        attention_values = {
            k: v.permute(0, 4, 1, 2, 3).contiguous()
            for k, v in record_features.items()
        }
        # [ [1, 1536, 3, 30, 52] ]

        features = []
        for fet in attention_values.values():
            # [1536, 3, 30, 52]
            features.append(fet.squeeze(0).contiguous())
        del fet

        # Concatenate all features
        feature = torch.cat(features, dim=0)
        # [1536 * L, 3, 30, 52], L is the number of blocks
        return feature

    def generate_features(
        self,
        latents,
        conditional_dict,
        timestep,
        kv_cache,
        crossattn_cache,
        current_start,
        # below are for drag optimization
        model_config: DictConfig = None,
    ):
        """
        Run one generator forward pass and return prediction + stacked features.

        Notes:
        - KV cache is deep-cloned/detached before forward to avoid mutating
          the caller's cache during optimization.
        - Forward runs under CUDA bfloat16 autocast.
        - Returned `record_features` are converted via `stack_features(...)`.

        Returns:
            denoised_pred:
                Model denoised prediction tensor.
            record_features:
                Dict `variant_key -> Tensor[1536 * L, 3, 30, 52]`. L is the number of blocks.
        """
        temp_kv_cache = [
            {
                "k": kv_cache[block_index]["k"].clone().detach(),
                "v": kv_cache[block_index]["v"].clone().detach(),
                "global_end_index": kv_cache[block_index]["global_end_index"]
                .clone()
                .detach(),
                "local_end_index": kv_cache[block_index]["local_end_index"]
                .clone()
                .detach(),
            }
            for block_index in range(self.num_transformer_blocks)
        ]
        # print(f"{temp_kv_cache[0]['k'].shape = }")
        # Forward pass through the transformer with user-specified autocast dtype
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Extract features during forward pass
            _, denoised_pred = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=temp_kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                # below are for drag optimization
                model_config=model_config,
            )
            denoised_pred, record_features = denoised_pred
        # record_features: dict[block_index] -> dict[key -> Tensor] [1, 3, 30, 52, 1536]

        record_features = transpose_dict_2d(record_features)
        # record_features: Dict[key] -> Dict[block_index -> Tensor] [1, 3, 30, 52, 1536]
        record_features = {
            k: self.stack_features(v) for k, v in record_features.items()
        }
        # record_features: Dict[key] -> Tensor [1536 * L, 3, 30, 52]
        return denoised_pred, record_features

    @staticmethod
    def precompute_conditions(
        trajectories: list[dict[str, bool | list[torch.Tensor]]],
        # N x trajectory
        masks: list[np.ndarray],  # N x mask
        dtype: torch.dtype,
        device: torch.device,
        model_config: DictConfig,
        downsample_movable_mask: torch.Tensor,
        previous_record_features: dict[int, dict[str, torch.Tensor]],
        latent_spatial_size: tuple[int, int],  # (H_lat, W_lat)
    ):
        with torch.no_grad():
            _any_prev_feat = next(iter(previous_record_features.values()))
            feat_spatial_size = tuple(_any_prev_feat.shape[-2:])  # (Hf, Wf)

            # 1) Warped masks in image space
            warped_masks = [
                warp_tensor_sequence(
                    tensor=torch.tensor(mask, device=device).float(),
                    is_rotation=trajectory["is_rotation"],
                    deltas=trajectory["deltas"],
                    rotation_center=trajectory.get("rotation_center", None),
                    original_height=IMAGE_HEIGHT,
                    mode="nearest",
                    cumulative=False,
                )
                for trajectory, mask in zip(trajectories, masks)
            ]  # N x frame x [H_img, W_img]
            # print(f"{len(warped_masks) = }, {len(warped_masks[0]) = }, {warped_masks[0][0].shape = }")

            # 2) Downsampled warped masks in feature space
            down_warp_masks = [
                [
                    resize_tensor(
                        warped_mask.detach(),
                        size=feat_spatial_size,
                        mode="nearest",
                    ).detach()
                    for warped_mask in traj_warped_masks
                ]
                for traj_warped_masks in warped_masks
            ]  # N x frame x [Hf, Wf]

            # 3) Gaussian heatmaps per trajectory
            gaussian_heatmaps_per_traj = [
                [
                    build_anisotropic_gaussian_from_mask(
                        warped_mask,
                        padding_scale=model_config.drag_optim_config.gradient_gaussian_padding,
                        sigma_scale=model_config.drag_optim_config.gradient_gaussian_sigma,
                    ).detach()
                    for warped_mask in traj_warped_masks
                ]
                for traj_warped_masks in warped_masks
            ]  # N x frame x  [H_img, W_img]

            # 4) Combined downsampled movable mask (OR of all warped + original)
            all_down_warp_masks = [
                dwm
                for traj_down_masks in down_warp_masks
                for dwm in traj_down_masks
            ]
            all_down_warp_masks.append(downsample_movable_mask.clone())
            combined_downsample_movable_mask = combine_masks_or(
                all_down_warp_masks
            )

            # 5) Precompute warped attention values per variant
            warped_for_prev: dict[str | int, List[List[torch.Tensor]]] = {
                key: [
                    warp_tensor_sequence(
                        tensor=prev_feat.to(dtype=dtype, device=device),
                        is_rotation=trajectory["is_rotation"],
                        deltas=[
                            d.to(
                                dtype=dtype,
                                device=device,
                            )
                            for d in trajectory["deltas"]
                        ],
                        rotation_center=trajectory.get("rotation_center", None),
                        original_height=IMAGE_HEIGHT,
                        mode="nearest",
                        cumulative=False,
                    )
                    for trajectory in trajectories
                ]
                for key, prev_feat in previous_record_features.items()
            }
            # warped_for_prev: dict[key] -> list[traj_index] -> list[frame_index] -> Tensor [1536 * L, 30 * scaling, 52 * scaling]

            # 6) Combined Gaussian heatmaps in latent space
            combined_gaussian_heatmaps = None
            num_frames = len(gaussian_heatmaps_per_traj[0])
            num_trajs = len(gaussian_heatmaps_per_traj)
            combined_gaussian_heatmaps = torch.stack(
                [
                    combine_gaussian_maps(
                        [
                            gaussian_heatmaps_per_traj[traj_idx][frame_idx]
                            for traj_idx in range(num_trajs)
                        ]
                    )  # [H_img, W_img]
                    for frame_idx in range(num_frames)
                ],
                dim=0,
            ).to(
                device=device
            )  # [F, H_img, W_img]

            combined_gaussian_heatmaps = resize_tensor(
                combined_gaussian_heatmaps,
                size=latent_spatial_size,
                mode="bilinear",
            ).detach()  # [F, H_lat, W_lat]
            combined_gaussian_heatmaps = combined_gaussian_heatmaps.to(
                dtype=dtype
            )
        return (
            warped_masks,
            down_warp_masks,
            gaussian_heatmaps_per_traj,
            combined_downsample_movable_mask,
            warped_for_prev,
            combined_gaussian_heatmaps,
        )

    def optimize_latent(
        self,
        latents,
        conditional_dict,
        timestep,
        kv_cache,
        crossattn_cache,
        current_start,
        # below are for drag optimization
        trajectories: list[dict[str, bool | list[torch.Tensor]]],
        # N x trajectory,
        # trajectory has keys 'is_rotation' 'deltas' 'start_point'
        # if is_rotation: trajectory also has 'rotation_center'
        masks: list[np.ndarray],  # N x mask
        movable_mask: np.ndarray,
        clean_previous_record_feature: dict[int, dict[str, torch.Tensor]],
        # dict[block_index] -> dict[key -> Tensor] [1, 3, 30, 52, 1536]
        noisy_previous_record_feature: dict[int, dict[str, torch.Tensor]],
        # dict[block_index] -> dict[key -> Tensor] [1, 3, 30, 52, 1536]
        model_config: DictConfig,
        optimize_target_latent_index: int = -1,
    ):
        """
        :param trajectories:
            N x trajectory,
            trajectory has keys 'is_rotation' 'deltas' 'start_point'
            if is_rotation: trajectory also has 'rotation_center'
        :param masks:
            N x mask
        """
        assert isinstance(model_config.drag_optim_config.optimize_iter, int)
        assert isinstance(model_config.drag_optim_config.optimize_lr, float)
        assert (
            len(model_config.drag_optim_config.record_feature_block_indexes) > 0
        )
        assert len(trajectories) == len(masks)
        if len(trajectories) == 0:
            return latents

        print(f"{trajectories = }")
        print(f"{len(masks) = }")

        original_latents = latents.clone().detach()

        original_denoised_pred = self.generate_features(
            latents=latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            # below are for drag optimization
            model_config=model_config,
        )
        original_denoised_pred, _ = original_denoised_pred

        # Check if optimization is enabled
        if model_config.drag_optim_config.optimize_iter <= 0:
            return latents
        # Cast latents to configured dtype for optimization
        latent_original_dtype = latents.dtype
        latents = latents.to(dtype=torch.bfloat16)
        timestep_original_dtype = timestep.dtype
        timestep = timestep.to(dtype=latents.dtype)

        for param in self.generator.parameters():
            generator_original_dtype = param.dtype
            break
        self.generator = self.generator.to(dtype=latents.dtype)
        # self.generator.train(True)
        for param in self.generator.parameters():
            param.requires_grad = False

        split_trajectories_list = split_trajectories_segments(
            trajectories=trajectories,
            translation_step=model_config.drag_optim_config.translation_step,
            rotation_step=model_config.drag_optim_config.rotation_step,
        )
        # split_trajectories_list: list[segment_index] -> list[trajectory_index] -> trajectory dict

        def _select_variant(feat):
            if isinstance(feat, dict):
                keys = list(feat.keys())
                if not keys:
                    raise ValueError("Empty feature dict provided.")
                non_orig = [k for k in keys if str(k) != "original"]
                key = random.choice(keys)
                print(f"Selected feature variant {key = } from {keys = }")
                # key = "original" if "original" in feat else keys[0]
                return feat[key]
            return feat

        def get_previous_last(
            prev: dict[int, dict[str, torch.Tensor]],
            # dict[block_index] -> dict[key -> Tensor] [1, 3, 30, 52, 1536]
        ) -> dict[str, torch.Tensor]:
            out = transpose_dict_2d(prev)
            # out: dict[key] -> dict[block_index -> Tensor] [1, 3, 30, 52, 1536]
            out = {k: self.stack_features(v) for k, v in out.items()}
            # out: dict[key] -> Tensor [1536 * L, 3, 30, 52]
            out = {k: v[:, -1, ...].detach() for k, v in out.items()}
            # out: dict[key] -> Tensor [1536 * L, 30, 52]
            out = {
                k: resize_tensor(
                    v.detach(),
                    scale_factor=model_config.drag_optim_config.feature_scaling_factor,
                    mode="bilinear",
                ).detach()
                for k, v in out.items()
            }
            # out: dict[key] -> Tensor [1536 * L, 30 * scaling, 52 * scaling]
            return out

        previous_record_features: dict[str, torch.Tensor] = get_previous_last(
            noisy_previous_record_feature
        )
        # previous_record_features: dict[key] -> Tensor [1536 * L, 30 * scaling, 52 * scaling]

        movable_mask_torch = torch.tensor(
            movable_mask, device=latents.device
        ).float()
        downsample_movable_mask = resize_tensor(
            movable_mask_torch.detach(),
            size=tuple(original_denoised_pred.shape[-2:]),
            mode="nearest",
        ).detach()
        # print(f"{downsample_movable_mask.shape = }") # [60, 104]

        with torch.enable_grad():

            latents.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [latents],
                lr=model_config.drag_optim_config.optimize_lr,
            )

            for split_traj_idx, split_trajectories in enumerate(
                split_trajectories_list
            ):
                # split_trajectories: N x trajectory, list[trajectory_index] -> trajectory dict
                (
                    warped_masks,
                    down_warp_masks,
                    gaussian_heatmaps_per_traj,
                    combined_downsample_movable_mask,
                    warped_previous_record_features,
                    combined_gaussian_heatmaps,
                ) = CausalInferencePipeline.precompute_conditions(
                    trajectories=split_trajectories,
                    masks=masks,
                    dtype=latents.dtype,
                    device=latents.device,
                    model_config=model_config,
                    downsample_movable_mask=downsample_movable_mask,
                    previous_record_features=previous_record_features,
                    latent_spatial_size=tuple(latents.shape[-2:]),
                )
                # -------------------------
                # Optimization iterations (reuse precomputed items)
                # -------------------------
                for optimize_iter_idx in range(
                    model_config.drag_optim_config.optimize_iter
                ):
                    print(f"{optimize_iter_idx = }")
                    print(f"{latents.mean((0, 2, 3, 4)) = }")
                    print(f"{latents.std((0, 2, 3, 4)) = }")

                    denoised_pred, record_features = self.generate_features(
                        latents=latents,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                        model_config=model_config,
                    )
                    # denoised_pred.shape [1, 3, 16, 60, 104]
                    # record_features: dict[key] -> Tensor [1536 * L, 3, 30, 52]

                    print(f"selecting features for optimization")
                    record_features_selected = _select_variant(
                        record_features,
                    )
                    # print(f"{record_features_selected.shape = }")
                    # record_features_selected: Tensor [1536 * L, 3, 30, 52]
                    del record_features

                    if optimize_target_latent_index >= 0:
                        record_features_selected = record_features_selected[
                            :,
                            optimize_target_latent_index : optimize_target_latent_index
                            + 1,
                        ]

                    record_features_selected = resize_tensor(
                        record_features_selected,
                        scale_factor=model_config.drag_optim_config.feature_scaling_factor,
                        mode="bilinear",
                    )
                    # print(
                    #     f"{record_features_selected.shape = }"
                    # )  # [1536 * L, 3, 30 * scaling, 52 * scaling]

                    print(f"selecting warped previous features")
                    warped_previous_feature_selected = _select_variant(
                        warped_previous_record_features,
                    )  # list[traj_index] -> list[frame_index] -> Tensor [1536 * L, 30 * scaling, 52 * scaling]

                    loss = 0
                    loss_cnt = 0

                    # Iterate over each trajectory point
                    for trajectory_index, trajectory in enumerate(
                        split_trajectories
                    ):
                        assert record_features_selected.shape[1] == len(
                            trajectory["deltas"]
                        )

                        for frame_index in range(len(trajectory["deltas"])):
                            warped_attention_values = (
                                warped_previous_feature_selected[
                                    trajectory_index
                                ][frame_index]
                            )

                            downsample_warped_mask = down_warp_masks[
                                trajectory_index
                            ][frame_index]

                            pixel_wise_loss = F.mse_loss(
                                warped_attention_values
                                * downsample_warped_mask,
                                record_features_selected[:, frame_index]
                                * downsample_warped_mask,
                                reduction="none",
                            ).mean(dim=0)
                            # print(f"{pixel_wise_loss.shape = }")  # [60, 104]

                            # Add weighted loss
                            loss = (
                                loss
                                + (
                                    downsample_warped_mask * pixel_wise_loss
                                ).sum()
                            )
                            loss_cnt += downsample_warped_mask.sum()

                    print(f"{loss = }  /  {loss_cnt = }")
                    loss = loss / max(1e-8, loss_cnt)
                    print(f"{loss = }")

                    unchanged_mask = 1.0 - combined_downsample_movable_mask
                    unchanged_loss = F.mse_loss(
                        denoised_pred * unchanged_mask.detach(),
                        original_denoised_pred.detach()
                        * unchanged_mask.detach(),
                    )
                    print(f"{unchanged_loss = }")
                    loss = loss + unchanged_loss * 1.0

                    # Update latents
                    self.generator.zero_grad()
                    optimizer.zero_grad()
                    if loss_cnt > 0:
                        loss.backward()
                        assert (
                            combined_gaussian_heatmaps.shape[0] == 1
                            or combined_gaussian_heatmaps.shape[0]
                            == latents.shape[-4]
                        )
                        assert (
                            combined_gaussian_heatmaps.shape[-2:]
                            == latents.shape[-2:]
                        )
                        latents.grad.mul_(
                            combined_gaussian_heatmaps[:, None, :, :]
                        )
                        # Clip gradients
                        clip_grad_norm_(
                            [latents],
                            max_norm=1.0,
                            norm_type=2,
                        )
                        optimizer.step()
                    if (
                        model_config.drag_optim_config.normalize_latent_after_drag_optimize
                    ):
                        print(f"Normalizing latents after optimize iteration")
                        latents = (
                            normalize_tensor_to_match_tensor(
                                latents.detach().clone(),
                                dim=(0, 3, 4),
                                reference_tensor=original_latents.to(
                                    dtype=latents.dtype
                                ),
                            )
                            .detach()
                            .clone()
                        )
                        # latents = latents.clamp(
                        #     min=latents_min,
                        #     max=latents_max,
                        # ).detach().clone()
                        latents.requires_grad_(True)
                        optimizer = torch.optim.AdamW(
                            [latents],
                            lr=model_config.drag_optim_config.optimize_lr,
                        )
                    # Clean up to save memory
                    gc.collect()
                    torch.cuda.empty_cache()

        latents = latents.detach().requires_grad_(False)

        if model_config.drag_optim_config.normalize_latent_after_post_merge:
            latents = (
                normalize_tensor_to_match_tensor(
                    latents,
                    dim=None,
                    reference_tensor=original_latents,
                )
                .detach()
                .clone()
            )

        # Convert back to original dtype
        self.generator = self.generator.to(dtype=generator_original_dtype)
        self.generator.train(False)
        latents = latents.to(dtype=latent_original_dtype)
        timestep = timestep.to(dtype=timestep_original_dtype)

        # Detach latents and remove gradient
        latents = latents.detach().requires_grad_(False)
        return latents

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        do_not_decode_video: bool = False,
        do_not_recompute_initial_latents: bool = False,
        # below are for drag optimization
        model_config: DictConfig = None,
        previous_record_feature_list: dict[
            int, dict[int, dict[str, torch.Tensor]]
        ] = None,
        # dict[denoising_step] -> dict[block_index] -> dict[key -> Tensor] [1, 3, 30, 52, 1536]
        is_drag_optimize: bool = False,
        block_trajectories: list[
            list[dict[str, bool | list[torch.Tensor]]]
        ] = [],
        masks: list[np.ndarray] = [],
        movable_mask: np.ndarray = None,
        drag_optimize_target_latent_index: int = -1,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            :param block_trajectories:
                block_num x N x trajectory,
                trajectory has keys 'is_rotation' 'deltas' 'start_point'
                if is_rotation: trajectory also has 'rotation_center'
            :param masks:
                N x mask
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (
            self.independent_first_frame and initial_latent is not None
        ):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = (
            initial_latent.shape[1] if initial_latent is not None else 0
        )
        num_output_frames = (
            num_frames + num_input_frames
        )  # add the initial latent frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            if do_not_recompute_initial_latents:
                pass
            else:
                print(f"Resetting caches")
                self._reset_crossattn_cache()
                self._reset_kv_cache()

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = (
                torch.ones(
                    [batch_size, 1], device=noise.device, dtype=torch.int64
                )
                * 0
            )
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (
                    num_input_frames - 1
                ) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                if do_not_recompute_initial_latents:
                    pass
                else:
                    print(f"Recompute KV cache based on Initial Latents")
                    self.generator(
                        noisy_image_or_video=initial_latent[:, :1],
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame
                        * self.frame_seq_length,
                    )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[
                    :,
                    current_start_frame : current_start_frame
                    + self.num_frame_per_block,
                ]
                output[
                    :,
                    current_start_frame : current_start_frame
                    + self.num_frame_per_block,
                ] = current_ref_latents
                if do_not_recompute_initial_latents:
                    pass
                else:
                    print(f"Recompute KV cache based on Initial Latents")
                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame
                        * self.frame_seq_length,
                    )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_chunk_index, current_num_frames in enumerate(
            tqdm(all_num_frames),
            start=num_input_blocks if initial_latent is not None else 0,
        ):
            print(f"\n{current_chunk_index = } ; {current_start_frame = }")
            if profile:
                block_start.record()

            noisy_input = noise[
                :,
                current_start_frame
                - num_input_frames : current_start_frame
                + current_num_frames
                - num_input_frames,
            ]

            if model_config is not None and OmegaConf.select(
                model_config, "drag_optim_config.record_feature_block_indexes"
            ):
                record_attention_values_list = {}
            # Step 3.1: Spatial denoising loop
            for time_step_index, current_timestep in enumerate(
                self.denoising_step_list
            ):
                print(f"{time_step_index = } ; {current_timestep = }")
                # set current timestep
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if (
                    is_drag_optimize
                    and time_step_index
                    in model_config.drag_optim_config.optimize_denoising_steps_indexes
                ):

                    noisy_input = self.optimize_latent(
                        latents=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame
                        * self.frame_seq_length,
                        # below are for drag optimization
                        trajectories=block_trajectories[
                            current_chunk_index - num_input_blocks
                        ],
                        masks=masks,
                        movable_mask=movable_mask,
                        clean_previous_record_feature=previous_record_feature_list[
                            -1
                        ],
                        noisy_previous_record_feature=previous_record_feature_list[
                            time_step_index
                        ],
                        model_config=model_config,
                        optimize_target_latent_index=drag_optimize_target_latent_index,
                    )
                    print(f"{noisy_input.mean() = }")
                    print(f"{noisy_input.std() = }")

                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    model_config=model_config,
                )
                if model_config is not None and OmegaConf.select(
                    model_config,
                    "drag_optim_config.record_feature_block_indexes",
                ):
                    denoised_pred, record_features = denoised_pred
                    if (
                        time_step_index
                        in model_config.drag_optim_config.optimize_denoising_steps_indexes
                    ):
                        record_attention_values_list[time_step_index] = (
                            record_features
                        )

                if (
                    model_config is not None
                    and OmegaConf.select(
                        model_config,
                        "drag_optim_config.dynamic_chunk_normalization_block_number",
                        default=0,
                    )
                    > 0
                ):
                    num_norm_blocks = (
                        model_config.drag_optim_config.dynamic_chunk_normalization_block_number
                    )
                    # Exclude the first chunk (independent first frame latent) by starting no earlier than num_frame_per_block
                    dynamic_normalize_start_frame_index = max(
                        self.num_frame_per_block,
                        (current_chunk_index - num_norm_blocks)
                        * self.num_frame_per_block,
                    )
                    # print(f"{dynamic_normalize_start_frame_index = }")
                    if (
                        dynamic_normalize_start_frame_index
                        < current_start_frame
                    ):
                        reference_tensor = torch.cat(
                            [
                                output[
                                    :,
                                    dynamic_normalize_start_frame_index:current_start_frame,
                                ],
                                denoised_pred,
                            ],
                            dim=1,
                        )
                        denoised_pred = normalize_tensor_to_match_tensor(
                            denoised_pred,
                            dim=None,
                            reference_tensor=reference_tensor,
                        )
                        # print(f"{denoised_pred.mean() = }")
                        # print(f"{denoised_pred.std() = }")

                if time_step_index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[
                        time_step_index + 1
                    ]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])

            # Step 3.2: record the model's output
            output[
                :,
                current_start_frame : current_start_frame + current_num_frames,
            ] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = (
                torch.ones_like(timestep) * self.args.context_noise
            )
            _, denoised_pred = self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                model_config=model_config,
            )
            if model_config is not None and OmegaConf.select(
                model_config, "drag_optim_config.record_feature_block_indexes"
            ):
                denoised_pred, record_features = denoised_pred
                record_attention_values_list[-1] = record_features

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        if not do_not_decode_video:
            start_decode_time = time.time()
            video = self.vae.decode_to_pixel(output, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            print(
                f"{self.__class__.__name__}.inference() VAE decode time: {time.time() - start_decode_time:.2f} seconds"
            )

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(
                f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)"
            )
            print(
                f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)"
            )
            for i, block_time in enumerate(block_times):
                print(
                    f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)"
                )
            print(
                f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)"
            )
            print(f"  - Total time: {total_time:.2f} ms")

        return_values = []
        if not do_not_decode_video:
            return_values.append(video)
        if return_latents:
            return_values.append(output)
        if model_config is not None and OmegaConf.select(
            model_config, "drag_optim_config.record_feature_block_indexes"
        ):
            return_values.append(record_attention_values_list)

        if len(return_values) == 0:
            return
        elif len(return_values) == 1:
            return return_values[0]
        else:
            return tuple(return_values)

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        print(
            f"""
{type(self).__name__}._initialize_kv_cache
    {batch_size = }
    {dtype = }
    {device = }
"""
        )
        kv_cache1 = []
        if self.local_attn_size != -1:
            print(f"use {self.local_attn_size = }")
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760
        print(f"{kv_cache_size = }")

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(
        self,
        batch_size,
        dtype,
        device,
    ):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        print(
            f"""
{type(self).__name__}._initialize_crossattn_cache
    {batch_size = }
    {dtype = }
    {device = }
"""
        )
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def _reset_crossattn_cache(self):
        # reset cross attn cache
        print(f"{type(self).__name__}._reset_crossattn_cache")
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False

    def _reset_kv_cache(self):
        # reset kv cache
        print(f"{type(self).__name__}._reset_kv_cache")
        for block_index in range(len(self.kv_cache1)):
            self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                [0],
                dtype=torch.long,
                device=self.kv_cache1[block_index]["global_end_index"].device,
            )
            self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                [0],
                dtype=torch.long,
                device=self.kv_cache1[block_index]["local_end_index"].device,
            )

    def is_kv_cache_initialized(self):
        return hasattr(self, "kv_cache1") and self.kv_cache1 is not None

    def is_crossattn_cache_initialized(self):
        return (
            hasattr(self, "crossattn_cache")
            and self.crossattn_cache is not None
        )
