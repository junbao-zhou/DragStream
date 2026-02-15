import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from pathlib import Path
import time
from typing import Dict, List, Union
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

device = torch.device("cuda")


def get_video(video):
    video = rearrange(video, "b t c h w -> b t h w c").cpu()
    video = 255.0 * video
    assert video.shape[0] == 1
    video = video[0]
    return video


class StreamInferenceWrapper:
    def __init__(
        self,
        stream_model_config: DictConfig,
        checkpoint_path: str,
        total_generate_block_number: int,
        use_ema: bool = True,
        seed: int = 0,
    ):
        torch.set_grad_enabled(False)

        # Initialize pipeline
        assert hasattr(stream_model_config, "denoising_step_list")
        self.pipeline = CausalInferencePipeline(
            stream_model_config,
            device=device,
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.pipeline.generator.load_state_dict(
            state_dict[("generator" if not use_ema else "generator_ema")]
        )

        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        self.pipeline.text_encoder.to(device=gpu)
        self.pipeline.generator.to(device=gpu)
        if not stream_model_config.vae_offload_cpu:
            self.pipeline.vae.to(device=gpu)
        else:
            self.pipeline.vae.to(device="cpu")  # Offload VAE to CPU

        set_seed(seed)
        self.seed = seed
        self.initial_noise = torch.randn(
            [
                1,
                total_generate_block_number * self.pipeline.num_frame_per_block,
                16,
                60,
                104,
            ],
            device=device,
            dtype=torch.bfloat16,
        )

        self.recorded_latents = None
        self.video = None

        self.stream_model_config = stream_model_config

        print(
            f"""
{self.__class__.__name__}.__init__():
    {self.initial_noise.shape = }
{self.stream_model_config = }
"""
        )

    def block_to_latent_index(self, block_index: int) -> int:
        return block_index * self.pipeline.num_frame_per_block

    def latent_to_video_index(self, latent_frame_index: int) -> int:
        if latent_frame_index <= 0:
            return 0
        return (latent_frame_index - 1) * 4 + 1

    def block_to_video_index(self, block_index: int) -> int:
        return self.latent_to_video_index(
            self.block_to_latent_index(block_index)
        )

    def get_sampled_noise(
        self,
        start_block_index: int,
        end_block_index: int,
    ):
        current_start_latent_frame_index = self.block_to_latent_index(
            start_block_index
        )
        current_end_latent_frame_index = self.block_to_latent_index(
            end_block_index
        )
        print(
            f"{current_start_latent_frame_index = } | {current_end_latent_frame_index = }"
        )

        assert current_start_latent_frame_index < self.initial_noise.shape[1]
        assert current_end_latent_frame_index <= self.initial_noise.shape[1]
        sampled_noise = self.initial_noise[
            :,
            current_start_latent_frame_index:current_end_latent_frame_index,
            ...,
        ]
        return sampled_noise

    def get_initial_latents(
        self,
        start_block_index: int,
    ):
        if self.recorded_latents is None:
            return None
        print(f"{start_block_index = }")

        return self.recorded_latents[
            :,
            : self.block_to_latent_index(start_block_index),
        ]

    def decode_to_pixel(
        self,
        latents: torch.Tensor,
    ):
        start_decode_time = time.time()
        # Move VAE to GPU if offloaded
        if self.stream_model_config.vae_offload_cpu:
            self.pipeline.vae.to(device=gpu)
        video = self.pipeline.vae.decode_to_pixel(latents, use_cache=False)
        # Optionally move VAE back to CPU after decoding
        if self.stream_model_config.vae_offload_cpu:
            self.pipeline.vae.to(device="cpu")
        video = (video * 0.5 + 0.5).clamp(0, 1)
        print(
            f"{self.__class__.__name__}.decode_to_pixel() VAE decode time: {time.time() - start_decode_time:.2f} seconds"
        )
        return video

    def update_video(
        self,
        video: torch.Tensor,
        start_latent_frame_index: int,
    ):
        video = get_video(video)  # t, h, w, c
        start_video_frame_index = self.latent_to_video_index(
            start_latent_frame_index
        )
        if self.video is None:
            self.video = video
        else:
            self.video = self.video[:start_video_frame_index]
            self.video = torch.cat([self.video, video], dim=0)

    def decode_and_update_video(
        self,
        start_block_index: int,
        end_block_index: int,
    ):
        if start_block_index == 0:
            current_chunk_latent = self.recorded_latents[
                :,
                self.block_to_latent_index(
                    start_block_index
                ) : self.block_to_latent_index(end_block_index),
            ]
            current_chunk_video = self.decode_to_pixel(current_chunk_latent)
        else:
            current_chunk_latent = self.recorded_latents[
                :,
                self.block_to_latent_index(
                    start_block_index - 1
                ) : self.block_to_latent_index(end_block_index),
            ]
            current_chunk_video = self.decode_to_pixel(current_chunk_latent)
            current_chunk_video = current_chunk_video[:, 9:]
        self.update_video(
            current_chunk_video, self.block_to_latent_index(start_block_index)
        )

    def inference(
        self,
        start_block_index: int,
        end_block_index: int,
        prompt: str,
    ):
        assert start_block_index >= 0
        assert end_block_index > start_block_index
        print(
            f"""
{self.__class__.__name__}.inference():
    {start_block_index = }  |  {end_block_index = }
"""
        )
        sampled_noise = self.get_sampled_noise(
            start_block_index, end_block_index
        )
        prompts = [prompt]

        initial_latents = self.get_initial_latents(
            start_block_index,
        )
        if initial_latents is not None:
            print(f"{initial_latents.shape = }")

        latents_result = self.pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latents,
            do_not_decode_video=True,
            do_not_recompute_initial_latents=True,
        )
        latents = latents_result
        print(f"{latents.shape = }")
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
        print(f"{self.recorded_latents.shape = }")

        self.decode_and_update_video(start_block_index, end_block_index)

        return (
            self.video,
            self.video[self.block_to_video_index(start_block_index) :],
        )

    def reset(
        self,
    ):
        self.recorded_latents = None
        self.video = None
        # Clear VAE cache
        self.pipeline.vae.model.clear_cache()
        # Optionally move VAE back to CPU after reset if offloading
        if self.stream_model_config.vae_offload_cpu:
            self.pipeline.vae.to(device="cpu")

        if self.pipeline.is_kv_cache_initialized():
            self.pipeline._reset_kv_cache()
        if self.pipeline.is_crossattn_cache_initialized():
            self.pipeline._reset_crossattn_cache()


def main():

    output_block_number = 27

    config_dir = "configs"
    stream_config_name = "self_forcing_dmd_vsink_stream"

    data_path = "prompts/MovieGenVideoBench_extended.txt"

    seed = 42
    set_seed(seed)

    output_folder = "outputs-stream"
    output_folder = f"{output_folder}/blk{output_block_number}-{stream_config_name}-seed{seed}"

    print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
    # low_memory = get_cuda_free_memory_gb(gpu) < 40

    # Create dataset
    dataset = TextDataset(prompt_path=data_path)
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False
    )

    os.makedirs(output_folder, exist_ok=True)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path=config_dir):
        stream_config = compose(config_name=stream_config_name)
    print(f"{stream_config = }")

    stream_inference = StreamInferenceWrapper(
        stream_model_config=stream_config,
        checkpoint_path="./checkpoints/self_forcing_dmd.pt",
        total_generate_block_number=output_block_number,
        use_ema=True,
        seed=seed,
    )

    for i, batch_data in tqdm(enumerate(dataloader)):
        idx = batch_data["idx"].item()
        print(f"{idx = }")

        # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
        # Unpack the batch data for convenience
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]  # First (and only) item in the batch

        # For text-to-video, batch is just the text prompt
        prompt = batch["prompts"][0]
        print(f"{prompt = }")
        extended_prompt = (
            batch["extended_prompts"][0]
            if "extended_prompts" in batch
            else None
        )
        print(f"{extended_prompt = }")

        set_seed(seed)
        stream_inference.reset()

        current_block_index = 0
        block_step = 3
        while current_block_index < output_block_number:
            end_block_index = current_block_index + block_step
            all_video, current_video = stream_inference.inference(
                start_block_index=current_block_index,
                end_block_index=end_block_index,
                prompt=prompt,
            )

            # Save the video if the current prompt is not a dummy prompt
            if idx < num_prompts:
                current_video_output_path = os.path.join(
                    output_folder,
                    f"{idx:04d}-{prompt[:50].replace(' ', '_')}-{current_block_index:02d}-{end_block_index:02d}.mp4",
                )
                write_video(current_video_output_path, current_video, fps=16)
                all_video_output_path = os.path.join(
                    output_folder,
                    f"{idx:04d}-{prompt[:50].replace(' ', '_')}-{0:02d}-{end_block_index:02d}.mp4",
                )
                write_video(all_video_output_path, all_video, fps=16)

            current_block_index = end_block_index


if __name__ == "__main__":
    main()
