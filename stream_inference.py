import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tqdm import tqdm
from torchvision.io import write_video
from torch.utils.data import DataLoader, SequentialSampler

from stream_inference_wrapper import StreamInferenceWrapper
# from stream_drag_inference_wrapper import StreamDragInferenceWrapper
from utils.dataset import TextDataset
from utils.misc import set_seed
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from demo_utils.memory import gpu, get_cuda_free_memory_gb


def main():

    output_block_number = 27

    config_dir = "configs"
    stream_config_name = "self_forcing_dmd_vsink_stream"
    # stream_config_name = "self_forcing_dmd_vsink_stream_drag"

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
