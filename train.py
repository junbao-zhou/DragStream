import argparse
import os
import time
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from trainer import (
    DiffusionTrainer,
    GANTrainer,
    ODETrainer,
    ScoreDistillationTrainer,
)

import torch.distributed as dist
import logging
from pathlib import Path
from utils.misc import format_dict


def _current_node_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() // dist.get_world_size()
    # fallback to env vars set by torchrun/launch
    return int(os.environ.get("NODE_RANK", 0))


def _current_process_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    # fallback to env vars set by torchrun/launch
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def _add_rank_to_record():
    factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.rank = _current_process_rank()
        return record

    logging.setLogRecordFactory(record_factory)


def _configure_logging(
    logdir: Path,
):
    """
    Hydra installs its own logging handlers before main() runs.
    This function forcefully replaces them so our file + format take effect.
    """
    _add_rank_to_record()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logfile = logdir / f"train-node{_current_node_rank()}-rank{_current_process_rank()}-{ts}.log"

    fmt = logging.Formatter(
        "[rank:{rank}] [{levelname}] [{asctime}] : {message}",
        style="{",
    )

    file_handler = logging.FileHandler(logfile, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    # Replace existing handlers (Hydra already configured them)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [file_handler, stream_handler]


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="rolling_forcing_dmd",
)
def main(
    config: DictConfig,
):
    # Strict mode: CLI override keys must already exist in cfg, otherwise error.
    OmegaConf.set_struct(config, True)

    config_name = HydraConfig.get().job.config_name

    # Keep paths stable even if Hydra changes working dir
    orig_cwd = hydra.utils.get_original_cwd()

    logdir = Path(config.logdir)
    if not logdir.is_absolute():
        logdir = Path(orig_cwd) / logdir
    logdir.mkdir(parents=True, exist_ok=True)

    _configure_logging(logdir)

    logging.info(f"""
{config_name = }
config = {format_dict(config)}
""")

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
