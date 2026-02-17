<p align="center">
<h1 align="center">DragStream</h1>
<h3 align="center">Streaming Drag-Oriented Interactive Video Manipulation: Drag Anything, Anytime!</h3>
</p>
<p align="center">
  <p align="center">
    <a>Junbao Zhou</a><sup>1</sup>
    ·
    <a>Yuan Zhou</a><sup>1</sup>
    ·
    <a>Kesen Zhao</a><sup>2</sup>
    ·
    <a>Qingshan Xu</a><sup>2</sup>
    ·
    <a>Beier Zhu</a><sup>1</sup>
    ·
    <a>Richang Hong</a><sup>2</sup>
    ·
    <a>Hanwang Zhang</a><sup>1</sup><br>
    <sup>1</sup>Nanyang Technological University <sup>2</sup>Hefei University of Technology
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2510.03550">Paper</a> | <a href="https://junbao-zhou.github.io/DragStream.github.io/">Website</a></h3>
</p>

---

Achieving streaming, fine-grained control over the outputs of autoregressive video diffusion models remains challenging, making it difficult to ensure that they consistently align with user expectations. To bridge this gap, we propose **stReaming drag-oriEnted interactiVe vidEo manipuLation (REVEL)**, a new task that enables users to modify generated videos *anytime* on *anything* via fine-grained, interactive drag. Beyond DragVideo and SG-I2V, REVEL unifies drag-style video manipulation as editing and animating video frames with both supporting user-specified translation, deformation, and rotation effects, making drag operations versatile. In resolving REVEL, we observe: *i*) drag-induced perturbations accumulate in latent space, causing severe latent distribution drift that halts the drag process; *ii*) streaming drag is easily disturbed by context frames, thereby yielding visually unnatural outcomes. We thus propose a training-free approach, **DragStream**, comprising: *i*) an adaptive distribution self-rectification strategy that leverages neighboring frames' statistics to effectively constrain the drift of latent embeddings; *ii*) a spatial-frequency selective optimization mechanism, allowing the model to fully exploit contextual information while mitigating its interference via selectively propagating visual cues along generation. Our method can be seamlessly integrated into existing autoregressive video diffusion models, and extensive experiments firmly demonstrate the effectiveness of our DragStream

---

![alt text](image.png)

## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 40 GB memory.
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation

### 1. Follow Self-Forcing to Install Dependencies

Create a conda environment and install dependencies:
```
conda create -n drag-stream python=3.10 -y
conda activate drag-stream
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

### 2. Follow Self-Forcing to Download Checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints --local-dir ./checkpoints
```

### 3. Follow Segment-Anything to Install SAM

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

### 4. Follow Segment-Anything to Download SAM Checkpoint

- [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## Drag/Animate Video with GUI

```
python click_gui_video.py
```

## CLI Inference with Saved Trajectories

```
python offline_run.py
```

## Reproducibility

To ensure every Drag/Animation is performed on the same generated video given the same input conditions, we set the random seed before the initialization of random noise and before the generation process.

Please refer to the `set_seed(seed)` in `inference.py`, `stream_inference.py`, `click_gui_video.py`, and `offline_run.py` for details.


## Acknowledgements
This codebase is built on top of the open-source implementation of [Self-Forcing](https://self-forcing.github.io/).
