# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RynnWorld-4D is a 4D embodied world model for robotic manipulation (Alibaba DAMO Academy). It co-generates synchronized RGB, depth, and optical flow via tri-branch diffusion transformers, and includes a policy head (RynnWorld-4D-Policy) for bimanual dexterous manipulation.

## Environment Setup

```bash
# World model
conda create -n rynnworld4d python=3.10 -y && conda activate rynnworld4d
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-build-isolation

# Policy head (additional)
cd rynnworld4d_policy && bash policy_env.sh && cd ..
```

CUDA 12.1 for world model, CUDA 12.6 for policy. bf16 mixed precision throughout.

## Key Commands

### Training (world model, all 3 stages use the same entry point)

```bash
# Stage 1: SFT warm-up (no cross-modal attention)
bash scripts/rynnworld4d-stage1.sh

# Stage 2: Tri-branch RoPE (enables joint attention, freezes non-joint params)
bash scripts/rynnworld4d-stage2.sh

# Stage 3: Full fine-tuning with joint attention + cosine decay
bash scripts/rynnworld4d-stage3.sh
```

All stages call `finetune_rynnworld4d.py`. Multi-node training requires env vars: `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NPROC_PER_NODE`.

### Inference

```bash
python inference-sft.py \
  --model_path ./pretrained/Wan2.2-TI2V-5B-Diffusers \
  --checkpoint_path ./training/rynnworld4d-stage3/checkpoint-1000 \
  --json_path ./data/sample.json \
  --output_dir ./results/inference-sft
```

### Policy training and serving

```bash
# Train
bash scripts/rynnworld4d-policy.sh

# Serve (OpenPI websocket server)
python rynnworld4d_policy/serve_rynnworld4d_policy.py

# Smoke test
python rynnworld4d_policy/smoke_test_serve.py
```

### No formal test suite or linting config exists in the repo.

## Architecture

### Tri-branch World Model (`core/`)

The `core/` package (installed via `pip install -e .`) contains training, inference, and tokenizer code.

**Training pipeline** (`core/finetune/`):
- `trainer.py` — Base Trainer class (adapted from finetrainers)
- `models/wan_i2v/module.py` — `RynnWorld4DTransformer3DModel`: extends WanTransformer3DModel with independent depth and optical flow branches, each initialized from pretrained RGB weights
- `models/wan_i2v/module_joint.py` — `JointRynnWorld4DTransformerBlock`: cross-modal joint attention where each branch attends to the other two via shared K/V projections with zero-initialized gating
- `models/wan_i2v/rynnworld4d_trainer.py` — `RynnWorld4DTrainer`: orchestrates 3-branch training with EMA, branch dropout, cosine decay
- `datasets/wan_dataset.py` — `RynnWorld4DDataset`: loads precomputed latent safetensors + JSON manifests

**3-stage training progression**:
1. Stage 1: `fusion_mode=none` — independent branches, no cross-modal attention
2. Stage 2: `fusion_mode=joint` + `freeze_non_joint=True` — trains only joint attention layers (joint_out/kv/q/norm/gate + modality_embed)
3. Stage 3: `fusion_mode=joint` + `freeze_non_joint=False` — full fine-tuning with cosine decay on depth/flow-to-RGB injection

**Inference** (`core/inference/rynnworld4d.py`): generates rgb.mp4, depth.mp4, flow.mp4 per sample.

### Policy Head (`rynnworld4d_policy/`)

Separate sub-project with its own requirements and Hydra-based config system (`policy_conf/`).

- `policy_models/vpp_policy.py` — `VPP_Policy`: main model
- `policy_models/module/wan_feature_extractor.py` — extracts features from frozen RynnWorld-4D backbone
- `policy_models/module/Video_Former.py` — Perceiver Resampler (3D)
- `policy_models/edm_diffusion/flow_matching.py` — flow matching action head (4-step Euler ODE)
- `third_party/` — vendored Depth-Anything-3 and OpenPI server (retain their own licenses)

### Distributed Training

- Stage 1 uses `accelerate launch` with configs in `configs_acc/`
- Stages 2-3 use `torchrun` with DeepSpeed ZeRO configs in `configs_zero/`
- Cross-node coordination uses rendezvous files on shared NAS for MASTER_PORT sync

### Important Implementation Details

- `WanTimeTextImageEmbedding.forward` and `WanTransformer3DModel.forward` are monkey-patched at import time for float32 stability during mixed-precision training
- Joint attention per block adds 12 linear layers (shared K/V reused across queries from other branches, cross-modal Q, gated output)
- Pretrained models expected under `pretrained/` (Wan2.2-TI2V-5B-Diffusers, RynnWorld-4D checkpoint, da3 for policy)

# 撰写设计/方案/分析/解析文档规范

* 图表用mermaid, 数学相关的用LaTex, 必要时可以用py脚本画一些更能帮助读者理解的图片(图片中的文字用英文). 这些脚本和图一般放在与生成的文档同目录的`asset`子文件夹中.
* 分析,解析和撰写文档时, 可以参考论文或代码库的官网, 官方文档, GItHUb, 参考github中的issues, 代码和pull requests, 也可参考网上其它可信来源的相关文章, 但参考内容要列出, 所生产的文档中若有与被参考对象相关的内容也要指出内容的出处. 
* 分析要深入仔细, 既要包括纵向分析(算法或方法的由来与演进历史, 以及在该算法或方法的基础上又演进和优化出了些什么解决类似问题的方法, 新老方法各有什么优缺点, 各适合应用到什么场景), 纵向分析(同时期同类算法的对比分析, 不同算法或方法各有什么优缺点, 各适合应用到什么场景), 和 消融分析(算法或方法中哪些点是在benchmark实验或实践中被证明有效的, 哪些点相对来说更有效, 哪些没那么有效).
* 记得深入分析模型或方法的输入,输出,在输入输出间做了些什么处理. 当然, 各组成模块的输入输出以及中间的处理也要分析. 为了训这个模型用了什么数据集和任务, 训出来后能做什么任务, 训练和推理时的输入输出数据格式大概长什么样.
* 系统或程序的设计要包括静态架构(组件图,类图,组件和类的职责与关系等等)和动态架构(数据流图,序列图,工作流图,不同场景下的各组件或类的调用与协调图.如果是算法还会涉及forward阶段的数据流,模型组件间的调用,以及backwawrd阶段的数据流,gradient流,哪些权重冻结哪些会被更新,和模型组件间的调用等等).
* 解释要深入浅出, 图文并茂, 可以举一些易于理解的例子帮助说明, 对关键的逻辑也要进行深入的代码解读, 要用严谨的科普论文的风格.

