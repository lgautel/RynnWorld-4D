# RynnWorld-4D 代码深入分析报告

> **分析对象**：RynnWorld-4D — 面向机器人操作的 4D 具身世界模型  
> **代码仓库**：本地 `RynnWorld-4D/` 目录（以实际代码为准）  
> **论文来源**：arXiv 2607.06559，TeX 源码位于 `b/d/p/TeX_Source/`  
> **分析日期**：2026-07-24

---

## 目录

1. [概述与论文核心思想](#1-概述与论文核心思想)
2. [代码库静态架构](#2-代码库静态架构)
3. [数据处理流水线](#3-数据处理流水线)
4. [三分支 World Model 架构深入分析](#4-三分支-world-model-架构深入分析)
5. [训练系统动态架构](#5-训练系统动态架构)
6. [推理流水线](#6-推理流水线)
7. [Policy Head 深入分析](#7-policy-head-深入分析)
8. [论文-代码对照分析](#8-论文-代码对照分析)
9. [关键设计决策与工程洞察](#9-关键设计决策与工程洞察)

---

## 1. 概述与论文核心思想

### 1.1 核心问题与动机

机器人在开放世界中的操作任务需要 world model（世界模型）来预测环境在交互下如何演变。现有方法存在两大类局限：

1. **2D 视频生成模型**（如 CogVideoX、Wan 2.2）：在像素空间预测未来帧，丢失了深度、6-DoF 位姿等 3D 空间信息，且存在时序不一致（物体尺度漂移、非物理形变）。
2. **3D/4D 重建方法**（NeRF/3DGS、Dynamic SfM）：要么是优化式的（慢、场景特定），要么是前馈式的（仅针对单物体），无法与预训练视频扩散模型的强大先验结合。

### 1.2 RGB-DF：投影 4D 表示

RynnWorld-4D 的核心创新在于提出了 **RGB-DF 表示**——同步生成 RGB 视频、深度视频和光流视频。这一表示的几何意义在于：

- **深度 (D)** 将每个像素提升到 3D 空间：$\mathbf{P}_t = D_t(u,v) \cdot \mathbf{K}^{-1} \cdot \mathbf{p}_t$
- **光流 (F)** 提供像素级运动：$\mathbf{f}_{opt} = [\Delta u, \Delta v]$
- 深度 + 光流可通过反投影得到 **3D 场景流**：$\mathbf{f}_{3D} = \mathbf{P}_{t+1} - \mathbf{P}_t$

这样既保留了与 2D 视频扩散模型兼容的格式（2D 对齐的张量），又使得 3D 几何和运动信息可被显式提取。

### 1.3 四大贡献

| # | 贡献 | 描述 |
|---|------|------|
| 1 | RGB-DF 投影 4D 表示 | 同时生成 RGB、深度、光流，支持 3D 场景流解释 |
| 2 | RynnWorld-4D 三分支世界模型 | 三分支 DiT + Joint Cross-Modal Attention |
| 3 | Rynn4DDataset 1.0 | 254.4M 帧的 4D 具身视频数据集 |
| 4 | RynnWorld-4D-Policy | 逆动力学策略头，利用内部 4D 表征进行高频闭环控制 |

### 1.4 横向分析：与同期 4D World Model 的定位区分

```mermaid
graph LR
    subgraph "2D World Models"
        A[CogVideoX] --> A1[RGB only]
        B[Wan 2.2] --> B1[RGB only]
    end
    subgraph "4D World Models"
        C[TesserAct] --> C1[RGB + Depth + Normals]
        D[4DNeX] --> D1[Dynamic Point Clouds]
        E[Free4D] --> E1[NeRF-based 4D]
        F["RynnWorld-4D"] --> F1["RGB + Depth + Optical Flow<br/>(RGB-DF)"]
    end
    
    style F fill:#e1f5fe,stroke:#0288d1
    style F1 fill:#e1f5fe,stroke:#0288d1
```

| 方法 | 表示 | 是否有光流 | 是否支持场景流 | 与视频扩散先验兼容 |
|------|------|-----------|--------------|-------------------|
| TesserAct | RGB-DN (含法线) | 否 | 否 | 是 |
| 4DNeX | 动态点云 | 否 | 间接（点运动） | 否 |
| Free4D | NeRF 4D 场 | 否 | 否 | 否 |
| **RynnWorld-4D** | **RGB-DF** | **是** | **是（反投影）** | **是** |

关键区分：RynnWorld-4D 是唯一包含光流分支的 4D 世界模型，使其能通过深度+光流反投影得到 3D 场景流，提供了显式的逐点 3D 运动线索。

### 1.5 纵向分析：从 2D 到 4D World Model 的演进

```mermaid
graph TB
    subgraph "第一代: 状态空间 World Model"
        WM1["Ha & Schmidhuber (2018)<br/>World Models (低维状态)"]
        WM2["Sutton's Dyna<br/>Planning in State Space"]
    end
    subgraph "第二代: 2D 视频 World Model"
        WM3["HunyuanVideo / Wan / CogVideoX<br/>像素空间视频预测"]
        WM4["SuSIE / UniPi<br/>2D 未来预测引导策略"]
    end
    subgraph "第三代: 3D/4D World Model"
        WM5["NeRF/3DGS-based<br/>优化式 3D 重建"]
        WM6["TesserAct (RGB-DN)<br/>多模态视频生成"]
        WM7["RynnWorld-4D (RGB-DF)<br/>三分支扩散 + 场景流"]
    end
    
    WM1 --> WM3
    WM2 --> WM3
    WM3 --> WM4
    WM3 --> WM6
    WM5 --> WM6
    WM6 --> WM7
    WM4 --> WM7
    
    style WM7 fill:#e1f5fe,stroke:#0288d1
```

RynnWorld-4D 的**关键进步**在于：（1）将 TesserAct 的法线替换为光流，获得了显式运动信息；（2）相比 SuSIE/UniPi 等 2D 方法，策略头直接消费内部 4D 特征（单次前向传播），无需逐步视频解码/去噪。

---

## 2. 代码库静态架构

### 2.1 顶层目录结构

```mermaid
graph TD
    ROOT["RynnWorld-4D/"]
    ROOT --> CORE["core/<br/>核心库（pip install -e .）"]
    ROOT --> POLICY["rynnworld4d_policy/<br/>策略头子项目"]
    ROOT --> SCRIPTS["scripts/<br/>训练启动脚本"]
    ROOT --> CONFIGS_ACC["configs_acc/<br/>Accelerate 配置"]
    ROOT --> CONFIGS_ZERO["configs_zero/<br/>DeepSpeed ZeRO 配置"]
    ROOT --> DATA["data/<br/>样本数据"]
    ROOT --> UTILS["utils/<br/>工具函数"]
    ROOT --> FINETUNE["finetune_rynnworld4d.py<br/>训练入口"]
    ROOT --> INFER["inference-sft.py<br/>推理入口"]
    
    CORE --> CORE_FT["finetune/<br/>训练框架"]
    CORE --> CORE_INF["inference/<br/>推理管线"]
    CORE --> CORE_TOK["tokenizer/<br/>视频编解码器"]
    
    CORE_FT --> MODELS["models/wan_i2v/<br/>模型实现"]
    CORE_FT --> DATASETS["datasets/<br/>数据集"]
    CORE_FT --> SCHEMAS["schemas/<br/>配置定义"]
    CORE_FT --> FT_UTILS["utils/<br/>训练工具"]
    
    POLICY --> P_MODELS["policy_models/<br/>策略模型"]
    POLICY --> P_CONF["policy_conf/<br/>Hydra 配置"]
    POLICY --> P_THIRD["third_party/<br/>第三方依赖"]
    POLICY --> P_TRAIN["train.py<br/>策略训练入口"]
    POLICY --> P_SERVE["serve_rynnworld4d_policy.py<br/>部署服务"]
    
    style CORE fill:#e8f5e9,stroke:#388e3c
    style POLICY fill:#fff3e0,stroke:#f57c00
    style FINETUNE fill:#e3f2fd,stroke:#1976d2
    style INFER fill:#e3f2fd,stroke:#1976d2
```

### 2.2 核心类继承层次

#### 2.2.1 Trainer 继承链

```mermaid
classDiagram
    class Trainer {
        <<abstract>>
        +accelerator: Accelerator
        +args: Args
        +state: State
        +components: Components
        +fit()
        +train()
        +prepare_models()
        +prepare_dataset()
        +prepare_trainable_parameters()
        +prepare_optimizer()
        +prepare_for_training()
        #collate_fn()*
        #load_components()*
        #compute_loss()*
        #encode_video()*
    }
    
    class WanI2VTrainer {
        +UNLOAD_LIST: ["text_encoder"]
        +load_components()
        +encode_video()
        +compute_loss()
        +inference()
        +decode_latents()
    }
    
    class RynnWorld4DTrainer {
        +UNLOAD_LIST: ["text_encoder", "vae"]
        +load_components()
        +prepare_for_training()
        +prepare_trainable_parameters()
        +prepare_optimizer()
        +prepare_dataset()
        +compute_loss()
        +train()
        -__prepare_saving_loading_hooks()
        -_maybe_save_checkpoint()
    }
    
    Trainer <|-- WanI2VTrainer
    WanI2VTrainer <|-- RynnWorld4DTrainer
```

**关键设计**：`RynnWorld4DTrainer` 重写了大部分方法，实际上是一个几乎完整的重新实现。它从 `WanI2VTrainer` 继承了 `encode_video()` 和 `decode_latents()` 等 VAE 相关功能，但训练循环、损失计算、参数准备等核心逻辑均为独立实现。

#### 2.2.2 模型继承链

```mermaid
classDiagram
    class WanTransformerBlock {
        +norm1: FP32LayerNorm
        +attn1: WanAttention
        +norm2: FP32LayerNorm
        +attn2: WanAttention
        +norm3: FP32LayerNorm
        +ffn: FeedForward
        +scale_shift_table: Parameter
        +forward(hidden_states, temb, rotary_emb, encoder_hidden_states)
    }
    
    class RynnWorld4DTransformerBlock {
        +norm1_depth/flow: FP32LayerNorm
        +attn1_depth/flow: WanAttention
        +norm2_depth/flow: FP32LayerNorm
        +attn2_depth/flow: WanAttention
        +ffn_depth/flow: FeedForward [optional]
        +video_to_depth_zero: Linear [optional]
        +video_to_flow_zero: Linear [optional]
        +depth_to_video_zero: Linear [optional]
        +flow_to_video_zero: Linear [optional]
        +forward() → (video, depth, flow)
    }
    
    class JointRynnWorld4DTransformerBlock {
        +joint_kv_video/depth/flow: Linear
        +joint_q_video/depth/flow: Linear
        +joint_out_video/depth/flow: Linear [zero-init]
        +joint_gate_video/depth/flow: Parameter [init=1.0]
        +joint_gate_video_decay: Buffer [init=1.0]
        +modality_embed_video/depth/flow: Parameter [zero-init]
        +joint_align_video/depth/flow: LayerNorm
        +joint_norm_q/k: RMSNorm
        +forward() → (video, depth, flow)
        -_joint_cross_modal_attention()
        -_apply_rope()
        -_compute_attention()
    }
    
    WanTransformerBlock <|-- RynnWorld4DTransformerBlock
    WanTransformerBlock <|-- JointRynnWorld4DTransformerBlock
    
    class WanTransformer3DModel {
        +patch_embedding: Conv3d
        +condition_embedder
        +rope: WanRotaryPosEmbed
        +blocks: ModuleList
        +norm_out: FP32LayerNorm
        +proj_out: Linear
        +forward(hidden_states, timestep, encoder_hidden_states)
    }
    
    class RynnWorld4DTransformer3DModel {
        +patch_embedding_depth: Conv3d
        +patch_embedding_flow: Conv3d
        +norm_out_depth/flow: FP32LayerNorm
        +proj_out_depth/flow: Linear
        +forward() → (video_out, depth_out, flow_out)
        +from_pretrained() [weight cloning]
    }
    
    class JointRynnWorld4DTransformer3DModel {
        +joint_start_layer: int
        +joint_end_layer: int
        +joint_every_n_layers: int
        +from_pretrained() [weight cloning + joint params]
        +load_state_dict() [tolerant loading]
    }
    
    WanTransformer3DModel <|-- RynnWorld4DTransformer3DModel
    WanTransformer3DModel <|-- JointRynnWorld4DTransformer3DModel
```

**重要观察**：`RynnWorld4DTransformer3DModel`（Stage 1）和 `JointRynnWorld4DTransformer3DModel`（Stage 2/3）是**平行的两个子类**，而非继承关系。Stage 2 的模型并非从 Stage 1 的类继承，而是独立实现了三分支结构 + Joint Attention。这意味着从 Stage 1 切换到 Stage 2 时，会构建一个全新的模型类并通过 `load_state_dict` 加载 Stage 1 的权重。

#### 2.2.3 Policy 模型组件

```mermaid
classDiagram
    class VPP_Policy {
        +TVP_encoder: WanFeatureExtractor [frozen]
        +Video_Former: Video_Former_3D [trainable]
        +model: FlowMatchingPolicy [trainable]
        +latent_dim: 384
        +multistep: 10
        +extract_predictive_feature(batch)
        +training_step(batch)
        +eval_forward(obs, goal)
        +step(obs, goal)
    }
    
    class WanFeatureExtractor {
        +transformer: RynnWorld4D/Joint model
        +vae: AutoencoderKLWan
        +text_encoder: UMT5
        +da3_model: DepthAnything3 [lazy-loaded]
        +extract_block_idx: int = 15
        +forward(pixel_values, texts, timestep)
        -_transformer_step_rynnworld4d()
        -_build_rynnworld4d_latents()
        -_StopForward [sentinel exception]
    }
    
    class Video_Former_3D {
        +goal_emb: MLP [9216→384]
        +latents: Parameter [learnable queries]
        +time_pos_emb: Parameter
        +layers: PerceiverAttn + TempAttn + FFW
        +forward(features) → (B, 336, 384)
    }
    
    class FlowMatchingPolicy {
        +inner_model: DiffusionTransformer
        +loss(state, actions, goal) → scalar
        +sample(state, goal, shape, n_steps=4) → actions
        -predict_velocity(state, x_t, goal, t)
    }
    
    class DiffusionTransformer {
        +encoder: TransformerEncoder [4 layers]
        +decoder: TransformerFiLMDecoder [4 layers]
        +sigma_emb: SinusoidalPosEmb + MLP
        +goal_emb/lang_emb/action_emb: MLP/Linear
        +proprio_emb: MLP
        +action_pred: Linear
        +forward_enc_only()
        +forward_dec_only()
    }
    
    VPP_Policy --> WanFeatureExtractor : frozen backbone
    VPP_Policy --> Video_Former_3D : feature compression
    VPP_Policy --> FlowMatchingPolicy : action generation
    FlowMatchingPolicy --> DiffusionTransformer : inner model
```

### 2.3 模块间依赖关系

```mermaid
graph TB
    subgraph "入口层"
        FT["finetune_rynnworld4d.py"]
        INF["inference-sft.py"]
        PT["rynnworld4d_policy/train.py"]
    end
    
    subgraph "核心训练框架 (core/finetune/)"
        TRAINER["trainer.py<br/>Base Trainer"]
        WAN_TRAINER["wan_trainer.py<br/>WanI2VTrainer"]
        RW4D_TRAINER["rynnworld4d_trainer.py<br/>RynnWorld4DTrainer"]
    end
    
    subgraph "模型实现 (core/finetune/models/wan_i2v/)"
        MODULE["module.py<br/>Stage 1 三分支"]
        MODULE_J["module_joint.py<br/>Stage 2/3 Joint Attention"]
    end
    
    subgraph "推理管线 (core/inference/)"
        RW4D_PIPE["rynnworld4d.py<br/>RynnWorld4DPipeline"]
    end
    
    subgraph "数据 (core/finetune/datasets/)"
        WAN_DS["wan_dataset.py<br/>RynnWorld4DDataset"]
    end
    
    subgraph "策略 (rynnworld4d_policy/)"
        VPP["vpp_policy.py"]
        WFE["wan_feature_extractor.py"]
        VF["Video_Former.py"]
        FM["flow_matching.py"]
        DT["diffusion_decoder.py"]
        TDS["tianji_dataset.py"]
    end
    
    subgraph "外部依赖"
        DIFFUSERS["diffusers<br/>WanTransformer3DModel"]
        DEEPSPEED["deepspeed"]
        ACCELERATE["accelerate"]
        HYDRA["hydra-core"]
    end
    
    FT --> RW4D_TRAINER
    INF --> RW4D_PIPE
    PT --> VPP
    
    RW4D_TRAINER --> WAN_TRAINER --> TRAINER
    RW4D_TRAINER --> MODULE
    RW4D_TRAINER --> MODULE_J
    RW4D_TRAINER --> WAN_DS
    
    RW4D_PIPE --> MODULE
    
    VPP --> WFE
    VPP --> VF
    VPP --> FM --> DT
    WFE --> MODULE
    WFE --> MODULE_J
    
    MODULE --> DIFFUSERS
    MODULE_J --> DIFFUSERS
    TRAINER --> ACCELERATE
    TRAINER --> DEEPSPEED
    PT --> HYDRA
```

---

## 3. 数据处理流水线

### 3.1 Rynn4DDataset 1.0 构建流程

论文描述的数据集包含超过 **254.4M 帧**，来源涵盖人类中心数据集（Epic-Kitchens、EgoVid）和机器人操作数据集（RoboMIND、RDT-1B、Galaxea、RoboCoin、AgiBot）。

每帧生成三种伪标签：

```mermaid
flowchart LR
    RAW["原始视频"] --> CAP["视频标注<br/>Qwen3-VL"]
    RAW --> FLOW["光流估计<br/>DPFlow"]
    RAW --> DEPTH["深度估计<br/>Depth Anything 3"]
    
    CAP --> |"结构化描述<br/>max 512 tokens"| CAPTION["文本标注"]
    FLOW --> |"逐帧对光流<br/>Middlebury 颜色编码"| FLOW_VID["光流视频 (MP4)"]
    DEPTH --> |"30 FPS<br/>clip [0, 5m]<br/>8-bit 灰度"| DEPTH_VID["深度视频 (MP4)"]
    
    CAPTION --> PREPROCESS["预处理<br/>utils/pre-process.py"]
    FLOW_VID --> PREPROCESS
    DEPTH_VID --> PREPROCESS
    RAW --> PREPROCESS
    
    PREPROCESS --> |"VAE 编码"| RGB_LAT["rgb_latents.safetensors<br/>(video_latents + text_embeds)"]
    PREPROCESS --> |"VAE 编码"| FD_LAT["flow_depth_latents.safetensors<br/>(depth_latents + flow_latents)"]
    PREPROCESS --> MANIFEST["data manifest (JSON)"]
```

### 3.2 预处理代码分析：`utils/pre-process.py`

`PreProcess` 类封装了 VAE 编码和文本编码的全流程：

**VAE 编码** (`encode_video`)：
```python
# 伪代码概要（基于 utils/pre-process.py 实际实现）
frames = decord.VideoReader(video_path)           # 读取视频
frames = center_crop_and_resize(frames, 640, 480) # 裁剪到目标分辨率
frames = (frames / 255.0 - 0.5) * 2.0             # 归一化到 [-1, 1]
latent = vae.encode(frames).latent_dist.sample()   # VAE 编码
latent = (latent - latents_mean) / latents_std     # 潜变量标准化
```

**文本编码** (`_get_t5_prompt_embeds`)：
- 使用 `UMT5EncoderModel` + `T5TokenizerFast`
- 文本预处理：`ftfy.fix_text` + `html.unescape` + 去除多余空白
- 最大序列长度 512，零填充
- 输出维度：`(1, 512, 4096)`

**输出数据格式**：

| 文件 | 键名 | 形状 | 说明 |
|------|------|------|------|
| `*_rgb.safetensors` | `video_latents` | `(C_z, F_lat, H_lat, W_lat)` | RGB VAE 潜变量 |
| | `text_embeds` | `(1, seq_len, 4096)` | UMT5 文本嵌入 |
| `*_flow_depth.safetensors` | `depth_latents` | `(C_z, F_lat, H_lat, W_lat)` | 深度 VAE 潜变量 |
| | `flow_latents` | `(C_z, F_lat, H_lat, W_lat)` | 光流 VAE 潜变量 |

其中 $C_z = 16$（Wan VAE 潜变量通道数），$F_{lat} = (F_{raw} - 1) / 4 + 1$（Causal VAE 4x 时间压缩），$H_{lat} = H / 8$，$W_{lat} = W / 8$（8x 空间压缩）。

### 3.3 训练数据集：`RynnWorld4DDataset`

> 源码位置：`core/finetune/datasets/wan_dataset.py`

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant DS as RynnWorld4DDataset
    participant FS as 文件系统
    
    DS->>FS: 读取 JSON manifest
    DS->>DS: 过滤缺失键的条目
    DS->>DS: _prepare_null_embedding()
    Note over DS: SHA256(prompt) → cache 查找<br/>仅 LOCAL_RANK=0 生成<br/>其他 rank 等待（≤60s）
    
    DL->>DS: __getitem__(idx)
    DS->>FS: load_safetensors(rgb_latents_path)
    DS->>FS: load_safetensors(flow_depth_latents_path)
    Note over DS: 含重试逻辑（5次，间隔1s）
    DS->>DS: 形状一致性检查
    Note over DS: 若 RGB/depth/flow 空间维度不匹配<br/>则回退到随机样本
    DS->>DS: 提取首帧潜变量
    DS-->>DL: {encoded_video, encoded_depth,<br/>encoded_flow, img_latent,<br/>depth_latent, flow_latent,<br/>null_embedding, text_embedding}
```

**关键实现细节**：
- 首帧潜变量提取：`img_latent = encoded_video[:, :1, :, :]`，保留时间维度为 1
- 空嵌入缓存：使用 `prompt` 字符串的 SHA256 哈希作为缓存键，避免重复编码
- 多 GPU 安全：只有 `LOCAL_RANK=0` 的进程生成空嵌入并保存到文件系统，其他进程轮询等待

### 3.4 Tianji 机器人数据集：`TianjiVideoDataset`

> 源码位置：`rynnworld4d_policy/policy_models/datasets/tianji_dataset.py`

用于 Policy 训练的数据集，读取 TIANJI M6 机器人的遥操作采集数据：

**Episode 目录结构**：
```
episode_00001/
├── observation.images.head.mp4        # 头部摄像头 (1280×720, 30fps)
├── timeseries.parquet                 # 每帧动作/状态数据
├── metadata.json                      # 任务描述、fps、总帧数
└── [可选] observation.images.left_wrist.mp4
```

**数据处理**：
- RGB 增强：CenterCrop → ColorJitter(亮度/对比度/饱和度=0.2) → ToTensor → Normalize 到 [-1, 1]
- 深度：CenterCrop → ToTensor → Normalize 到 [-1, 1]（无增强）
- 动作标准化：计算全 episode 的逐维均值/标准差，保存为 `action_stats.json`

**输出字典**：

| 键 | 形状 | 说明 |
|----|------|------|
| `rgb_obs.rgb_static` | `(obs_seq_len, 3, H, W)` | RGB 观测 |
| `state` | `(54,)` | 本体感知（关节位置） |
| `actions` | `(10, 54)` | 标准化动作序列 |
| `depth_static` | `(1, 3, H, W)` | 深度图（如可用） |

---

## 4. 三分支 World Model 架构深入分析

### 4.1 基座模型：Wan 2.2-TI2V-5B

RynnWorld-4D 构建在 Wan 2.2 Text/Image-to-Video 5B 扩散 Transformer 之上。该模型的核心参数：

| 参数 | 论文描述 | 代码构造器默认值 | 说明 |
|------|---------|-----------------|------|
| 隐藏维度 $d$ | 3072 | `inner_dim = num_attention_heads × attention_head_dim` | 代码默认 40×128=5120，但实际由预训练模型 config 决定 |
| 层数 | 30 | `num_layers = 40`（默认） | 同上，实际值来自预训练 config |
| FFN 维度 | 14,336 | `ffn_dim = 13824`（默认） | 同上 |
| 注意力头数 | — | `num_attention_heads = 40`（默认） | 5B 模型实际约 24 头 |
| 头维度 | — | `attention_head_dim = 128` | — |
| Patch 大小 | — | `(1, 2, 2)` | 时间步幅 1，空间步幅 2 |
| 输入通道 | — | `in_channels = 16` | VAE 潜变量通道数 |
| 文本维度 | — | `text_dim = 4096` | UMT5 编码器输出维度 |

> **注**：代码中的构造器默认值（如 `num_attention_heads=40, num_layers=40`）为 Python 参数默认值，实际运行时由 `from_pretrained()` 加载的预训练模型 `config.json` 覆盖。论文描述的 Wan 2.2-TI2V-5B 参数为 $d=3072$、30 层。`module_joint.py` 文件头注释中使用 `dim=5120` 计算参数开销，可能是基于更大变体的估算。

#### Flow Matching 目标函数

Wan 2.2 使用 **Rectified Flow**（整流流）作为生成框架。给定数据 $z_0$ 和噪声 $\epsilon \sim \mathcal{N}(0, I)$，线性插值路径为：

$$z_t = (1 - \sigma_t) z_0 + \sigma_t \epsilon$$

其中 $\sigma_t$ 经过 **flow shift** 调度：

$$\sigma_t = \frac{s \cdot \text{shift}}{1 + (\text{shift} - 1) \cdot s}, \quad s = \frac{t}{T}$$

`shift` 为调度器超参数（Wan 默认 `flow_shift=5.0`），控制噪声调度的非线性程度。

速度预测目标为：

$$v_\theta(z_t, t) = \epsilon - z_0$$

训练损失即速度场的均方误差。

#### First-Frame Conditioning

Wan 2.2 的 Image-to-Video 模式使用**首帧条件注入**：

1. 将首帧图像编码为潜变量 $z_0^{f=0}$
2. 在每个去噪步骤中，将噪声潜变量的第 0 帧替换为干净的首帧潜变量
3. 使用 **per-token timestep**：第 0 帧的时间步设为 0（表示干净信号），其他帧使用采样的时间步 $t$

代码中 per-token timestep 的构建（`compute_loss` 方法）：

```python
# first_frame_mask: shape [B, C_z, F_lat, H_lat, W_lat], 第0帧=0，其余=1
# shifted_timesteps: 标量时间步
per_token_timestep = first_frame_mask[0][0][:, ::2, ::2] * shifted_timesteps
# 下采样 ::2 匹配 patch 后的空间维度
```

### 4.2 Stage 1：三分支独立训练

> 源码位置：`core/finetune/models/wan_i2v/module.py`  
> 对应训练脚本：`scripts/rynnworld4d-stage1.sh`  
> 训练参数：`fusion_mode=none`, `share_ffn=False`, `lr=1.5e-5`, `loss_weight_flow=0.5`

#### 4.2.1 权重克隆策略

Stage 1 的核心设计决策是从预训练的 Wan 2.2 RGB 模型权重**完全克隆**出深度和光流分支。`from_pretrained` 类方法（`module.py`）实现了精细的权重映射：

```mermaid
flowchart TB
    PRETRAINED["预训练 Wan2.2-TI2V-5B<br/>WanTransformer3DModel"]
    
    subgraph "权重克隆映射"
        direction LR
        P1["patch_embedding.*"] --> D1["patch_embedding_depth.*"]
        P1 --> F1["patch_embedding_flow.*"]
        P2["norm_out.*"] --> D2["norm_out_depth.*"]
        P2 --> F2["norm_out_flow.*"]
        P3["proj_out.*"] --> D3["proj_out_depth.*"]
        P3 --> F3["proj_out_flow.*"]
        P4["blocks.*.attn1.*"] --> D4["blocks.*.attn1_depth.*"]
        P4 --> F4["blocks.*.attn1_flow.*"]
        P5["blocks.*.attn2.*"] --> D5["blocks.*.attn2_depth.*"]
        P5 --> F5["blocks.*.attn2_flow.*"]
        P6["blocks.*.ffn.*"] --> D6["blocks.*.ffn_depth.*"]
        P6 --> F6["blocks.*.ffn_flow.*"]
    end
    
    subgraph "保留零初始化"
        Z1["video_to_depth_zero (zero)"]
        Z2["video_to_flow_zero (zero)"]
        Z3["depth_to_video_zero (zero)"]
        Z4["flow_to_video_zero (zero)"]
    end
    
    PRETRAINED --> P1
    PRETRAINED --> P2
    PRETRAINED --> P3
    PRETRAINED --> P4
    PRETRAINED --> P5
    PRETRAINED --> P6
```

这意味着在训练开始时，三个分支具有**完全相同的权重**，各自从同一起点开始适应不同模态。融合层（`*_zero` 线性层）保持零初始化，确保初始前向传播行为与单分支模型一致。

#### 4.2.2 `RynnWorld4DTransformerBlock` 架构

每个 Transformer block 的内部结构：

```mermaid
flowchart TB
    subgraph "RynnWorld4DTransformerBlock"
        INPUT_V["video hidden states"]
        INPUT_D["depth hidden states"]
        INPUT_F["flow hidden states"]
        
        subgraph "AdaLN 调制"
            TEMB["temb (时间嵌入)"] --> SST["scale_shift_table<br/>6 组调制参数"]
        end
        
        subgraph "自注意力 (独立)"
            SA_V["norm1 → attn1<br/>(video)"]
            SA_D["norm1_depth → attn1_depth<br/>(depth)"]
            SA_F["norm1_flow → attn1_flow<br/>(flow)"]
        end
        
        subgraph "文本交叉注意力 (共享 K/V)"
            CA_V["norm2 → attn2<br/>(video, 独立 Q)"]
            CA_D["norm2_depth → attn2_depth<br/>(depth, 独立 Q, 共享 K/V)"]
            CA_F["norm2_flow → attn2_flow<br/>(flow, 独立 Q, 共享 K/V)"]
        end
        
        subgraph "分支间融合 [fusion_mode=bidirectional]"
            FUS["video_to_depth_zero (→depth)<br/>video_to_flow_zero (→flow)<br/>depth_to_video_zero (→video)<br/>flow_to_video_zero (→video)"]
        end
        
        subgraph "前馈网络"
            FFN_V["norm3 → ffn<br/>(video)"]
            FFN_D["norm3 → ffn_depth<br/>(depth, 独立 or 共享)"]
            FFN_F["norm3 → ffn_flow<br/>(flow, 独立 or 共享)"]
        end
        
        INPUT_V --> SA_V --> CA_V --> FUS --> FFN_V
        INPUT_D --> SA_D --> CA_D --> FUS --> FFN_D
        INPUT_F --> SA_F --> CA_F --> FUS --> FFN_F
    end
```

**关键实现细节**：

1. **AdaLN 共享**：三个分支共用同一组 `scale_shift_table` 调制参数（来自时间嵌入），不做分支特定的调制。这是因为三个分支共享相同的去噪时间步。

2. **文本交叉注意力的 K/V 共享**（`module.py` 中）：
   ```python
   # 深度和光流的 K/V 投影直接指向 video 的 attn2
   self.attn2_depth.to_k = self.attn2.to_k
   self.attn2_depth.norm_k = self.attn2.norm_k
   self.attn2_depth.to_v = self.attn2.to_v
   self.attn2_flow.to_k = self.attn2.to_k
   self.attn2_flow.norm_k = self.attn2.norm_k
   self.attn2_flow.to_v = self.attn2.to_v
   ```
   这意味着文本条件信号以相同的 K/V 投影进入三个分支，但每个分支用独立的 Q 投影来选择性提取不同的信息。

3. **FFN 独立性**：当 `share_ffn=False`（Stage 1 实际使用的配置）时，深度和光流分支拥有独立的 FFN。消融实验表明共享 FFN 会导致性能崩溃，因为 RGB 纹理、深度流形和运动场的潜空间本质上是异构的。

4. **融合模式**（Stage 1 实际使用 `fusion_mode=none`，不启用任何融合层）：
   - `"bidirectional"`：4 个零初始化 Linear，双向信息交换
   - `"unidirectional"`：2 个零初始化 Linear，仅 depth/flow → video
   - `"none"`：无融合层（Stage 1 实际配置）

#### 4.2.3 Stage 1 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `fusion_mode` | `none` | 无跨模态注意力 |
| `share_ffn` | `False` | 独立 FFN |
| `learning_rate` | `1.5e-5` | |
| `lr_warmup_steps` | `300` | |
| `gradient_accumulation` | `4` | |
| `train_resolution` | `25×480×832` | (帧数, 高, 宽) |
| `use_ema` | `True` | EMA 衰减 0.9999 |
| `loss_weight_flow` | `0.5` | 光流损失权重降低 |
| 启动方式 | `accelerate launch` | 多节点 DeepSpeed |

> **光流损失权重 0.5 的原因**：在 Stage 1 中，光流的首帧条件（白色图 = 零光流）信息量远低于 RGB 的首帧条件（实际图像），因此降低光流损失权重以避免其主导训练。

### 4.3 Stage 2：Joint Cross-Modal Attention

> 源码位置：`core/finetune/models/wan_i2v/module_joint.py`  
> 对应训练脚本：`scripts/rynnworld4d-stage2.sh`  
> 训练参数：`fusion_mode=joint`, `freeze_non_joint=True`, `joint_every_n_layers=3`

Stage 2 是架构设计的核心创新——引入 **Joint Cross-Modal Attention (JA)** 实现三分支之间的深层特征交互。

#### 4.3.1 JA 模块架构

```mermaid
flowchart TB
    subgraph "Joint Cross-Modal Attention Module"
        subgraph "输入对齐"
            V_IN["video hidden states"] --> V_ALIGN["LayerNorm<sub>video</sub>"]
            D_IN["depth hidden states"] --> D_ALIGN["LayerNorm<sub>depth</sub>"]
            F_IN["flow hidden states"] --> F_ALIGN["LayerNorm<sub>flow</sub>"]
            
            V_ALIGN --> V_ME["+modality_embed<sub>video</sub>"]
            D_ALIGN --> D_ME["+modality_embed<sub>depth</sub>"]
            F_ALIGN --> F_ME["+modality_embed<sub>flow</sub>"]
        end
        
        subgraph "Q/K/V 投影"
            V_ME --> V_Q["Q<sub>video</sub>"] 
            V_ME --> V_KV["K<sub>video</sub>, V<sub>video</sub>"]
            D_ME --> D_Q["Q<sub>depth</sub>"]
            D_ME --> D_KV["K<sub>depth</sub>, V<sub>depth</sub>"]
            F_ME --> F_Q["Q<sub>flow</sub>"]
            F_ME --> F_KV["K<sub>flow</sub>, V<sub>flow</sub>"]
        end
        
        subgraph "QK 归一化 + RoPE"
            V_Q --> V_QN["RMSNorm<sub>q</sub> + RoPE"]
            D_Q --> D_QN["RMSNorm<sub>q</sub> + RoPE"]
            F_Q --> F_QN["RMSNorm<sub>q</sub> + RoPE"]
            V_KV --> V_KN["RMSNorm<sub>k</sub> + RoPE"]
            D_KV --> D_KN["RMSNorm<sub>k</sub> + RoPE"]
            F_KV --> F_KN["RMSNorm<sub>k</sub> + RoPE"]
        end
        
        subgraph "跨模态注意力"
            V_QN --> VA["Attn(Q<sub>v</sub>, [K<sub>d</sub>;K<sub>f</sub>], [V<sub>d</sub>;V<sub>f</sub>])"]
            D_QN --> DA["Attn(Q<sub>d</sub>, [K<sub>v</sub>;K<sub>f</sub>], [V<sub>v</sub>;V<sub>f</sub>])"]
            F_QN --> FA["Attn(Q<sub>f</sub>, [K<sub>v</sub>;K<sub>d</sub>], [V<sub>v</sub>;V<sub>d</sub>])"]
            V_KN --> DA
            V_KN --> FA
            D_KN --> VA
            D_KN --> FA
            F_KN --> VA
            F_KN --> DA
        end
        
        subgraph "门控输出"
            VA --> V_OUT["OutProj<sub>video</sub>(zero-init) × tanh(gate<sub>video</sub>) × decay"]
            DA --> D_OUT["OutProj<sub>depth</sub>(zero-init) × tanh(gate<sub>depth</sub>)"]
            FA --> F_OUT["OutProj<sub>flow</sub>(zero-init) × tanh(gate<sub>flow</sub>)"]
        end
        
        V_OUT --> V_RES["+ video residual"]
        D_OUT --> D_RES["+ depth residual"]
        F_OUT --> F_RES["+ flow residual"]
    end
```

#### 4.3.2 参数效率分析

论文声称使用 "shared K/V" 设计将参数从朴素设计的 $18d^2$ 降低到 $12d^2$。代码实现验证了这一点：

**朴素设计**（每个分支对其他两个分支各自独立 Q/K/V）：
- 3 个分支 × 2 个交叉注意力 × 3 个投影 (Q, K, V) = 18 个 $d \times d$ 线性层

**实际设计**（shared K/V）：
- 3 个分支 × 1 个 K 投影 = 3 个线性层（K 被其他两个分支的 Q 共享）
- 3 个分支 × 1 个 V 投影 = 3 个线性层
- 3 个分支 × 1 个 Q 投影 = 3 个线性层
- 3 个分支 × 1 个 Out 投影 = 3 个线性层
- **总计 12 个 $d \times d$ 线性层**

JA 放置策略（`joint_every_n_layers=3`）：在 30 层中每隔 3 层放置一个 JA 模块，共 10 个 JA 模块（在 blocks 0, 3, 6, 9, 12, 15, 18, 21, 24, 27）。

#### 4.3.3 Gate 设计深入分析

这是 RynnWorld-4D 最精妙的工程设计之一。代码中每个 JA 模块的输出组合为：

$$\text{output}_m = \underbrace{\text{OutProj}_m(\text{Attn}_m)}_{\text{zero-init}} \cdot \underbrace{\tanh(g_m)}_{\tanh(1.0)} \cdot \underbrace{d_m}_{\text{decay (仅 video)}}$$

**初始化状态**：
- $\text{OutProj}_m$：权重和偏置均初始化为 **零**
- $g_m$：初始化为 **1.0**（`nn.Parameter(torch.ones(1))`）
- $d_m$（decay buffer）：初始化为 **1.0**

**为什么不用 double zero-init？**

`module_joint.py` 的注释（约第 134-143 行）给出了明确解释。如果 `OutProj` 和 `gate` 都初始化为零，会产生**鞍点死锁（saddle-point deadlock）**：

$$\frac{\partial \mathcal{L}}{\partial W_{out}} = \tanh(g) \cdot \frac{\partial \mathcal{L}}{\partial \text{output}}$$

如果 $g = 0$，则 $\tanh(0) = 0$，导致 $\frac{\partial \mathcal{L}}{\partial W_{out}} = 0$。同时：

$$\frac{\partial \mathcal{L}}{\partial g} = W_{out} \cdot x \cdot (1 - \tanh^2(g)) \cdot \frac{\partial \mathcal{L}}{\partial \text{output}}$$

如果 $W_{out} = 0$，则 $\frac{\partial \mathcal{L}}{\partial g} = 0$。两个参数互相锁死，无法逃离零点。

**对比 ControlNet 的 zero-init**：ControlNet 使用单一的零初始化卷积（没有 gate），梯度可以正常流动。但 RynnWorld-4D 的双层设计需要避免这种死锁。

**当前设计的梯度分析**：

$$\begin{aligned}
\text{output} &= W_{out} \cdot x \cdot \tanh(g) \\
\frac{\partial \mathcal{L}}{\partial W_{out}} &= \tanh(1.0) \cdot x \cdot \frac{\partial \mathcal{L}}{\partial \text{output}} \approx 0.762 \cdot x \cdot \frac{\partial \mathcal{L}}{\partial \text{output}} \neq 0
\end{aligned}$$

$W_{out}$ 可以立即获得非零梯度，从而开始学习。同时，由于 $W_{out} = 0$，初始输出仍然为零，保证了从 Stage 1 到 Stage 2 的平滑过渡。

#### 4.3.4 3D RoPE 在 Joint Attention 中的应用

当 `joint_use_rope=True` 时，Q 和 K 在跨模态注意力中也应用 3D 旋转位置编码。实现位于 `_apply_rope` 方法：

```python
def _apply_rope(self, x, cos, sin):
    # x: [B, N, dim] → unflatten → [B, N, H, D]
    x = x.unflatten(-1, (self.num_heads, self.head_dim))
    x1, x2 = x.chunk(2, dim=-1)
    # 旋转：[x1·cos - x2·sin, x1·sin + x2·cos]
    x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x.flatten(-2, -1)  # → [B, N, dim]
```

**设计意图**：3D RoPE 编码了每个 token 的 $(t, h, w)$ 空间位置。在跨模态注意力中应用 RoPE，使得同一空间位置的 RGB/深度/光流 token 之间的注意力权重天然更高，实现了**空间对齐的特征融合**。消融实验证实去除 RoPE 会导致深度精度 $\delta_1$ 从 0.610 下降到 0.450。

#### 4.3.5 Frame-wise Attention

当 `joint_frame_wise=True` 时，跨模态注意力被限制在**同一时间帧**内的 token 之间：

```python
if self.joint_frame_wise and num_frames > 1:
    # [B, T*S, dim] → [B*T, S, dim]
    # 每个分支独立 reshape，然后在 S 维度上做 cross-attention
```

这将注意力的序列长度从 $T \times S$ 降低到 $S$（其中 $S = H' \times W'$ 为单帧空间 token 数），显著降低了计算开销。物理直觉是：同一时刻的 RGB/深度/光流之间的对应关系是最直接的。

#### 4.3.6 Branch Dropout

Stage 2 使用 `branch_dropout_prob=0.2`：

```python
# compute_loss 中的实现
if branch_dropout_prob > 0:
    # 随机选择一个分支（从 ["depth", "flow"] 中）
    # 将该分支的非首帧噪声潜变量替换为纯高斯噪声
    # video 分支永远不会被 dropout
```

**设计意图**：强制 JA 学习从可见模态重建被遮蔽模态的能力，增强模型的鲁棒性。Video（RGB）分支作为外观锚点（appearance anchor）永远不被 dropout。

#### 4.3.7 Stage 2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `fusion_mode` | `joint` | 启用 Joint Attention |
| `freeze_non_joint` | `True` | 冻结 backbone，仅训练 JA 参数 |
| `joint_start_layer` | `0` | 从第 0 层开始 |
| `joint_end_layer` | `30` | 到第 30 层结束 |
| `joint_every_n_layers` | `3` | 每 3 层一个 JA 模块 |
| `joint_frame_wise` | `True` | 帧内跨模态注意力 |
| `joint_use_rope` | `True` | 使用 3D RoPE |
| `joint_unidirectional` | `True` | video 仅作为 K/V 源 |
| `branch_dropout_prob` | `0.2` | 分支 dropout |
| `learning_rate` | `5e-5` | |
| `joint_out_lr` | `2.5e-4` | JA 输出投影专用 LR |
| `lr_warmup_steps` | `200` | |
| 启动方式 | `torchrun` + ZeRO-2 | CPU offload |

**单向模式 (`joint_unidirectional=True`)**：在 Stage 2 中，video 分支不接收来自 depth/flow 的信息（仅作为 K/V 源）。只有 depth/flow 分支从 video 中获取信息。这意味着此阶段仅训练 depth/flow 的跨模态理解能力。

### 4.4 Stage 3：全参数联合微调

> 对应训练脚本：`scripts/rynnworld4d-stage3.sh`  
> 训练参数：`freeze_non_joint=False`, `joint_video_decay=True`

#### 4.4.1 Cosine Decay 机制

Stage 3 的关键创新是 `joint_gate_video_decay`——一个注册为 buffer 的衰减因子，对**仅 video 分支**的 JA 输出进行渐进衰减：

$$d(t) = \frac{1}{2}\left(1 + \cos\left(\pi \cdot \min\left(1, \frac{t}{T_{decay}}\right)\right)\right)$$

其中 $T_{decay}$ 默认为 700 步。

```python
# rynnworld4d_trainer.py train() 方法中
decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, global_step / decay_steps)))
for block in model.blocks:
    if hasattr(block, 'joint_gate_video_decay'):
        block.joint_gate_video_decay.fill_(decay)
```

衰减过程：$d(0) = 1.0 \to d(T_{decay}/2) = 0.5 \to d(T_{decay}) = 0.0$

**应用于 forward pass**：
```python
# module_joint.py forward 中
hidden_states += self.joint_out_video(attn_out_v) * self.joint_gate_video.tanh() * self.joint_gate_video_decay
```

**设计意图**：在 Stage 3 初期，video 分支仍然接收来自 depth/flow 的跨模态信息（与 Stage 2 结束时一致）。随着训练进行，逐渐移除 depth/flow 对 video 的反向注入，使得 **video 分支回归独立生成能力**，而 depth/flow 分支继续从 video 中获取信息。这实现了一种非对称的信息流：

$$\text{video} \xrightarrow{\text{始终}} \text{depth/flow}, \quad \text{depth/flow} \xrightarrow{\text{渐进消失}} \text{video}$$

#### 4.4.2 差异化学习率

Stage 3 使用三组不同的学习率：

| 参数组 | 学习率 | 说明 |
|--------|--------|------|
| `joint_out_*` | `1e-5` (`joint_out_lr`) | JA 零初始化输出投影 |
| 其他 `joint_*` + `modality_embed` | `5e-6 × 1.0` | JA 其他参数（`joint_other_lr_multiplier=1.0`） |
| Backbone 参数 | `5e-6` | 基础学习率 |

对比 Stage 2 中 `joint_out_lr=2.5e-4` 和 `joint_other_lr_multiplier=10.0`，Stage 3 显著降低了所有学习率，体现了全参数微调阶段的保守策略。

#### 4.4.3 Stage 3 训练配置

| 参数 | 值 | 与 Stage 2 的差异 |
|------|-----|------------------|
| `freeze_non_joint` | `False` | 解冻所有参数 |
| `learning_rate` | `5e-6` | ↓10× |
| `joint_out_lr` | `1e-5` | ↓25× |
| `joint_other_lr_multiplier` | `1.0` | ↓10× |
| `branch_dropout_prob` | `0.05` | ↓4× |
| `lr_warmup_steps` | `500` | ↑ |
| `ema_decay` | `0.999` | vs 0.9999 |
| `joint_video_decay` | `True` | 新增 |
| 权重加载 | `--load_stage2_model_weights` | 仅加载模型权重，不加载优化器状态 |

### 4.5 三阶段训练总结

```mermaid
flowchart LR
    subgraph "Stage 1: 模态适应"
        S1_DESC["独立三分支 SFT<br/>无跨模态注意力<br/>fusion_mode=none"]
        S1_TRAIN["所有分支参数可训练<br/>共享 noise<br/>lr=1.5e-5"]
    end
    
    subgraph "Stage 2: 联合注意力训练"
        S2_DESC["添加 JA 模块<br/>Backbone 冻结<br/>仅训练 JA + modality_embed"]
        S2_TRAIN["零初始化 → 平滑过渡<br/>branch dropout=0.2<br/>lr=5e-5"]
    end
    
    subgraph "Stage 3: 全参数微调"
        S3_DESC["所有参数解冻<br/>video decay 渐进衰减<br/>差异化 LR"]
        S3_TRAIN["保守 LR (5e-6)<br/>branch dropout=0.05<br/>cosine decay 700步"]
    end
    
    S1_DESC --> |"权重迁移"| S2_DESC
    S2_DESC --> |"模型权重迁移<br/>(重置优化器)"| S3_DESC
```

**每个阶段都重置优化器/调度器**：从前一阶段仅加载模型权重（model-only checkpoint），不继承优化器状态。这允许每个阶段使用不同的学习率和调度策略。

---

## 5. 训练系统动态架构

### 5.1 训练 Forward Pass 完整流程

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant TR as RynnWorld4DTrainer
    participant SCH as FlowMatchScheduler
    participant MODEL as Tri-branch Transformer
    participant LOSS as Loss Computer
    
    DL->>TR: batch = {encoded_video, encoded_depth,<br/>encoded_flow, img/depth/flow_latent,<br/>null/text_embedding}
    
    Note over TR: 1. 噪声采样（三分支共享）
    TR->>TR: noise_video = randn_like(encoded_video)
    TR->>TR: noise_depth = noise_video.clone()
    TR->>TR: noise_flow = noise_video.clone()
    
    Note over TR: 2. 时间步采样
    TR->>SCH: sample random timestep t
    SCH-->>TR: σ_t = shift·s / (1 + (shift-1)·s)
    
    Note over TR: 3. 构建噪声潜变量
    TR->>TR: noisy_video = (1-σ)·video + σ·noise
    TR->>TR: noisy_depth = (1-σ)·depth + σ·noise
    TR->>TR: noisy_flow = (1-σ)·flow + σ·noise
    
    Note over TR: 4. First-frame conditioning
    TR->>TR: noisy_video[:, :1] = img_latent
    TR->>TR: noisy_depth[:, :1] = depth_latent
    TR->>TR: noisy_flow[:, :1] = flow_latent
    
    Note over TR: 5. Branch Dropout
    alt p < branch_dropout_prob
        TR->>TR: 随机选 depth 或 flow
        TR->>TR: 替换其帧[1:]为纯噪声
    end
    
    Note over TR: 6. CFG Dropout
    alt p < 0.15
        TR->>TR: text_emb = null_embedding
    end
    
    Note over TR: 7. Per-token timestep
    TR->>TR: timestep_per_token[frame=0] = 0
    TR->>TR: timestep_per_token[frame>0] = t
    
    TR->>MODEL: forward(noisy_video, noisy_depth, noisy_flow,<br/>timestep, text_emb)
    MODEL-->>TR: pred_video, pred_depth, pred_flow
    
    Note over LOSS: 8. 损失计算（排除首帧）
    TR->>LOSS: target = noise - clean (velocity)
    LOSS-->>TR: loss = MSE(pred_v, target_v)<br/> + MSE(pred_d, target_d)<br/> + λ_flow · MSE(pred_f, target_f)
```

### 5.2 共享 Noise 的设计分析

代码中三个分支使用**完全相同的噪声样本**：

```python
noise_video = torch.randn_like(encoded_videos)
noise_depth = noise_video.clone()
noise_flow = noise_video.clone()
```

`rynnworld4d_trainer.py` 中的注释解释了这一设计：相同的噪声**对齐了三个分支的去噪轨迹**。在 Flow Matching 框架下，从相同的噪声出发、沿相同的时间步去噪，使得三个分支在每个时间步都处于相似的噪声水平，有利于 Joint Cross-Modal Attention 的跨模态特征对齐。

### 5.3 分布式训练架构

```mermaid
flowchart TB
    subgraph "Stage 1: accelerate + DeepSpeed"
        ACC["accelerate launch<br/>--multi_gpu / --num_machines"]
        ACC --> DS_S1["DeepSpeed Config<br/>(configs_acc/*)"]
        DS_S1 --> GPU_S1["多 GPU / 多节点"]
    end
    
    subgraph "Stage 2/3: torchrun + ZeRO-2"
        TORCHRUN["torchrun<br/>--nproc_per_node --nnodes"]
        TORCHRUN --> DS_S2["DeepSpeed ZeRO-2<br/>+ CPU optimizer offload<br/>(configs_zero/zero2_offload.json)"]
        
        subgraph "跨节点 Rendezvous"
            MASTER["节点 0: 写入<br/>MASTER_PORT → NAS 文件"]
            WORKER["节点 1+: 读取<br/>NAS 文件 → MASTER_PORT"]
            MASTER --> |"共享 NAS"| WORKER
            Note_RDV["含过期文件保护<br/>(>3600s 则覆盖)"]
        end
    end
    
    subgraph "环境变量"
        ENV["WORLD_SIZE, RANK,<br/>MASTER_ADDR, MASTER_PORT,<br/>NPROC_PER_NODE"]
    end
```

**Stage 1 vs Stage 2/3 启动方式的差异**：
- **Stage 1** 使用 `accelerate launch`，依赖 HuggingFace Accelerate 的分布式抽象
- **Stage 2/3** 切换到 `torchrun`，使用 DeepSpeed ZeRO-2 with CPU optimizer offload

这可能是因为 Stage 2/3 需要更精细的显存管理（JA 模块增加了大量参数），ZeRO-2 with CPU offload 可以将优化器状态卸载到 CPU 来节省 GPU 显存。

### 5.4 EMA 机制

```mermaid
sequenceDiagram
    participant TRAIN as 训练循环
    participant EMA as EMAModel
    participant CPU as CPU 内存
    participant GPU as GPU
    
    Note over TRAIN: 初始化
    TRAIN->>EMA: 创建 EMAModel（仅跟踪可训练参数）
    EMA->>CPU: shadow_params → CPU（节省 GPU 显存）
    TRAIN->>TRAIN: 记录参数名 → self._ema_param_names
    
    Note over TRAIN: 每个 optimizer step
    TRAIN->>GPU: 正常前向/反向/更新
    
    alt sync_gradients == True
        TRAIN->>CPU: 复制可训练参数到 CPU（non_blocking）
        TRAIN->>EMA: ema_model.step(current_params)
        Note over EMA: shadow = decay × shadow + (1-decay) × current
    end
    
    Note over TRAIN: 保存 checkpoint
    TRAIN->>CPU: 构建 {param_name: shadow_param} dict
    TRAIN->>TRAIN: 保存为 ema_weights.pt
```

**关键设计**：EMA 仅跟踪**可训练参数**，不跟踪冻结的 backbone（在 Stage 2 中可节省大量 CPU 内存）。Shadow 参数存储在 CPU 以节省 GPU 显存。

### 5.5 Checkpoint Save/Load Hooks

`RynnWorld4DTrainer` 注册了自定义的保存/加载钩子，将模型权重分离为两部分：

1. **LoRA 权重** → `high_noise_lora/` 目录
2. **三分支/JA 权重** → `rynnworld4d_layers.bin` 文件

分支权重的筛选关键词：
```python
branch_keywords = ("depth", "flow", "video_to_depth", "video_to_flow",
                   "depth_to_video", "flow_to_video", "joint_")
```

### 5.6 Monkey Patching 详解

代码在 **import 时** 对 diffusers 库的两个方法进行了 monkey patch：

#### Patch 1：FP32 时间嵌入稳定性

```python
# 原始 diffusers: 使用混合精度下的 bf16 计算时间嵌入
# Patched: 强制 FP32 计算 time embedding
with torch.autocast(device_type=timestep.device.type, dtype=torch.float32, enabled=True):
    temb = self.time_embedder(timestep)
    timestep_proj = self.time_proj(self.act_fn(temb))
```

**原因**：时间嵌入中的正弦/余弦位置编码在 bf16 下可能产生数值不稳定，尤其是在高频分量处。FP32 计算确保了时间信号的精确性。

#### Patch 2：添加 control_video_latent 参数

```python
# 原始 diffusers: WanTransformer3DModel.forward 仅接受 hidden_states
# Patched: 新增 control_video_latent 参数，支持首帧条件注入
def wan_forward(self, hidden_states, ..., control_video_latent=None, is_concat=False):
    if control_video_latent is not None:
        if is_concat:
            hidden_states = torch.cat([hidden_states, control_video_latent], dim=1)
            hidden_states = self.control_patch_embedding(hidden_states)
        else:
            control_emb = self.control_patch_embedding(control_video_latent)
            hidden_states = hidden_states + control_emb
```

### 5.7 训练诊断日志

每 50 步，trainer 输出以下诊断信息：

1. **Gate 值**：`mean(tanh(joint_gate_*))` 跨所有 blocks 的平均值，确认跨模态通道是否激活
2. **Joint Ratio**：`mean(_joint_ratio_*)` 跨所有 blocks 的平均值，衡量 JA 输出占总隐藏状态的比例
3. **Decay 值**：当前的 `joint_gate_video_decay` 值

这些诊断指标无需 GPU→CPU 同步（使用 `torch.tensor` 在 device 上计算），不会影响训练吞吐量。

---

## 6. 推理流水线

### 6.1 推理流程总览

> 源码位置：`core/inference/rynnworld4d.py` (`RynnWorld4DPipeline`) 和 `inference-sft.py`

```mermaid
sequenceDiagram
    participant USER as 用户输入
    participant PIPE as RynnWorld4DPipeline
    participant VAE as VAE Encoder/Decoder
    participant T5 as UMT5 Text Encoder
    participant MODEL as Tri-branch Transformer
    participant SCH as Scheduler
    
    USER->>PIPE: RGB 图像 + 深度图 + prompt
    
    Note over PIPE: 1. 编码条件
    PIPE->>T5: encode_prompt(prompt)
    T5-->>PIPE: text_emb, null_text_emb
    PIPE->>VAE: encode(RGB 图像)
    VAE-->>PIPE: rgb_first_frame_latent
    PIPE->>VAE: encode(深度图)
    VAE-->>PIPE: depth_first_frame_latent
    PIPE->>VAE: encode(白色图 = 零光流)
    VAE-->>PIPE: flow_first_frame_latent
    
    Note over PIPE: 2. 初始化噪声潜变量
    PIPE->>PIPE: latent_video = randn(1, C_z, F_lat, H_lat, W_lat)
    PIPE->>PIPE: latent_depth = randn(...)
    PIPE->>PIPE: latent_flow = randn(...)
    
    Note over PIPE: 3. 去噪循环
    loop 每个时间步 t (从 T 到 0)
        PIPE->>PIPE: 替换 frame[0] = 首帧条件
        PIPE->>PIPE: 构建 per-token timestep
        
        Note over PIPE: CFG: 两次前向
        PIPE->>MODEL: forward(conditional)
        MODEL-->>PIPE: pred_v_c, pred_d_c, pred_f_c
        PIPE->>MODEL: forward(unconditional)
        MODEL-->>PIPE: pred_v_u, pred_d_u, pred_f_u
        
        Note over PIPE: CFG 组合
        PIPE->>PIPE: pred = pred_u + scale × (pred_c - pred_u)
        
        PIPE->>SCH: step(pred_v, t) → latent_v
        PIPE->>SCH: step(pred_d, t) → latent_d
        PIPE->>SCH: step(pred_f, t) → latent_f
    end
    
    Note over PIPE: 4. 最终首帧替换 + VAE 解码
    PIPE->>PIPE: 替换 frame[0] = 首帧条件
    PIPE->>VAE: decode(latent_video) → RGB 视频
    PIPE->>VAE: decode(latent_depth) → 深度视频
    PIPE->>VAE: decode(latent_flow) → 光流视频
    
    PIPE-->>USER: rgb.mp4, depth.mp4, flow.mp4
```

### 6.2 关键实现细节

#### 6.2.1 三分支独立 Scheduler Step

推理时每个分支使用**独立的 scheduler step**：

```python
latents_video = self.scheduler.step(noise_pred_video, t, latents_video).prev_sample
latents_depth = self.scheduler.step(noise_pred_depth, t, latents_depth).prev_sample
latents_flow = self.scheduler.step(noise_pred_flow, t, latents_flow).prev_sample
```

每个分支在各自的潜变量空间中独立去噪，跨模态一致性通过 Joint Attention 在 Transformer 内部实现。

#### 6.2.2 省显存的逐分支 VAE 解码

```python
# 逐分支解码以节省显存
video_output = self.vae.decode(latents_video)
depth_output = self.vae.decode(latents_depth)
flow_output = self.vae.decode(latents_flow)
```

三个分支共享同一个 VAE 解码器，但逐一解码以避免同时占用三倍显存。

#### 6.2.3 Checkpoint 加载策略

`inference-sft.py` 支持多种 checkpoint 格式：

1. **DeepSpeed ZeRO 格式**：从 `mp_rank_00_model_states.pt` 加载，去除 `module.` 前缀
2. **Safetensors 分片**：使用 `mmap=True` 内存映射加载，减少内存峰值
3. **EMA 权重**：从 `ema_weights.pt` 加载 shadow 参数
4. **Joint Attention 消融**：支持 `disable_joint=True` 关闭 JA 模块用于消融实验

---

## 7. Policy Head 深入分析

### 7.1 整体架构

```mermaid
flowchart TB
    subgraph "输入"
        RGB["RGB 观测<br/>(B, obs_seq, 3, H, W)"]
        DEPTH_C["深度条件<br/>(预计算 or DA3 实时估计)"]
        TEXT["任务描述文本<br/>(UMT5 embedding)"]
        PROPRIO["本体感知<br/>(B, 54)"]
    end
    
    subgraph "WanFeatureExtractor (冻结 ~5B params)"
        VAE_ENC["VAE 编码<br/>RGB→latent, Depth→latent, White→latent"]
        LATENT_BUILD["三分支 Latent 构建<br/>frame[0]=条件, frame[1:]=噪声"]
        TF_FWD["RynnWorld-4D Transformer<br/>t=500, 提取 block 15"]
        CONCAT["三分支特征拼接<br/>channel concat → 3d 维度"]
    end
    
    subgraph "Video_Former_3D (可训练)"
        PROJ["MLP 投影<br/>3d → 384"]
        PERCEIVER["Perceiver Resampler<br/>6 层 CrossAttn + TempSelfAttn"]
        LATENT_Q["Learnable Queries<br/>336 tokens"]
    end
    
    subgraph "FlowMatchingPolicy (可训练)"
        ENCODER["Encoder (4 layers)<br/>[goal | state | proprio]<br/>Non-causal Self-Attention"]
        DECODER["Decoder (4 layers)<br/>Noisy actions + timestep (AdaLN-Zero)<br/>Causal Cross-Attention"]
        ODE["4-step Euler ODE<br/>t: 0→1"]
    end
    
    subgraph "输出"
        ACTIONS["动作序列<br/>(B, 10, 54)"]
    end
    
    RGB --> VAE_ENC
    DEPTH_C --> VAE_ENC
    VAE_ENC --> LATENT_BUILD --> TF_FWD --> CONCAT
    TEXT --> ENCODER
    PROPRIO --> ENCODER
    CONCAT --> PROJ --> PERCEIVER
    LATENT_Q --> PERCEIVER
    PERCEIVER --> ENCODER
    ENCODER --> DECODER --> ODE --> ACTIONS
```

### 7.2 特征提取：`WanFeatureExtractor`

> 源码位置：`rynnworld4d_policy/policy_models/module/wan_feature_extractor.py`

#### 7.2.1 单次前向传播特征提取

Policy 不进行完整的扩散去噪循环。它在**固定时间步 $t=500$** 做单次前向传播，提取中间层特征：

```python
# vpp_policy.py extract_predictive_feature 中
features = self.TVP_encoder(pixel_values, texts, timestep=500, ...)
```

**物理意义**：$t=500$ 位于去噪过程的中间位置（总步数 1000），此时模型既有足够的噪声信号以提取全局结构信息，又保留了足够的数据信号以获取细节。

#### 7.2.2 Forward Hook + Early Exit 机制

```python
class _StopForward(Exception):
    """Sentinel exception to abort transformer forward early."""
    pass

def _register_hook(self, block_idx):
    def hook(module, input, output):
        # 捕获输出
        self._features[block_idx] = output
        if block_idx == max(self._extract_block_idx):
            raise _StopForward()  # 提前终止前向传播
    return block.register_forward_hook(hook)
```

当所有需要的 block 输出都被捕获后，通过抛出异常中止 Transformer 的剩余前向传播。

**物理删除优化**：当 `num_inference_layers` 被设置时（默认配置中为 20），直接从 `nn.ModuleList` 中删除不需要的 blocks：

```python
if num_inference_layers is not None:
    self.transformer.blocks = self.transformer.blocks[:num_inference_layers]
```

由于特征从 block 15 提取，blocks 16-29 在推理时是不必要的。物理删除节省了 GPU 显存和加载时间。

#### 7.2.3 三分支 Latent 构建

```python
def _build_rynnworld4d_latents(self, pixel_values, depth_cond):
    # Video: frame[0] = VAE(RGB condition), frame[1:] = noise
    latents_video = torch.randn(...)
    latents_video[:, :, :1] = self._vae_encode_single_frame(pixel_values[:, -1])
    
    # Depth: frame[0] = VAE(DA3 depth estimation), frame[1:] = noise
    latents_depth = torch.randn(...)
    depth_latent = self._estimate_depth_latent(pixel_values[:, -1])
    latents_depth[:, :, :1] = depth_latent
    
    # Flow: frame[0] = VAE(white image = zero flow), frame[1:] = noise
    latents_flow = torch.randn(...)
    white_latent = self._vae_encode_single_frame(white_image)
    latents_flow[:, :, :1] = white_latent
    
    return latents_video, latents_depth, latents_flow
```

#### 7.2.4 DA3 深度估计集成

```python
def _get_da3_model(self):
    # 延迟加载 Depth-Anything-3
    # 支持 int8 量化（通过 bitsandbytes）
    if self.da3_quantize:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = DepthAnything3.from_pretrained(..., quantization_config=quant_config)
    else:
        model = DepthAnything3.from_pretrained(...)
    return model
```

DA3 模型按需加载，支持 int8 量化以节省显存。

#### 7.2.5 特征输出维度

从 block 15 提取的三分支特征经过拼接后的维度变化：

| 步骤 | 形状 | 说明 |
|------|------|------|
| Block 15 输出（每分支） | `(B, F×H'×W', d)` | 展平的序列 |
| Reshape | `(B, F, d, H', W')` | 恢复空间结构 |
| 三分支拼接 | `(B, F, 3d, H', W')` | Channel concat |
| Rearrange | `(B, F, H'×W', 3d)` | 为 Video_Former 准备 |

其中 $d$ 为 Transformer 隐藏维度（Wan 2.2-5B 中 $d=3072$），最终 `condition_dim = 3d = 9216`。

### 7.3 Video_Former_3D (Perceiver Resampler)

> 源码位置：`rynnworld4d_policy/policy_models/module/Video_Former.py`

#### 7.3.1 架构设计

```mermaid
flowchart TB
    INPUT["输入特征<br/>(B, F, N_spatial, 9216)"]
    
    PROJ["goal_emb MLP<br/>9216 → 768 → 384"]
    
    TPE["+ Temporal Pos Embed<br/>(num_time_embeds, 1, 384)"]
    
    FLAT["Flatten to (B×F, N, 384)"]
    
    LATENTS["Learnable Queries<br/>(F, Q/F, 384)<br/>repeat → (B×F, Q/F, 384)"]
    
    subgraph "× depth 层"
        PA["PerceiverAttention<br/>Q=latents, KV=[features; latents]"]
        TA["TemporalSelfAttention<br/>reshape to (B, F, Q/F, 384)"]
        FFW["Feed-Forward<br/>384 → 384"]
    end
    
    NORM["LayerNorm"]
    
    OUTPUT["输出<br/>(B, 336, 384)"]
    
    INPUT --> PROJ --> TPE --> FLAT
    LATENTS --> PA
    FLAT --> PA --> TA --> FFW
    FFW --> PA
    FFW --> NORM --> OUTPUT
```

**关键参数**（来自 `train_config.yaml`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 384 | 内部维度 |
| `depth` | 6 | Perceiver 层数 |
| `condition_dim` | 9216 | 输入特征维度 |
| `num_latents` | 336 | 总 latent query 数 |
| `num_frame` | 21 | 帧数（81帧视频经 VAE 压缩后） |
| `num_time_embeds` | 21 | 时间位置嵌入数 |
| `heads` | 8 | 注意力头数 |
| `dim_head` | 64 | 每头维度 |
| `use_temporal` | True | 启用时间自注意力 |

**Perceiver Attention 的工作原理**：

1. Queries 来自 learnable latent tokens（$Q/F$ 个 per frame，共 $Q=336$）
2. Keys/Values 来自 visual features + latent tokens 的拼接（self + cross）
3. Cross-attention 将高维空间特征压缩到低维 latent 空间

**时间自注意力**：在 Perceiver Attention 之后，latent tokens 在时间维度上做 self-attention，使不同帧的信息能够交互。

### 7.4 FlowMatchingPolicy（动作生成）

> 源码位置：`rynnworld4d_policy/policy_models/edm_diffusion/flow_matching.py`

#### 7.4.1 DiffusionTransformer 架构

```mermaid
flowchart TB
    subgraph "Encoder (4 layers, Non-causal)"
        E_GOAL["Goal Embedding<br/>MLP: 4096 → 768 → 384<br/>(32 tokens)"]
        E_STATE["State Embedding<br/>Linear: 384 → 384<br/>(336 tokens)"]
        E_PROPRIO["Proprio Embedding<br/>MLP: 54 → 768 → 384<br/>(1 token)"]
        E_CONCAT["Concat → (B, 369, 384)"]
        E_POS["+ Positional Embedding"]
        E_SA["4× Self-Attention Block"]
        
        E_GOAL --> E_CONCAT
        E_STATE --> E_CONCAT
        E_PROPRIO --> E_CONCAT
        E_CONCAT --> E_POS --> E_SA
    end
    
    subgraph "Decoder (4 layers, Causal + Cross-Attn)"
        D_SIGMA["σ_emb: Sinusoidal → MLP<br/>(timestep embedding)"]
        D_ACT["Action Embedding<br/>Linear: 54 → 384<br/>(10 tokens)"]
        D_CA["4× ConditionedBlock<br/>(AdaLN-Zero + Causal Self-Attn<br/>+ Cross-Attn to Encoder)"]
        D_PRED["Action Prediction<br/>Linear: 384 → 54"]
        
        D_SIGMA --> D_CA
        D_ACT --> D_CA
        E_SA --> D_CA
        D_CA --> D_PRED
    end
```

**Block size 计算**：
$$\text{block\_size} = \underbrace{32}_{\text{goal}} + \underbrace{10}_{\text{action}} + \underbrace{1 \times 336}_{\text{obs}} + \underbrace{2}_{\text{padding}} = 380$$

#### 7.4.2 Flow Matching 训练与采样

**训练**（`loss` 方法）：

$$\begin{aligned}
t &\sim \text{Uniform}[10^{-4}, 1] \\
\epsilon &\sim \mathcal{N}(0, I) \\
x_t &= (1 - t) \epsilon + t \cdot x_1 \quad \text{（线性插值）} \\
v_{target} &= x_1 - \epsilon \\
\mathcal{L} &= \|v_\theta(x_t, t) - v_{target}\|^2
\end{aligned}$$

注意这里的方向与 World Model 的 Flow Matching 不同：
- **World Model**：$z_t = (1 - \sigma_t) z_0 + \sigma_t \epsilon$，目标 $v = \epsilon - z_0$（noise - clean）
- **Policy**：$x_t = (1 - t) \epsilon + t \cdot x_1$，目标 $v = x_1 - \epsilon$（data - noise）

两者在数学上是等价的（方向相反、符号不同），但 Policy 使用了更标准的 Flow Matching 形式。

**推理**（`sample` 方法）：4 步 Euler ODE 积分，从 $t=0$（噪声）到 $t=1$（数据）：

$$\begin{aligned}
\Delta t &= 1/4 \\
\text{for } i &= 0, 1, 2, 3: \\
\quad t_i &= i \cdot \Delta t \\
\quad v_i &= v_\theta(x_{t_i}, t_i) \\
\quad x_{t_{i+1}} &= x_{t_i} + v_i \cdot \Delta t
\end{aligned}$$

#### 7.4.3 Classifier-Free Guidance (目标 Dropout)

```python
def mask_cond(cond, goal_drop=0.1):
    # 以概率 goal_drop 将整个 goal embedding 置零
    mask = torch.bernoulli(torch.ones(B) * (1 - goal_drop))
    return cond * mask.unsqueeze(-1).unsqueeze(-1)
```

训练时以 10% 概率丢弃目标条件，推理时可使用 CFG 增强目标导向性。

### 7.5 实时控制架构

#### 7.5.1 延迟分解

论文报告的在 NVIDIA RTX 5090 上的推理延迟（使用 FP8 量化 + FlashAttention 3）：

| 组件 | 延迟 (ms) | 占比 |
|------|-----------|------|
| DA3 深度估计 | 85 | 7.7% |
| VAE 编码 + Latent 准备 | 18 | 1.6% |
| RynnWorld-4D Transformer | 990 | 89.5% |
| 特征 Reshape + Concat | 1 | 0.1% |
| Video_Former (Flow Former) | 4 | 0.4% |
| Action Flow Matching Head | 8 | 0.7% |
| **总计** | **~1,106** | **100%** |

#### 7.5.2 Action Chunking

```python
# vpp_policy.py step() 方法
def step(self, obs, goal):
    if self.rollout_step_counter % self.multistep == 0:
        # 每 multistep 步重新规划
        self.plan = self.eval_forward(obs, goal)  # (B, 10, 54)
    
    action = self.plan[:, self.rollout_step_counter % self.multistep]
    self.rollout_step_counter += 1
    return action
```

每次规划生成 $K=10$ 步动作序列，机器人以 50 Hz 执行缓存的动作块，同时在后台计算下一个动作块。有效控制频率：

$$f_{eff} = K \times f_{plan} = 10 \times \frac{1}{1.106\text{s}} \approx 9 \text{ Hz}$$

#### 7.5.3 CPU Offload 推理

> 源码位置：`rynnworld4d_policy/inference_cpu_offload.py`

提供了 int8 量化 + 手动 CPU offload 的推理方案：

```python
class Int8Linear(nn.Module):
    """Per-channel int8 量化线性层"""
    def __init__(self, weight_int8, scale, bias):
        self.weight_int8 = weight_int8  # int8 权重
        self.scale = scale              # 每通道缩放因子
    
    def forward(self, x):
        # 即时反量化 + 矩阵乘法
        return F.linear(x, self.weight_int8.float() * self.scale, self.bias)
```

手动 sequential offload：
1. 将小模块（Video_Former, policy head）常驻 GPU
2. 大模块（RynnWorld-4D transformer）临时从 CPU 移至 GPU
3. 前向传播完成后立即移回 CPU

---

## 8. 论文-代码对照分析

### 8.1 论文描述与代码实现的差异

| # | 论文描述 | 代码实现 | 分析 |
|---|---------|---------|------|
| 1 | "30-layer DiT, $d=3072$" | 构造器默认 `num_layers=40, num_attention_heads=40 → inner_dim=5120` | 构造器默认值为 Python 参数默认值，实际由 `from_pretrained()` 加载的预训练模型 config 覆盖。module_joint.py 注释中使用 `dim=5120` 估算参数，可能基于更大变体 |
| 2 | "JA 每隔 3 个 block" | `joint_every_n_layers=3`，从第 0 层开始 | 完全一致 |
| 3 | "Stage 2 LR $5 \times 10^{-5}$" | `lr=5e-5`，`joint_out_lr=2.5e-4` | 代码额外指定了 JA 输出投影的独立 LR，论文未提及 |
| 4 | "λ_flow = 0.5 (Stage 1), 1.0 (Stage 2/3)" | `loss_weight_flow=0.5` (stage1), `1.0` (stage2/3) | 完全一致 |
| 5 | "Branch Dropout $p=0.2$ (Stage 2)" | `branch_dropout_prob=0.2` | 完全一致 |
| 6 | "EMA decay 0.9999" | Stage 1/2: `0.9999`；Stage 3: `0.999` | Stage 3 使用更快的 EMA 衰减，论文未区分 |
| 7 | "train resolution 81×480×640" | Stage 1 脚本: `25×480×832` | Stage 1 使用更短的视频（25帧）和更宽的分辨率（832），可能为节省显存 |
| 8 | Stage 2 "Joint unidirectional" | `joint_unidirectional=True` | 代码中 Stage 2 是单向的（仅 depth/flow 从 video 获取信息），论文将其描述为双向 JA |
| 9 | Policy "block 15 features" | `wan_extract_block_idx=15` | 完全一致 |
| 10 | Policy "timestep t=500" | `timestep=500` | 完全一致 |

### 8.2 代码中存在但论文未提及的实现细节

1. **Monkey Patching**：FP32 时间嵌入和 `control_video_latent` 注入，属于工程稳定性措施
2. **差异化学习率**：`joint_out_lr` 和 `joint_other_lr_multiplier` 的精细调控
3. **`_StopForward` 异常**：Policy 的 early exit 机制
4. **物理删除 blocks**：`num_inference_layers=20` 直接截断 block list
5. **Int8 量化推理**：`inference_cpu_offload.py` 中的自定义 int8 线性层
6. **跨节点 rendezvous**：通过 NAS 文件同步 `MASTER_PORT` 的分布式协调机制
7. **`joint_gate_video_decay` vs `joint_gate_video`**：decay buffer（不可训练）vs gate parameter（可训练）的分离设计
8. **半余弦 LR 调度**：`rynnworld4d_trainer.py` 中自定义的 lambda scheduler 使用 `cos(π × 0.5 × 2.0 × progress)`

### 8.3 消融实验对应的代码开关

| 消融实验 | 代码配置 | 效果（论文报告） |
|---------|---------|-----------------|
| Independent Branches (无 JA) | `fusion_mode=none`（仅 Stage 1） | AbsRel 0.737↑, AEPE 0.247↑ |
| w/o Modality Adaptation (跳过 Stage 1) | 直接从预训练加载到 Stage 2 | δ₁ 0.479↓ |
| w/o 4D Pre-training | 仅用机器人任务数据 | AEPE 0.729 (崩溃) |
| w/o RoPE in JA | `joint_use_rope=False` | δ₁ 0.450↓, AEPE 0.210↑ |
| Shared FFN | `share_ffn=True` | AbsRel 0.580↑, δ₁ 0.380↓ (崩溃) |
| w/o RynnWorld-4D backbone | Policy 中使用 ResNet-18 替代 | 所有任务大幅下降 |
| RGB only | 仅使用 video 分支特征 | 次优 |
| RGB + Depth | 使用 video + depth 特征 | 空间精度任务提升 |
| RGB + Optical Flow | 使用 video + flow 特征 | 运动敏感任务提升 |

### 8.4 超参数对照

| 超参数 | 论文值 | Stage 1 脚本 | Stage 2 脚本 | Stage 3 脚本 | Policy 配置 |
|--------|--------|-------------|-------------|-------------|-------------|
| Learning Rate | 2e-5 (S1), 5e-5 (S2), 1e-5 (S3) | 1.5e-5 | 5e-5 | 5e-6 | 1e-4 |
| Warmup Steps | 500 (S1), 200 (S2), 500 (S3) | 300 | 200 | 500 | 2% of total |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 0.05 |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.9 |
| Batch Size / GPU | — | — | — | — | 1 |
| Gradient Accum | — | 4 | 2 | — | — |
| Resolution | 81×480×640 | 25×480×832 | — | — | 81×480×640 |
| EMA Decay | 0.9999 | 0.9999 | 0.9999 | 0.999 | 0.9999 |

---

## 9. 关键设计决策与工程洞察

### 9.1 共享 Noise 的跨模态对齐

三个分支使用相同的噪声样本是一个看似简单但影响深远的决策。在 Flow Matching 框架下，噪声样本定义了从数据分布到噪声分布的传输映射。共享噪声意味着三个分支在每个时间步都处于**相同的噪声水平**，使得 Joint Cross-Modal Attention 可以在对齐的信号-噪声比下进行特征交互。

如果使用独立噪声，三个分支在同一时间步可能处于不同的信噪比状态，JA 可能需要额外学习处理这种不对齐，增加了学习难度。

### 9.2 Gate 初始化的梯度工程

`gate=1.0 + OutProj=zero` 的组合是精心设计的工程解决方案。与 ControlNet 的单层 zero-init 不同，RynnWorld-4D 使用了两层控制（gate × projection），需要避免鞍点死锁。这一设计确保了：

1. **初始输出为零**：$\text{OutProj}(x) \cdot \tanh(1) = 0 \cdot 0.762 = 0$，平滑过渡
2. **梯度非零**：$\nabla_{W_{out}} \mathcal{L} = \tanh(1) \cdot ... \neq 0$，立即可学习
3. **Gate 可调节**：gate 从 1.0 开始，通过训练自适应地调节跨模态信息流强度

### 9.3 三阶段训练的渐进式设计哲学

三阶段设计体现了深度学习中的**课程学习**（curriculum learning）思想：

1. **Stage 1**：让每个分支先学会各自模态的生成能力（简单任务）
2. **Stage 2**：冻结已学会的分支，专注训练跨模态交互（中等任务）
3. **Stage 3**：全部解冻，联合优化，同时通过 decay 渐进移除对称性（困难任务）

每阶段重置优化器状态，避免了前一阶段的动量/方差累积对新阶段的干扰。

### 9.4 EMA CPU Offload 的显存优化

将 EMA shadow 参数存储在 CPU 而非 GPU 上，对于一个约 5B 参数的模型（bf16 下约 10GB），可以节省**约 10GB GPU 显存**。代价是每个 optimizer step 需要一次 GPU→CPU 的参数复制（使用 `non_blocking=True` 异步传输，开销极小）。

### 9.5 DeepSpeed ZeRO-2 vs ZeRO-3

- **Stage 1** 使用 ZeRO-3（或 Accelerate 默认），适合多 GPU 训练大模型
- **Stage 2/3** 使用 ZeRO-2 + CPU optimizer offload

ZeRO-2 仅分片优化器状态和梯度，不分片模型参数。配合 CPU offload，优化器状态（~30GB for AdamW with 5B params）被卸载到 CPU，而模型参数保留在 GPU 上。这比 ZeRO-3（分片所有内容但通信开销更大）在 Stage 2/3 中更高效，因为 Stage 2 只训练少量 JA 参数，优化器状态相对较小。

### 9.6 Policy 的 Single Forward Pass 设计

传统方法（如 SuSIE、UniPi）需要完整的扩散去噪循环来生成未来帧，然后从像素中提取信息。RynnWorld-4D-Policy 的设计直接跳过了这一过程：

1. 在固定时间步 $t=500$ 做单次前向传播
2. 提取中间层（block 15）特征
3. 这些特征已经包含了模型对未来场景演变的**内部表征**

这将推理时间从数十次去噪步骤压缩到单次前向传播，是实现实时控制的关键。

---

## 参考来源

- **论文**: Zhao et al., "RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation", arXiv 2607.06559
- **论文 TeX 源码**: `b/d/p/TeX_Source/`（Sec/1-intro.tex ~ 5-conclusion.tex, Appendix.tex）
- **代码**: 本地代码库各文件（以实际代码实现为准）
- **Wan 2.2 backbone**: HuggingFace diffusers (`WanTransformer3DModel`, `WanImageToVideoPipeline`)
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
- **ControlNet zero-init**: Zhang & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023
- **Depth Anything 3**: `rynnworld4d_policy/third_party/Depth-Anything-3/`
- **DPFlow**: 论文引用的光流估计方法
- **Rectified Flow**: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023
