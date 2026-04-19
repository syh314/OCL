# VLA-Adapter 代码结构图

下面这张图把这个项目整理成一张“仓库分层 + 训练主链 + 推理闭环”的总图，适合你讲代码时先总览，再按主线展开。

```mermaid
flowchart TB
    subgraph Entry["入口层"]
        F1["vla-scripts/finetune.py<br/>训练主入口"]
        F2["vla-scripts/train.py<br/>原始 OpenVLA 训练入口"]
        F3["vla-scripts/deploy.py<br/>部署入口"]
        F4["vla-scripts/vla_evaluation.py<br/>评测入口"]
    end

    subgraph Data["数据层"]
        D1["prismatic/vla/datasets/rlds/dataset.py<br/>读取 RLDS / TFDS 轨迹"]
        D2["prismatic/vla/datasets/datasets.py<br/>RLDSDataset / RLDSBatchTransform"]
        D3["prismatic/vla/action_tokenizer.py<br/>连续动作 ↔ action token"]
        D4["prismatic/vla/constants.py<br/>平台常量: ACTION_DIM / NUM_ACTIONS_CHUNK"]
    end

    subgraph Model["模型主干"]
        M1["prismatic/extern/hf/modeling_prismatic.py<br/>PrismaticForConditionalGeneration"]
        M2["OpenVLAForActionPrediction<br/>动作语义与 predict_action()"]
        M3["PrismaticVisionBackbone<br/>图像 -> patch features"]
        M4["PrismaticProjector<br/>视觉特征 -> LLM 维度"]
        M5["AutoModelForCausalLM<br/>语言模型主体"]
        M6["action_queries<br/>动作占位 token"]
    end

    subgraph Adapter["VLA-Adapter 增量层"]
        A1["prismatic/models/action_heads.py<br/>L1RegressionActionHead"]
        A2["prismatic/models/projectors.py<br/>ProprioProjector"]
        A3["prismatic/models/film_vit_wrapper.py<br/>FiLM 视觉调制"]
    end

    subgraph Runtime["推理与部署辅助"]
        R1["experiments/robot/openvla_utils.py<br/>加载模型 / 恢复模块 / 反归一化"]
        R2["dataset_statistics.json<br/>动作归一化统计"]
        R3["processor / image transform<br/>图像预处理"]
    end

    X1["原始机器人轨迹<br/>image + instruction + action + proprio"]
    X2["训练 batch<br/>pixel_values / input_ids / labels / actions"]
    X3["多模态 hidden states"]
    X4["连续动作输出"]
    X5["机器人可执行动作"]

    X1 --> D1
    D1 --> D2
    D3 --> D2
    D4 --> D2
    D2 --> X2

    F1 --> D2
    F1 --> M2
    F1 --> A1
    F1 --> A2
    F1 --> A3

    X2 --> M2
    M2 -.继承.-> M1
    M1 --> M3
    M3 --> M4
    A3 -.可选调制.-> M3
    M1 --> M6
    M1 --> M5

    M2 --> X3
    X3 --> A1
    A2 --> A1
    A1 --> X4

    F3 --> R1
    F4 --> R1
    R1 --> M2
    R1 --> A1
    R1 --> A2
    R1 --> R2
    R1 --> R3
    X4 --> R1
    R1 --> X5
```

## 读图说明

### 1. 最重要的一条训练主线

你讲代码时，最推荐按下面这条链路顺：

1. `vla-scripts/finetune.py`
2. `prismatic/vla/datasets/datasets.py` 里的 `RLDSBatchTransform`
3. `prismatic/extern/hf/modeling_prismatic.py` 里的 `PrismaticForConditionalGeneration.forward()`
4. `prismatic/extern/hf/modeling_prismatic.py` 里的 `OpenVLAForActionPrediction`
5. `prismatic/models/action_heads.py` 里的 `L1RegressionActionHead.predict_action()`

也就是：

`finetune() -> RLDSBatchTransform -> PrismaticForConditionalGeneration.forward() -> OpenVLAForActionPrediction -> L1RegressionActionHead.predict_action()`

### 2. 三层关系怎么理解

- `Prismatic`：多模态底座，负责图像和文本如何进入同一个模型。
- `OpenVLA`：建立在 `Prismatic` 上的机器人动作版本，增加动作 token、动作统计量和动作预测接口。
- `VLA-Adapter`：在 `OpenVLA` 之上新增的适配器层，重点是连续动作回归头、proprio projector、FiLM、多图像输入。

所以可以把整个项目压缩成一句话：

`Prismatic -> OpenVLA -> VLA-Adapter`

### 3. 为什么这张图里 action head 单独画一层

因为这篇工作最重要的变化，不只是让 VLM 输出离散动作 token，而是在 VLM 提供的多层 hidden states 之上，再加一个连续动作回归头。也就是说：

- VLM 主体负责多模态融合
- `OpenVLA` 负责动作语义占位与动作接口
- `L1RegressionActionHead` 负责最终连续动作回归

### 4. 推理闭环怎么收

推理和部署时，`openvla_utils.py` 会把下面这些东西重新拼起来：

- `OpenVLAForActionPrediction`
- `L1RegressionActionHead`
- `ProprioProjector`
- `dataset_statistics.json`
- `processor / image transform`

最后输出的不是归一化动作，而是经过统计量反归一化后的机器人可执行动作。

