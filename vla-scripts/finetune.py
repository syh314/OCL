"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
"""

import os
import time
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import numpy as np
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla



# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 决定是否使用更小型的 VLM 变体
    resum_vla_path: str = "openvla/openvla-7b"       # 当需要从某个特定的 VLA 检查点恢复训练时使用的路径

    # Dataset
    # AutoDL 默认数据盘（可通过 --data_root_dir 覆盖）
    data_root_dir: Path = Path("/root/autodl-tmp/data")
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    phase1_path: str = "None"

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0.1                       # 学习率预热步数
    num_steps_before_decay: int = 100000             # 在学习率降低 10 倍之前持续的步数
    grad_accumulation_steps: int = 1                 # 梯度累积步数，用于在显存有限时模拟更大的批处理量。
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # 验证和保存模型的频率（以步数为单位）
    val_time_limit: int = 180                        # 允许运行的最长时间
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # 是否从断点恢复训练，以及从哪一步开始恢复
    image_aug: bool = True                           # 是否开启图像增强
    diffusion_sample_freq: int = 50                  # 扩散模型采样频率
    # 当使用扩散模型（Diffusion Model）生成动作时，训练过程通常很快，但“采样”（即从噪声中生成完整动作）需要经过多个去噪步骤

    # LoRA
    use_lora: bool = False                           # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!
    # False (默认)：保存时只存一个小巧的 LoRA 适配器文件（几百 MB）。优点是存得快、占地小；缺点是推理时需要加载原始大模型 + 这个适配器。
    # True：保存时会将 LoRA 权重加回到原模型，生成一个完整的、可以直接独立使用的模型文件（如 7B 模型约为 14GB）。

    # Full Finetune
    use_fz: bool = False                             #  LoRA fine-tuning？

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # revision version
    use_pro_version: bool = True                             # the version number
    phase: str = "Training"
    seed: int = 7

    # Slot bottleneck
    use_slot_bottleneck: bool = True
    # 重建损失 α：total_loss = action_loss + α * slot_recon_loss
    # 经验：先用 0.1，根据训练初期两个 loss 的比例调到 recon 项 ≈ 0.1~0.5×action_loss
    slot_recon_loss_weight: float = 0.1
    # fmt: on



def remove_ddp_in_checkpoint(state_dict) -> dict:
    # 把 DDP 多卡训练保存的模型权重里的 module. 前缀删掉，让单卡也能正常加载！
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict



def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.config_file_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def apply_runtime_slot_overrides(model_config, cfg: FinetuneConfig) -> None:
    """Apply experiment-time slot toggles to the HF config before model instantiation."""
    model_config.use_slot_bottleneck = cfg.use_slot_bottleneck
    model_config.slot_recon_loss_weight = cfg.slot_recon_loss_weight


def append_metrics_record(metrics_path: Path, split: str, step: int, metrics: Dict[str, float]) -> None:
    """Append a single structured metrics record as JSONL for experiment automation."""
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            serializable_metrics[key] = float(value.item())
        else:
            serializable_metrics[key] = float(value)

    record = {"split": split, "step": int(step), **serializable_metrics}
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def set_seed_everywhere(seed: int) -> None:
    """Set random seed for reproducible ablation comparisons."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module. 加载断点续训的权重

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)



def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel. 把模块包装成 DDP 多卡训练模块

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module], # 要初始化的类（如 ProprioProjector）
    module_name: str, # 模块名（如 "proprio_projector"）
    cfg: FinetuneConfig, # 配置（如 FinetuneConfig）
    device_id: int,
    module_args: dict, # 初始化需要的参数（如 {"vision_dim": 1024, "llm_dim": 4096}）
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
# 通用模块初始化工具
# 把你给的参数 → 塞进模块类里 → 造出一个可用的神经网络模块，顺便帮你加载权重、放 GPU、包装多卡训练
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args) # 根据传入的类 + 参数，创建一个模块
    count_parameters(module, module_name) # 打印一下模块的参数量

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step) # 加载断点续训的权重
        module.load_state_dict(state_dict) # 把权重加载到模块里
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params) # 把模块包装成 DDP 多卡训练模块



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches, # 图像的patch数
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    num_visual_tokens = num_patches
    if getattr(output, "projector_features", None) is not None:
        num_visual_tokens = output.projector_features.shape[1] # 动态统计投影后的 slot token 数
    prediction_start = num_visual_tokens
    visual_token_start = 1
    visual_token_end = visual_token_start + num_visual_tokens

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids) # 序列里哪些位置属于“当前 step 的动作”（curr action）相关 token
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids) # 哪些位置属于“后续动作 chunk”（next actions）相关 token
    # 这俩 mask 直接决定从 text_hidden_states 取哪些 token 的隐藏状态
    
    #——————————————————————————————————————————————————————————————————————————————————
    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression):
        loss = output.loss
        predicted_token_ids = output.logits[:, prediction_start:-1].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

        if getattr(output, "slot_recon_loss", None) is not None:
            slot_recon_weight = cfg.slot_recon_loss_weight if cfg is not None else 1.0
            metrics["slot_recon_loss"] = output.slot_recon_loss.item()
            metrics["slot_recon_loss_weighted"] = (slot_recon_weight * output.slot_recon_loss).item()
        
    #——————————————————————————————————————————————————————————————————————————————————
    # Compute metrics for continuous action representations (L1 regression)
    else:
        # Get last layer hidden states
        multi_layer_hidden_states = []
        
        for item in output.hidden_states[0:]:  # 把大模型的每一层的特征都拿出来用
            # last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)

            text_hidden_states = item[:, prediction_start:-1]  # 保持与 labels[:, 1:] 的因果对齐
            # Get hidden states for action portion of response
            batch_size = batch["input_ids"].shape[0]
            # actions_hidden_states = text_hidden_states[:, -1, :].reshape(batch_size, 1, -1).to(torch.bfloat16)

            # 从文本里只挑出代表 “动作” 的那些位置（当前动作 + 未来动作 都挑出来）
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1,NUM_TOKENS, -1).to(torch.bfloat16)
            task_latten_states = item[:, visual_token_start:visual_token_end].reshape(batch_size, 1, num_visual_tokens , -1) # 把图片 / slot 特征单独抽出来

            # 图片特征（看到了什么） + 动作特征（要做什么）变成一组完整的控制特征，拼接
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)

            # 把大模型每一层的特征都拼在一起
            multi_layer_hidden_states.append(all_hidden_states)
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)

        predicted_actions = action_head.module.predict_action(
            multi_layer_hidden_states,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase,
            )

        action_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
        loss = action_loss
        slot_recon_weight = cfg.slot_recon_loss_weight if cfg is not None else 1.0
        if getattr(output, "slot_recon_loss", None) is not None:
            # total_loss = action_loss + α * slot_recon_loss
            loss = loss + slot_recon_weight * output.slot_recon_loss

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "action_l1_loss": action_loss.item(),
            }
        )
        if getattr(output, "slot_recon_loss", None) is not None:
            metrics["slot_recon_loss"] = output.slot_recon_loss.item()  # 原始（未加权）
            metrics["slot_recon_loss_weighted"] = (
                slot_recon_weight * output.slot_recon_loss
            ).item()

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]  #  第0列：当前动作
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]  #  第1列及以后：未来动作
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())
                # print('next: ',next_actions_l1_loss.item())

            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics
    #——————————————————————————————————————————————————————————————————————————————————



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


# 把整个机器人模型（视觉 + 语言 + 动作控制）完整保存下来，包括所有小模块
def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
    
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz or not cfg.use_lora:
            vla.module.save_pretrained(checkpoint_dir) # directly save checkpoint without lora
        else:
            vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)  # Create a new model with configuration, the parameters are randomly initialized
            # print(new_state_dict['action_queries.weight'])
            new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
            missing_keys, unexpected_keys = base_vla.load_state_dict(new_state_dict, strict=False)
            
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
        )


        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        
        # Wait for merged model to be saved
        dist.barrier()



def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)

    return avg_val_metrics



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """ 

    global RAW_STATE_DICT
    
    #  不能同时用 L1 回归和扩散模型，二选一
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    metrics_jsonl_path = run_dir / "metrics.jsonl"

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    set_seed_everywhere(cfg.seed + device_id)
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # 只有主进程记录训练日志
    if distributed_state.is_main_process:
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="offline")
        if metrics_jsonl_path.exists() and not cfg.resume:
            metrics_jsonl_path.unlink()

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n" # 一次性预测多少步动作
        f"\tACTION_DIM: {ACTION_DIM}\n" #  7 维动作，动作维度（机器人有多少个控制量）
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n" # 本体感知维度（机器人自身状态长度）
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}" # 动作 & 状态用哪种归一化方式
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    if model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig) # 以后看到名字叫 openvla 的模型，配置文件就用 OpenVLAConfig 这个
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor) # 如果配置是 OpenVLAConfig，处理图片的模块，用 PrismaticImageProcessor
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor) # 如果配置是 OpenVLAConfig，整体的处理器（文本 + 图像一起处理）用 PrismaticProcessor
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction) # 如果配置是 OpenVLAConfig，主模型就用 OpenVLAForActionPrediction！


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path) # 根据配置文件，自动更新模型的映射关系
        # 打开模型文件夹里的 config.json，把自动加载模型的映射类，强行改成 OpenVLA 自己的类
        check_model_logic_mismatch(cfg.config_file_path) # 检查模型配置是否有逻辑错误 / 冲突

    # Wait for model files to be synced
    dist.barrier() # 其他进程必须等主进程完成文件操作，才能开始加载模型

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor) 
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True) # 从保存好的文件里加载完整配置给AutoProcessor

#——————————————————————————————————————————————————————————————————
    if cfg.use_minivlm:
        hf_token = '' # 因为是加载本地模型，不需要联网权限
        # 判断模型是不是特殊版 Prism，是的话用 load ()，否则用 load_vla ()，最终都得到一个旧模型 vlm
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json") # config.json 里写了 model_type: "openvla"
        apply_runtime_slot_overrides(config, cfg)
        # 系统就知道：哦！要用注册过的 OpenVLAConfig 来解析！
        # register 是 “登记类型”，from_pretrained 是 “加载具体参数”
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized

        # 旧模型的层名 → 新模型的层名
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            # 改名后的新权重字典
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
        # missing_keys：新模型有，但权重里没有的层
        # unexpected_keys：权重里有，但新模型没有的层
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict
#——————————————————————————————————————————————————————————————————
    else:
        RAW_STATE_DICT ={}
        config = AutoConfig.from_pretrained(cfg.config_file_path)
        apply_runtime_slot_overrides(config, cfg)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input) # 一次输入几张图片
#——————————————————————————————————————————————————————————————————
    # vla.set_version(cfg.version)

    if cfg.use_lora:
        # 设置 LoRA 参数
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha= 2 * cfg.lora_rank,  #  缩放系数
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",  #  把 LoRA 加在所有线性层
            init_lora_weights="gaussian",  #  初始化 LoRA 权重为高斯分布
        )
        # 给模型装上 LoRA 适配器
        vla = get_peft_model(vla, lora_config)
        # action_queries / slot_adapter / slot_projector 都是新加的随机初始化模块，
        # PEFT 会把所有非 LoRA 参数 freeze，需要在这里显式打开它们的 requires_grad，
        # 否则 slot_projector 拿不到下游 loss 的梯度，slot_adapter 也只能靠 recon loss。
        TRAINABLE_KEYWORDS = ("action_queries", "slot_adapter", "slot_projector")
        for name, param in vla.named_parameters():
            if any(k in name for k in TRAINABLE_KEYWORDS):
                param.requires_grad = True
        vla.print_trainable_parameters() # 打印：模型现在有哪些参数可训练

    else: # 不加 LoRA，只训练 action_queries / slot 相关模块
        TRAINABLE_KEYWORDS = ("action_queries", "slot_adapter", "slot_projector")
        for name, param in vla.named_parameters():
            if any(k in name for k in TRAINABLE_KEYWORDS):
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        # 统计一下套 FiLM 之前，视觉主干有多少参数
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        # 给视觉主干套上 FiLM 自适应封装
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone, # 原来的视觉主干
            llm_dim=vla.llm_dim, # 语言模型维度
        )
        # 打印封装后的参数量
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        # 如果是断点续训，加载 FiLM 权重
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP 多卡
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        # 把低维的状态数据，投影到语言模型能理解的维度
        proprio_projector = init_module(
            ProprioProjector, #类
            "proprio_projector",
            cfg, # 配置
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM}, # 初始化需要的参数，这个模型类的参数直接设置成这个
            to_bf16=True,
        )

    # If applicable, instantiate continuous action head for L1 regression
    # 初始化一个动作头
    if cfg.use_l1_regression:
        action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim, 
            "hidden_dim": vla.module.llm_dim, 
            "action_dim": ACTION_DIM,
            "use_pro_version": cfg.use_pro_version,
            },
        to_bf16=True,
        )

    # 一张图会被切成多少个小补丁（patch），乘以一次输入几张图，得到模型一次看到的视觉特征总数量
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # 看一下：If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    
    
    # Instantiate optimizer
    # 只把 requires_grad = True 的参数拿出来
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression: # 如果用 L1 回归，把动作头的参数也拿出来
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio: # 如果用本体感知，把本体感知投影器的参数也拿出来
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    # 打印一下：总共有多少个可训练参数
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")

#——————————————————————————————————————————————————————————————————-
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate) # 创建一个优化器，用 AdamW 算法，学习率是 cfg.learning_rate

    # Record original learning rate
    # 把最开始的学习率存起来
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    # 训练到一定步数后，自动把学习率变小
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],   # 多少步后开始降LR
        gamma=0.1,  # 衰减倍数：×0.1
    )
    # 2. CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=cfg.num_steps_before_decay, 
    #         eta_min=0.0001,          
    #         )

    # Create Action Tokenizer
    # 把机器人动作（坐标、角度）转成模型能理解的 token，训练时用它来把动作变成标签
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # 加载微调数据集
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    # 如果输入大于 1 张图 → 说明除了主相机，还有腕部相机（手腕摄像头）
    use_wrist_image = cfg.num_images_in_input > 1 # 判断是否使用腕部相机图像

    # Create training and optional validation datasets
    # 创建数据处理流水线 
    batch_transform = RLDSBatchTransform(
        action_tokenizer, #  动作编码器
        processor.tokenizer, # 文本分词器
        image_transform=processor.image_processor.apply_transform, # 图像预处理
        prompt_builder_fn=PurePromptBuilder, # 指令模板
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm
        )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,  #上面
        resize_resolution=tuple(vla.module.config.image_sizes), # 图像大小
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False, # 验证集不需要 shuffle
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    # 保存数据集统计信息，以便在推理时反归一化动作，-1~1 → 还原成真实动作值
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir) # 保存动作归一化参数

    # Create collator and dataloader
    # 创建一个数据整理器，把不同长度的样本补齐成相同长度，方便批量训练
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right" # 填充方向：右边
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print('Len of dataloader: ', len(dataloader))
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        # deque 就是一个固定长度的队列
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps), #总损失
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps), # 当前动作预测精度
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps), # 当前动作预测 L1 损失
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps), # 下一动作预测精度
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps), # 下一动作预测 L1 损失
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = (cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0) or (cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0)
            # 喂数据给模型 → 模型预测动作 → 算误差
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
            )

            # Normalize loss to account for gradient accumulation
            # 多批数据累积一次更新，所以损失要除以累积步数
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            # metrics 是当前这一步算出来的结果，比如：
            # loss: 0.85
            # 动作精度: 0.72
            # 动作 L1 损失: 0.33
            # 下一动作预测精度: 0.68
            # 下一动作预测 L1 损失: 0.42
            for metric_name, value in metrics.items():
                if metric_name not in recent_metrics:
                    recent_metrics[metric_name] = deque(maxlen=cfg.grad_accumulation_steps)
                recent_metrics[metric_name].append(value)

            # Compute gradient step index
            # 计算当前是 “第几次更新参数”
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics) # 把最近 N 步的指标拿出来，算平均值！

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                append_metrics_record(metrics_jsonl_path, "train", log_step, smoothened_metrics)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups: # 把计算好的学习率，实时更新给优化器
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            # 如果累积了 N 批数据，就更新一次参数
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step() # 更新参数
                scheduler.step() # 更新学习率
                optimizer.zero_grad() # 清空梯度
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                avg_val_metrics = run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                if distributed_state.is_main_process:
                    append_metrics_record(metrics_jsonl_path, "val", log_step, avg_val_metrics)
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
