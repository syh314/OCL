"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
    NUM_TOKENS
)
from .configuration_prismatic import OpenVLAConfig, PrismaticConfig



# Set up logger
logger = logging.getLogger(__name__)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper



# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor



def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class MLP(nn.Sequential):

    def __init__(self, in_dim: int, dims: List[int], ln: Optional[str] = None, dropout: float = 0.0) -> None:
        assert ln in [None, "pre", "post"]
        # dims: 每一层的输出维度列表
        # ln: 要不要加 LayerNorm → pre（前面）/post（后面）/None

        ci = in_dim
        layers: List[nn.Module] = []

        if ln == "pre": # 要不要加 LayerNorm → pre（前面）/post（后面）/None
            layers.append(nn.LayerNorm(ci))

        for i, c in enumerate(dims): # 遍历每一层目标维度
            if i + 1 < len(dims): # 不是最后一层：加 Linear + GELU + Dropout
                block: List[Optional[nn.Module]] = [
                    nn.Linear(ci, c),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout else None,
                ]
            else: # 是最后一层：只加 Linear，不加激活、不加 Dropout
                block = [nn.Linear(ci, c)]
            layers.extend(layer for layer in block if layer is not None)
            ci = c

        if ln == "post":
            layers.append(nn.LayerNorm(ci))

        super().__init__(*layers)



# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    """
    Vision backbone for Prismatic models that handles image feature extraction.

    Supports both single backbone (e.g., SigLIP) and fused backbone (e.g., SigLIP + DINOv2) configurations.
    For fused backbones, features from both models are concatenated along the feature dimension.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        Initialize the vision backbone.

        Args:
            use_fused_vision_backbone: Whether to use two backbones and fuse their features
            image_sizes: List of image sizes for each backbone
            timm_model_ids: List of TIMM model IDs to use for each backbone
            timm_override_act_layers: List of activation layer overrides for each backbone
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # Default value, can be overridden later

        # 检查最多支持 2 个 backbone
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # Create primary featurizer
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create secondary featurizer if using fused backbone
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules for HF compatibility
        self._patch_layer_scales()


    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        Create a TIMM-based featurizer model with appropriate configurations.

        Args:
            model_id: The TIMM model ID to load
            img_size: Input image size for the model
            act_layer: Override for the activation layer type

        Returns:
            A configured featurizer model
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch the forward function to extract the second-to-last layer features
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer


    def _patch_layer_scales(self) -> None:
        """
        Patch all LayerScale modules to be compatible with HF's parameter naming.

        HF Transformers overwrites parameters with names containing 'gamma',
        so we need to rename and modify the forward method.
        """
        # Patch primary featurizer
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # Patch secondary featurizer if it exists
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)


    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches


    def get_num_images_in_input(self) -> int:
        """
        Returns the number of input images for the vision backbone.

        Returns:
            Number of images expected in the input
        """
        return self.num_images_in_input


    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        Sets the number of input images for the vision backbone.

        Args:
            num_images_in_input: Number of images to expect in the input
        """
        self.num_images_in_input = num_images_in_input


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone.

        If `self.use_fused_vision_backbone == True`, uses both SigLIP and DINOv2 transformers to extract visual features
        (otherwise uses SigLIP only). Allows multi-image inputs (but only for fused vision backbone).

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along the patch dimension
            return torch.cat(all_patches, dim=1)



# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features

# ————————————————————————————————————————— Slot Attention —————————————————————————————————————————
class SlotAttention(nn.Module):

    def __init__(
        self,
        num_iter: int,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        kv_dim: Optional[int] = None,
        trunc_bp: Optional[str] = "bi-level", # 梯度截断
    ) -> None:
        super().__init__()
        kv_dim = kv_dim or embed_dim
        assert trunc_bp in ["bi-level", None]

        self.num_iter = num_iter
        self.trunc_bp = trunc_bp
        self.norm1q = nn.LayerNorm(embed_dim) # 对槽位做归一化
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False) # 槽位 → Q
        self.norm1kv = nn.LayerNorm(kv_dim) # 对图像特征做归一化
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False) # 图像 → K
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False) # 图像 → V
        self.rnn = nn.GRUCell(embed_dim, embed_dim) # 槽位更新用的GRU
        self.norm2 = nn.LayerNorm(embed_dim) # 对槽位做归一化
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)

    def forward(
        self,
        input: torch.Tensor,
        query: torch.Tensor,
        smask: Optional[torch.Tensor] = None,
        num_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input: in shape (b,h*w,c)
        query: in shape (b,n,c) n 个待优化的槽向量
        smask: slots' mask, shape=(b,n), dtype=bool. True means there is a valid slot. # 槽位掩码（哪些槽有效）
        """
        b, n, _ = query.shape
        self_num_iter = num_iter or self.num_iter

        kv = self.norm1kv(input)
        k = self.proj_k(kv)
        v = self.proj_v(kv)
        q = query

        for i in range(self_num_iter):
            if i + 1 == self_num_iter and self.trunc_bp == "bi-level":
                q = q.detach() + query - query.detach()

            slots_prev = q # 保存当前槽位
            q = self.norm1q(q)
            q = self.proj_q(q)
            updates, attn = self.inverted_scaled_dot_product_attention(q, k, v, smask)
            slots = self.rnn(updates.flatten(0, 1), slots_prev.flatten(0, 1)).view(b, n, -1)
            q = slots + self.ffn(self.norm2(slots))

        return q, attn

    @staticmethod
    def inverted_scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        smask: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = q.size(-1) ** -0.5
        logit = torch.einsum("bqc,bkc->bqk", q * scale, k) # 计算注意力分数
        if smask is not None:
            logit = logit.where(smask[:, :, None], -torch.inf)

        attn_logits = logit.softmax(1) #  softmax (1) → 对 Q 做归一化
        attn = attn_logits / (attn_logits.sum(2, keepdim=True) + eps) # 让每个 slot 的权重总和为 1
        updates = torch.einsum("bqv,bvc->bqc", attn, v) # 计算加权求和后的槽位更新值
        return updates, attn_logits


class NormalSharedInitializer(nn.Module):
    """without preheating"""

    def __init__(
        self,
        num: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.num = num
        self.emb_dim = emb_dim

        self.mean = nn.Parameter(torch.empty(1, 1, emb_dim, dtype=torch.float))
        self.logstd = nn.Parameter(torch.empty(1, 1, emb_dim, dtype=torch.float))
        nn.init.xavier_uniform_(self.mean)
        nn.init.xavier_uniform_(self.logstd)

    def forward(
        self, encode: torch.Tensor, condit: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del condit
        b = encode.size(0) # batch size
        mean = self.mean.expand(b, self.num, -1) # 复制 batch 份
        return mean + torch.randn_like(mean) * self.logstd.exp() # 添加正态噪声


class ARRandTransformerDecoder(nn.Module):
    """Transformer 解码器"""

    def __init__(
        self,
        vfm_dim: int,  # 视觉稠密 token 特征维度（decoder 内与 patch token 对齐，避免与 slot_dim 命名混淆）
        slot_dim: int,  # Slot 特征维度
        num_tokens: int, # 要重建多少个稠密token
        nhead: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.vfm_dim = vfm_dim
        self.num_tokens = num_tokens

        self.mask_token = nn.Parameter(torch.randn(1, 1, vfm_dim) * vfm_dim**-0.5) # 遮一些token，让模型根据 Slot 去猜被遮住的内容
        self.position_embed = nn.Parameter(torch.randn(1, num_tokens, vfm_dim) * vfm_dim**-0.5) # 位置编码，每张图都共享一个位置编码
        self.project1 = nn.Sequential(
            nn.Linear(vfm_dim, vfm_dim, bias=False),
            nn.LayerNorm(vfm_dim),
        )
        self.project2 = nn.Sequential(
            nn.Linear(slot_dim, vfm_dim, bias=False), # 把 Slot 维度 → 投影到视觉稠密特征维度
            nn.LayerNorm(vfm_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vfm_dim,
            nhead=nhead,
            dim_feedforward=vfm_dim * 4, # MLP 中间放大 4 倍
            dropout=0.0,
            activation="gelu",
            batch_first=True, # 输入形状是 (batch, seq_len, dim) 
            norm_first=True, # Pre-LayerNorm 模式（先归一化，再注意力）
            bias=False,
        )
        self.backbone = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        #  归一化的位置从内部 → 搬到了外部
        self.norm0 = self.backbone.layers[0].norm1 # 把第一层的第一个 norm 拿出来单独保存
        self.backbone.layers[0].norm1 = nn.Identity()
        self.readout = nn.Identity()  # 什么都不做，输入啥输出啥
        self._attent = None #缓存

        def attent_hook_forward_pre(module, args, kwargs):
            kwargs["need_weights"] = True #  强制要求“返回注意力权重”

        def attent_hook_forward(module, args, output):
            self._attent = output[1] # 缓存注意力权重
        
        # 注册钩子，在注意力计算前/后 → 自动记录注意力权重
        self.backbone.layers[-1].multihead_attn.register_forward_pre_hook(
            attent_hook_forward_pre, with_kwargs=True
        )
        self.backbone.layers[-1].multihead_attn.register_forward_hook(attent_hook_forward)

    def forward(
        # input: target to be destructed, shape=(b,m=h*w,c)
        # slots: slots, shape=(b,n,c)
        # smask: slots' mask, shape=(b,n), dtype=bool. True means there is a valid slot.
        self, input: torch.Tensor, slots: torch.Tensor, smask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, m, c = input.shape
        if m != self.num_tokens: # 输入的 token 数量 ≠ 期望重建的 token 数量
            raise ValueError(f"Slot decoder expects {self.num_tokens} tokens, got {m}.")

        tokens = self.project1(input) # 把输入图像特征 → 投影到与 Slot 维度相同的特征空间
        pos_embed = self.position_embed[:, :m, :] # 位置编码，每张图都共享一个位置编码

        if self.training:
            # 打乱 token 位置
            # 给 batch 里每张图都生成一套随机打乱索引，idxs 形状：(b, m)
            idxs = torch.stack(
                [torch.randperm(m, device=input.device) for _ in range(b)], dim=0
            )
            idxs_expanded = idxs[:, :, None].expand(-1, -1, c) # 把打乱索引扩展到特征维度
            shuffled = tokens.gather(1, idxs_expanded) # 用打乱索引 → 重排 token 位置 
            shuffled_pos = pos_embed.expand(b, -1, -1).gather(1, idxs_expanded) # 位置编码也跟着一起打乱
 
            # 随机保留一部分 token，让模型根据 Slot 去猜被遮住的内容
            # 随机生成一个数：随机保留前 N 个 token
            keep_rand = torch.randint(0, m - 1, (b, 1), device=input.device) 
            # 固定值：保留前 10% 的 token
            keep_fixed = torch.full(
                (b, 1),
                fill_value=max(int(m * 0.1) - 1, 1),
                dtype=torch.long,
                device=input.device,
            )
            # 两种策略：50% 概率随机保留，50% 概率固定保留 10%
            keep = torch.where(torch.rand(b, 1, device=input.device) < 0.5, keep_rand, keep_fixed)
            keep_mask = torch.arange(m, device=input.device)[None, :] < keep # 生成一个掩码，前 keep 个位置 = True（保留）
            query = torch.where(
                # 剩下所有 token 全部替换成可学习的 mask_token
                keep_mask[:, :, None],
                shuffled,
                self.mask_token.expand(b, m, -1),
            )
            query = query + shuffled_pos
        else:
            idxs = None
            query = tokens + pos_embed

        memory = self.project2(slots) # 让 Slots 维度和图像 token 对齐
        decode = self.backbone(
            self.norm0(query),
            memory=memory,
            memory_key_padding_mask=None if smask is None else ~smask, # 让模型忽略无效的 Slot
        )
        recon = self.readout(decode)

        attent = self._attent # 最后一层交叉注意力权重
        if self.training and idxs is not None:
            idxs_inverse = idxs.argsort(1)[:, :, None] # 求逆打乱索引（把乱序 → 变回原来顺序）
            recon = recon.gather(1, idxs_inverse.expand(-1, -1, recon.size(-1)))
            if attent is not None:
                attent = attent.gather(1, idxs_inverse.expand(-1, -1, attent.size(-1)))

        if attent is not None:
            attent = attent.permute(0, 2, 1)

        return recon, attent


class SmoothSASlotAdapter(nn.Module):

    def __init__(
        self,
        encode_project: nn.Module, # 把图像特征 → 投影到与 Slot 维度相同的特征空间
        initializ: nn.Module, # 初始化槽位
        aggregat: nn.Module,  # 聚合槽位
        decode: nn.Module,    # 解码器
    ) -> None:
        super().__init__()
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        # 给 project、aggregat、decode 做统一的参数初始化
        self.reset_parameters([self.encode_project, self.aggregat, self.decode])

    @staticmethod
    # 把所有层的偏置（bias）全部设为 0
    def reset_parameters(modules: List[Optional[nn.Module]]) -> None:
        for module in modules:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias) # 把 bias 初始化为 0
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh) # 把 bias_hh 初始化为 0

    def encode_slots(
        self, feature: torch.Tensor, condit: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encode = self.encode_project(feature) # 把图像特征 → 投影到与 Slot 维度相同的特征空间
        query = self.initializ(encode, condit) # 初始化槽位
        slotz, attenta = self.aggregat(encode, query) # 聚合槽位
        return encode, slotz, attenta

    def forward(
        self,
        feature: torch.Tensor,
        condit: Optional[torch.Tensor] = None,
        need_decode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        encode, slotz, attenta = self.encode_slots(feature, condit)

        recon, attentd = None, None
        if need_decode:
            recon, attentd = self.decode(encode, slotz)

        return encode, slotz, attenta, recon, attentd

# ————————————————————————————————————————— Main HF Class Definitions ————————————————————————————————————————————————————————

# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None
    slot_features: Optional[torch.FloatTensor] = None
    slot_attention: Optional[torch.FloatTensor] = None
    slot_reconstruction: Optional[torch.FloatTensor] = None
    slot_decoder_attention: Optional[torch.FloatTensor] = None
    slot_recon_loss: Optional[torch.FloatTensor] = None



class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa



class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )
        
        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size
        self.use_slot_bottleneck = config.use_slot_bottleneck
        self.num_slots = config.num_slots
        self.slot_dim = config.slot_dim

        if self.use_slot_bottleneck:
            self.slot_adapter = SmoothSASlotAdapter(
                encode_project=MLP(
                    self.vision_backbone.embed_dim,
                    [self.slot_dim, self.slot_dim],
                    "pre",
                    0.0,
                ),
                initializ=NormalSharedInitializer(
                    num=self.num_slots,
                    emb_dim=self.slot_dim,
                ),
                aggregat=SlotAttention(
                    num_iter=config.slot_num_iter,
                    embed_dim=self.slot_dim,
                    ffn_dim=config.slot_ffn_dim,
                    dropout=0.0,
                    kv_dim=self.slot_dim,
                    trunc_bp="bi-level",
                ),
                decode=ARRandTransformerDecoder(
                    vfm_dim=self.slot_dim,
                    slot_dim=self.slot_dim,
                    num_tokens=self.vision_backbone.get_num_patches(), # 一张图像的patch数
                    nhead=config.slot_num_heads,
                    num_layers=config.slot_num_decode_layers,
                ),
            )
            self.slot_projector = PrismaticProjector(
                False, vision_dim=self.slot_dim, llm_dim=config.text_config.hidden_size
            )
        else:
            self.slot_adapter = None
            self.slot_projector = None

        #Action query token
        self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim) # 创建 N 个动作专用向量
        self.action_queries.weight.data.zero_() # 一开始全部填 0

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    # 获取输入嵌入层 把文字 token → 向量
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()
    def set_version(self, version: str):
        self.version = version
        return self.version

    def get_num_visual_tokens(self) -> int:
        if self.use_slot_bottleneck:
            return self.num_slots * self.vision_backbone.get_num_images_in_input()
        return self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()

    # 设置输入嵌入层 把文字 token → 向量
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    # decoder = 大模型本体
    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)
    
    # 让输入嵌入层 和 输出层 用同一套参数
    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        # 给词表扩容！
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        # noisy_action_features对应action_queries
        # 把输入序列里，标记为 “动作位置” 的 embedding，全部替换成 action_queries
        """
        Replace embeddings in input_embeddings at positions where all_actions_mask is True
        with embeddings from noisy_action_features, using vectorized operations.

        Args:
            input_embeddings: Tensor of shape (B, S, D)
            all_actions_mask: Boolean tensor of shape (B, S)
            noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

        Returns:
            Modified input_embeddings tensor
        """
        # Clone input to avoid modifying the original tensor
        new_input_embeddings = input_embeddings.clone()

        # Create a tensor with the same shape of input_embeddings to hold the noisy action features
        # 创建一个全 0 空 input_embeddings，用来放动作向量
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # Create batch indices for splicing
        # 生成批次索引（方便批量赋值）
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # Get indices where mask is True for each sample
        # 找到所有 mask = True 的位置
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # Move the noisy action features into their correct positions
        # print(noisy_action_features.size())
        
        # 把动作向量填到对应位置
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # Combine original input embeddings and noisy action embeddings using the mask
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        # 得到动作位置的掩码
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(
        self,
        pixel_values,
        language_embeddings=None,
        use_film: bool = False,
        need_slot_decode: bool = True,
    ):
        """Process dense vision tokens and optionally compress them with a SmoothSA-style slot bottleneck."""
        if use_film:
            patch_features = self.vision_backbone(pixel_values, language_embeddings)
        else:
            patch_features = self.vision_backbone(pixel_values)

        slot_outputs = dict(
            slot_features=None,
            slot_attention=None,
            slot_reconstruction=None,
            slot_decoder_attention=None,
            slot_recon_loss=None,
        )

        if not self.use_slot_bottleneck:
            return self.projector(patch_features), slot_outputs

        batch_size, total_tokens, feature_dim = patch_features.shape
        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches() # 一张图像的patch数
        expected_tokens = num_images * num_patches
        if total_tokens != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} vision tokens ({num_images} images x {num_patches} patches), got {total_tokens}."
            )

        patch_features = patch_features.reshape(batch_size * num_images, num_patches, feature_dim) # 对一张图像做 Slot Attention
        encode, slotz, attenta, recon, attentd = self.slot_adapter(
            patch_features,
            need_decode=need_slot_decode,
        )
        # todo
        projected_slots = self.slot_projector(slotz).reshape(batch_size, num_images * self.num_slots, -1) # 把slot维度投影到与 LLM 维度特征空间

        slot_recon_loss = None
        if recon is not None:
            slot_recon_loss = F.mse_loss(recon, encode.detach()) # encode原图

        slot_outputs.update(
            slot_features=slotz.reshape(batch_size, num_images * self.num_slots, -1),
            slot_attention=attenta.reshape(batch_size, num_images, self.num_slots, num_patches),
            slot_reconstruction=None if recon is None else recon.reshape(batch_size, num_images, num_patches, -1),
            slot_decoder_attention=None if attentd is None else attentd.reshape(batch_size, num_images, self.num_slots, num_patches),
            slot_recon_loss=slot_recon_loss,
        )
        return projected_slots, slot_outputs

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            # 把机器人状态 → 变成大模型能看懂的向量
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            # 增加一个维度，变成 “一个 token”
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            # 拼到图像特征末尾
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """Build multimodal embeddings and attention mask"""
        # [<BOS>] + [图块1,图块2,图块3,图块4,图块5,图块6] + [指令1,指令2,动作1,动作2]
        # 图片的每一个块都是有效内容，没有空白。所以 mask 全是 True。
        # Update attention mask
        
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None
        slot_outputs = dict(
            slot_features=None,
            slot_attention=None,
            slot_reconstruction=None,
            slot_decoder_attention=None,
            slot_recon_loss=None,
        )

        # 自回归生成，只输入 1 个 token
        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # 仅输入语言特征
        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids is not None and input_ids.shape[0] == pixel_values.shape[0]) or (
            inputs_embeds is not None and inputs_embeds.shape[0] == pixel_values.shape[0]
        ):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            # Get input embeddings (from language model embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            
            # Extract action masks
            all_actions_mask = self._process_action_masks(labels)

            # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
            
            # print(input_embeddings[~all_actions_mask].size())
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )  # (B, lang_seq_len, llm_dim)

            # 投影后的 slot tokens
            projected_patch_embeddings, slot_outputs = self._process_vision_features(
                pixel_values,
                language_embeddings,
                use_film,
                need_slot_decode=self.training or output_projector_features,
            )

            # Process action embeddings
            # 不管 noisy_actions 是不是 None，都执行完全相同的操作
            if noisy_actions is not None: # 去噪用
                

                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)
                

            else:
                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)

            # Build multimodal embeddings & attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
            
            # Build labels for multimodal sequence if needed
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

            # Dispatch to language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                ) 

        # === Otherwise =>> Assume Invalid! ===
        elif (
            (input_ids is not None and input_ids.shape[0] != pixel_values.shape[0])
            or (inputs_embeds is not None and inputs_embeds.shape[0] != pixel_values.shape[0])
        ):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        total_loss = language_model_output.loss
        if slot_outputs["slot_recon_loss"] is not None:
            # 用 config.slot_recon_loss_weight 把重建损失缩放到与下游 loss 可比的尺度
            slot_recon_weight = getattr(self.config, "slot_recon_loss_weight", 1.0)
            weighted_recon = slot_recon_weight * slot_outputs["slot_recon_loss"]
            total_loss = weighted_recon if total_loss is None else total_loss + weighted_recon

        return PrismaticCausalLMOutputWithPast(
            loss=total_loss,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
            slot_features=slot_outputs["slot_features"],
            slot_attention=slot_outputs["slot_attention"],
            slot_reconstruction=slot_outputs["slot_reconstruction"],
            slot_decoder_attention=slot_outputs["slot_decoder_attention"],
            slot_recon_loss=slot_outputs["slot_recon_loss"],
        )


    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        # 自回归，开启缓存加速，每次只传最后 1 个 token，让生成动作更快
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)



class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    #  在推理时，给输入手动添加一段「占位符 token」，告诉大模型，等会儿要在这里输出机器人动作！
    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        # NUM_TOKENS = 动作需要多少个 token，生成一串全是 1 的占位符
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], NUM_TOKENS)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # 加一个 “结束标记”
        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        # STOP_INDEX = 1024  # 只是举例，数字编号而已，模型看到 1024 就知道：任务完成，停止输出
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        # 输入变长了，掩码也要跟着变长
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    #  在推理时，给文本标签手动同样添加一段「占位符 token」
    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        # 把模型输出的 [-1, 1] 动作 → 还原成机器人能执行的真实动作
        action_norm_stats = self.get_action_stats(unnorm_key)
        """
        # action_norm_stats = {
         "min": [10, 20, -50],    # 3个维度的真实最小值
         "max": [90, 60, 50],     # 3个维度的真实最大值
         "mask": [True, True, False]  # 哪些维度要反归一化
        }
        """
        # normalized_actions = np.array([-0.5, 0.5, -1.0])
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            # 用普通最大最小值归一化:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            # 用百分位数归一化:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        # actions = [30, 50, -1.0]

        return actions


    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PROMPT_TOKENS,
        action_head=None,
        proprio=None,
        proprio_projector=None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""

        action_queries = self.action_queries.weight  # (动作数, 维度数) # 拿出动作查询向量的权重
        # reshape  (b, chunk_size, h) (b, 动作数, 维度数)
        """
        [[0.1, 0.2, 0.3, 0.4],  # 第 1 个动作查询
         [0.5, 0.6, 0.7, 0.8]]  # 第 2 个动作查询
        """
        action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
        # Replace action token embeddings with noisy action embeddings
        input_embeddings = self._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, action_queries)
        """
        [指令1向量, 指令2向量, 【占位符1】, 【占位符2】]       形状：(1, 4, 4)
        mask [False, False, True, True]
        [指令1向量, 指令2向量, action_query0, action_query1]
        """
        # Build multimodal embeddings and attention mask
        # BOS + 图片 + 文字 + 动作 长序列
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        # 从大模型输出里，提取【图像特征】和【动作特征】，拼接成多隐层混合特征，给后续动作头输出动作！
        multi_layer_hidden_states = []

        # num_visual_tokens：图片或 slots 有多少个 token
        # NUM_PROMPT_TOKENS：文本指令有多少 token
        # NUM_TOKENS：动作占位符有多少个
        num_visual_tokens = projected_patch_embeddings.shape[1] # 动态统计真实视觉 token 数
        visual_token_start = 1
        visual_token_end = visual_token_start + num_visual_tokens
        prediction_start = num_visual_tokens
        action_start = prediction_start + NUM_PROMPT_TOKENS
        action_end = action_start + NUM_TOKENS

        for item in language_model_output.hidden_states[0:]:
            batch_size = item.shape[0]

            actions_hidden_states = item[:, action_start:action_end, :].reshape(
                batch_size, 1, NUM_TOKENS, -1
            ).to(torch.bfloat16)
            task_latten_states = item[:, visual_token_start:visual_token_end, :].reshape(
                batch_size, 1, num_visual_tokens, -1
            )
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states), 2)
            multi_layer_hidden_states.append(all_hidden_states)
        # 把所有层沿着层维度拼接
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)
        

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            # 输出 → 归一化动作 [-1, 1]
            normalized_actions = action_head.predict_action(multi_layer_hidden_states,
                                                proprio=proprio,
                                                proprio_projector=proprio_projector)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    action_start : action_start + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states


    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """

        pixel_values = kwargs["pixel_values"] # [1, 12, 224, 224]
        attention_mask = kwargs["attention_mask"] # 
    
        # Create fake labels tensor (needed for action mask)
        # 造假标签
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Prepare inputs by adding necessary tokens
        # 给输入补上【动作占位符 + 结束符】
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        # 把标签也补齐长度
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        # 把输入文本 → 变成向量
        input_embeddings = self.get_input_embeddings()(input_ids)
        # 最后用假标签生成动作位置掩码
        all_actions_mask = self._process_action_masks(labels) # 标出动作掩码

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        # 处理图像 → 变成图像特征，调到语言空间
        projected_patch_embeddings, _ = self._process_vision_features(
            pixel_values,
            language_embeddings,
            use_film,
            need_slot_decode=False,
        )

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)

        # Run regression or discrete token-based prediction
        # 运行回归或离散预测动作
        normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            NUM_PROMPT_TOKENS, 
            action_head=action_head,
            proprio=proprio, # [8]
            proprio_projector=proprio_projector,
            )
           
        # Unnormalize predicted actions
        # 归一化动作 → 原始动作
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states



    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
        # 返回 norm_stats[数据集名]["action"]
        """返回
        {
    "min": [0, 0, -50],
    "max": [100, 100, 50],
    "mask": [True, True, False]
}"""
