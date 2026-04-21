"""
qwen2_5.py

Class definition for all LLMs derived from QwenForCausalLM.

Local weights (avoid editing this file after every `git pull`):
  - export PRISMATIC_LLM_LOCAL_PATH=/abs/path/to/Qwen2.5-0.5B   # dir must contain config.json (+ weights)
  - or place an HF snapshot at ./pretrained_models/Qwen2.5-0.5B or <repo>/pretrained_models/Qwen2.5-0.5B
  - AutoDL 常见数据盘: /root/autodl-tmp/OCL/pretrained_models/Qwen2.5-0.5B（会自动探测）
  - or set hf_hub_path in QWEN25_MODELS to an absolute path (directory on disk)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Type

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
from prismatic.models.backbones.llm.prompting.qwen_prompter import QwenPromptBuilder


def _resolve_local_qwen_hf_path(hf_hub_path: str) -> str:
    """Map Hub id (e.g. Qwen/Qwen2.5-0.5B) to a local directory when present; else return hf_hub_path unchanged."""
    p = Path(hf_hub_path).expanduser()
    if p.is_dir() and (p / "config.json").exists():
        return str(p.resolve())

    env = os.environ.get("PRISMATIC_LLM_LOCAL_PATH", "").strip()
    if env:
        ep = Path(env).expanduser()
        if ep.is_dir() and (ep / "config.json").exists():
            return str(ep.resolve())

    hub_tail = hf_hub_path.rstrip("/").split("/")[-1]
    repo_root = Path(__file__).resolve().parents[4]
    for base in (Path.cwd() / "pretrained_models", repo_root / "pretrained_models"):
        for name in (hub_tail, hf_hub_path.replace("/", "--")):
            cand = base / name
            if cand.is_dir() and (cand / "config.json").exists():
                return str(cand.resolve())

    # AutoDL: weights often live at /root/autodl-tmp/OCL/pretrained_models/<model> regardless of cwd
    autodl_ocl = Path("/root/autodl-tmp/OCL/pretrained_models") / hub_tail
    if autodl_ocl.is_dir() and (autodl_ocl / "config.json").exists():
        return str(autodl_ocl.resolve())

    return hf_hub_path


# Registry =>> Support Qwen-2.5 Models (from HF Transformers)
# fmt: off
QWEN25_MODELS = {
    # === Pure Qwen2.5 (non-instruct/chat-tuned) Models ===
    "qwen25-0_5b-extra": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-0.5B"
    },
    "qwen25-0_5b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-0.5B"
    },
    "qwen25-1_5b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-1.5B"
    },
    "qwen25-3b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-3B"
    },
    "qwen25-7b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-7B"
    },

}
# fmt: on


class Qwen25LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        num_extra_tokens: int = 0,
    ) -> None:
        spec = dict(QWEN25_MODELS[llm_backbone_id])
        spec["hf_hub_path"] = _resolve_local_qwen_hf_path(spec["hf_hub_path"])
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **spec,
        )

        # add some more special tokens
        if num_extra_tokens > 0:
            added = self.tokenizer.add_tokens([f"<|extra_{i}|>" for i in range(num_extra_tokens)])
            assert added == num_extra_tokens, f"Added {added} of {num_extra_tokens} extra tokens to tokenizer!"
            print(f"Added {num_extra_tokens} extra tokens.")

        # there is already a special token for Qwen
        # self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return QwenPromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[torch.nn.Module]:
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[torch.nn.Module]:
        # TODO not sure that this works
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)