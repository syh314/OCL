"""
run_slot_ablation.py

Automates the small-scale OCL-vs-Baseline validation workflow on LIBERO-Spatial.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import draccus


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


@dataclass
class SlotAblationConfig:
    config_file_path: str = "openvla/openvla-7b"
    vlm_path: str = "openvla/openvla-7b"
    use_minivlm: bool = False
    use_pro_version: bool = True
    run_root_dir: Path = Path("runs/slot_ablation")
    # AutoDL 默认数据盘：/root/autodl-tmp/data（可通过命令行 --data_root_dir 覆盖）
    data_root_dir: Path = Path("/root/autodl-tmp/data")
    dataset_name: str = "libero_spatial_no_noops"
    num_images_in_input: int = 2
    use_proprio: bool = True
    image_aug: bool = True
    batch_size: int = 8
    grad_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    shuffle_buffer_size: int = 100_000
    seed: int = 7
    wandb_project: str = "slot-ablation"
    experiment_name: str = "libero_spatial_slot_ablation"
    phase: str = "all"  # one of: all, phase0, phase1, phase2
    skip_existing: bool = True
    save_rollout_videos: bool = False
    phase0_steps: int = 50
    phase1_steps: int = 2000
    phase2_steps: int = 10000
    phase1_eval_trials_per_task: int = 5
    phase2_eval_trials_per_task: int = 10
    phase1_eval_steps: str = "1000,2000"
    phase2_eval_steps: str = "2000,5000,10000"
    val_freq: int = 500
    save_freq: int = 1000
    wandb_log_freq: int = 1
    val_time_limit: int = 180


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _run(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_metrics(metrics_path: Path, split: str) -> List[Dict[str, float]]:
    records = []
    if not metrics_path.exists():
        return records

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("split") == split:
                records.append(record)
    return records


def _checkpoint_dir(run_dir: Path, step: int) -> Path:
    return Path(f"{run_dir}--{step}_chkpt")


def _latest_checkpoint(run_dir: Path, step: int) -> Path:
    checkpoint_dir = _checkpoint_dir(run_dir, step)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _build_finetune_cmd(cfg: SlotAblationConfig, run_id: str, use_slot_bottleneck: bool, max_steps: int, slot_alpha: float) -> List[str]:
    return [
        sys.executable,
        "vla-scripts/finetune.py",
        "--config_file_path", cfg.config_file_path,
        "--vlm_path", cfg.vlm_path,
        "--use_minivlm", _bool_str(cfg.use_minivlm),
        "--use_pro_version", _bool_str(cfg.use_pro_version),
        "--data_root_dir", str(cfg.data_root_dir),
        "--dataset_name", cfg.dataset_name,
        "--run_root_dir", str(cfg.run_root_dir),
        "--run_id_override", run_id,
        "--batch_size", str(cfg.batch_size),
        "--grad_accumulation_steps", str(cfg.grad_accumulation_steps),
        "--learning_rate", str(cfg.learning_rate),
        "--shuffle_buffer_size", str(cfg.shuffle_buffer_size),
        "--num_images_in_input", str(cfg.num_images_in_input),
        "--use_l1_regression", "True",
        "--use_lora", "False",
        "--use_film", "False",
        "--use_proprio", _bool_str(cfg.use_proprio),
        "--image_aug", _bool_str(cfg.image_aug),
        "--use_slot_bottleneck", _bool_str(use_slot_bottleneck),
        "--slot_recon_loss_weight", str(slot_alpha),
        "--use_val_set", "True",
        "--val_freq", str(cfg.val_freq),
        "--val_time_limit", str(cfg.val_time_limit),
        "--save_freq", str(cfg.save_freq),
        "--wandb_log_freq", str(cfg.wandb_log_freq),
        "--max_steps", str(max_steps),
        "--wandb_project", cfg.wandb_project,
        "--seed", str(cfg.seed),
    ]


def _build_eval_cmd(cfg: SlotAblationConfig, checkpoint_dir: Path, run_id: str, num_trials_per_task: int) -> List[str]:
    summary_json_path = checkpoint_dir / "libero_eval_summary.json"
    return [
        sys.executable,
        "experiments/robot/libero/run_libero_eval.py",
        "--pretrained_checkpoint", str(checkpoint_dir),
        "--use_l1_regression", "True",
        "--use_minivlm", _bool_str(cfg.use_minivlm),
        "--use_pro_version", _bool_str(cfg.use_pro_version),
        "--use_film", "False",
        "--use_proprio", _bool_str(cfg.use_proprio),
        "--num_images_in_input", str(cfg.num_images_in_input),
        "--task_suite_name", "libero_spatial",
        "--num_trials_per_task", str(num_trials_per_task),
        "--run_id_note", run_id,
        "--save_rollout_videos", _bool_str(cfg.save_rollout_videos),
        "--summary_json_path", str(summary_json_path),
    ]


def _run_training_if_needed(cfg: SlotAblationConfig, repo_root: Path, run_id: str, use_slot_bottleneck: bool, max_steps: int, slot_alpha: float) -> Path:
    run_dir = cfg.run_root_dir / run_id
    metrics_path = run_dir / "metrics.jsonl"
    if cfg.skip_existing and metrics_path.exists():
        return run_dir

    cmd = _build_finetune_cmd(cfg, run_id, use_slot_bottleneck, max_steps, slot_alpha)
    _run(cmd, repo_root)
    return run_dir


def _run_eval_if_needed(cfg: SlotAblationConfig, repo_root: Path, checkpoint_dir: Path, run_id: str, num_trials_per_task: int) -> Dict:
    summary_path = checkpoint_dir / "libero_eval_summary.json"
    if not (cfg.skip_existing and summary_path.exists()):
        cmd = _build_eval_cmd(cfg, checkpoint_dir, run_id, num_trials_per_task)
        _run(cmd, repo_root)

    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_alpha_from_phase0(metrics_path: Path) -> Dict[str, float]:
    train_records = [record for record in _load_metrics(metrics_path, "train") if record.get("step", 0) <= 50]
    action_values = [record["action_l1_loss"] for record in train_records if "action_l1_loss" in record]
    recon_values = [record["slot_recon_loss"] for record in train_records if "slot_recon_loss" in record]
    if not action_values or not recon_values:
        raise ValueError(f"Missing action/slot metrics in {metrics_path}")

    action_median = median(action_values)
    recon_median = median(recon_values)
    if recon_median <= 0:
        raise ValueError(f"Invalid slot recon median {recon_median} in {metrics_path}")

    alpha_center = 0.2 * action_median / recon_median
    candidates = [0.5 * alpha_center, alpha_center, 2.0 * alpha_center]
    return {
        "action_l1_median": action_median,
        "slot_recon_median": recon_median,
        "alpha_center": alpha_center,
        "candidates": candidates,
    }


def _summarize_phase1_candidate(run_dir: Path, eval_summaries: Dict[int, Dict]) -> Dict:
    train_records = _load_metrics(run_dir / "metrics.jsonl", "train")
    initial_record = next((record for record in train_records if "action_l1_loss" in record and "slot_recon_loss" in record), None)
    first_1000 = [
        record for record in train_records
        if 500 <= record.get("step", 0) <= 1000 and "action_l1_loss" in record and "slot_recon_loss_weighted" in record
    ]
    final_1000 = max(
        (record for record in train_records if record.get("step", 0) <= 1000 and "action_l1_loss" in record and "slot_recon_loss" in record),
        key=lambda record: record["step"],
        default=None,
    )

    pass_checks = True
    reasons = []
    if initial_record is None or final_1000 is None:
        pass_checks = False
        reasons.append("missing_train_metrics")
    else:
        if final_1000["action_l1_loss"] >= initial_record["action_l1_loss"]:
            pass_checks = False
            reasons.append("action_l1_not_down")
        if final_1000["slot_recon_loss"] >= initial_record["slot_recon_loss"]:
            pass_checks = False
            reasons.append("slot_recon_not_down")

    weighted_ratio = None
    if first_1000:
        ratios = [record["slot_recon_loss_weighted"] / record["action_l1_loss"] for record in first_1000 if record["action_l1_loss"] > 0]
        if ratios:
            weighted_ratio = median(ratios)
            if weighted_ratio < 0.1 or weighted_ratio > 0.3:
                pass_checks = False
                reasons.append("weighted_ratio_out_of_band")
    else:
        pass_checks = False
        reasons.append("missing_ratio_window")

    return {
        "pass_checks": pass_checks,
        "reasons": reasons,
        "weighted_ratio_median_500_1000": weighted_ratio,
        "eval_success_rates": {
            str(step): summary["final_success_rate"] for step, summary in eval_summaries.items()
        },
    }


def _choose_best_alpha(phase1_results: Dict[str, Dict]) -> Dict:
    sortable = []
    for alpha_key, result in phase1_results.items():
        eval_2000 = result["summary"]["eval_success_rates"].get("2000", 0.0)
        eval_1000 = result["summary"]["eval_success_rates"].get("1000", 0.0)
        sortable.append(
            (
                1 if result["summary"]["pass_checks"] else 0,
                eval_2000,
                eval_1000,
                alpha_key,
            )
        )

    sortable.sort(reverse=True)
    best_alpha_key = sortable[0][-1]
    return {"best_alpha": float(best_alpha_key), "ranking": sortable}


@draccus.wrap()
def run_slot_ablation(cfg: SlotAblationConfig) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg.run_root_dir.mkdir(parents=True, exist_ok=True)

    phase1_eval_steps = _parse_csv_ints(cfg.phase1_eval_steps)
    phase2_eval_steps = _parse_csv_ints(cfg.phase2_eval_steps)

    report_dir = cfg.run_root_dir / cfg.experiment_name
    report_dir.mkdir(parents=True, exist_ok=True)

    phase0_ocl_run_id = f"{cfg.experiment_name}--phase0-ocl-alpha0"
    phase0_baseline_run_id = f"{cfg.experiment_name}--phase0-baseline"

    alpha_plan_path = report_dir / "alpha_plan.json"
    phase1_results_path = report_dir / "phase1_results.json"
    phase2_results_path = report_dir / "phase2_results.json"
    final_report_path = report_dir / "final_report.json"

    alpha_plan: Optional[Dict] = None

    if cfg.phase in {"all", "phase0", "phase1", "phase2"}:
        _run_training_if_needed(cfg, repo_root, phase0_baseline_run_id, use_slot_bottleneck=False, max_steps=cfg.phase0_steps, slot_alpha=0.0)
        phase0_ocl_run_dir = _run_training_if_needed(cfg, repo_root, phase0_ocl_run_id, use_slot_bottleneck=True, max_steps=cfg.phase0_steps, slot_alpha=0.0)
        alpha_plan = _compute_alpha_from_phase0(phase0_ocl_run_dir / "metrics.jsonl")
        _write_json(alpha_plan_path, alpha_plan)

    if cfg.phase in {"all", "phase1", "phase2"}:
        if alpha_plan is None:
            with alpha_plan_path.open("r", encoding="utf-8") as f:
                alpha_plan = json.load(f)

        phase1_results = {}
        for alpha in alpha_plan["candidates"]:
            alpha_key = f"{alpha:.12g}"
            run_id = f"{cfg.experiment_name}--phase1-ocl-alpha-{alpha_key}"
            run_dir = _run_training_if_needed(cfg, repo_root, run_id, use_slot_bottleneck=True, max_steps=cfg.phase1_steps, slot_alpha=alpha)
            eval_summaries = {}
            for step in phase1_eval_steps:
                eval_summaries[step] = _run_eval_if_needed(
                    cfg,
                    repo_root,
                    _latest_checkpoint(run_dir, step),
                    f"{run_id}--eval-{step}",
                    cfg.phase1_eval_trials_per_task,
                )

            phase1_results[alpha_key] = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "alpha": alpha,
                "summary": _summarize_phase1_candidate(run_dir, eval_summaries),
            }

        selection = _choose_best_alpha(phase1_results)
        payload = {"alpha_plan": alpha_plan, "phase1_results": phase1_results, "selection": selection}
        _write_json(phase1_results_path, payload)

    if cfg.phase in {"all", "phase2"}:
        with phase1_results_path.open("r", encoding="utf-8") as f:
            phase1_payload = json.load(f)
        best_alpha = float(phase1_payload["selection"]["best_alpha"])

        phase2_runs = [
            ("baseline", False, 0.0),
            ("ocl", True, best_alpha),
        ]
        phase2_payload = {"best_alpha": best_alpha, "runs": {}}
        for label, use_slot_bottleneck, alpha in phase2_runs:
            run_id = f"{cfg.experiment_name}--phase2-{label}"
            run_dir = _run_training_if_needed(
                cfg,
                repo_root,
                run_id,
                use_slot_bottleneck=use_slot_bottleneck,
                max_steps=cfg.phase2_steps,
                slot_alpha=alpha,
            )
            eval_summaries = {}
            for step in phase2_eval_steps:
                eval_summaries[step] = _run_eval_if_needed(
                    cfg,
                    repo_root,
                    _latest_checkpoint(run_dir, step),
                    f"{run_id}--eval-{step}",
                    cfg.phase2_eval_trials_per_task,
                )

            phase2_payload["runs"][label] = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "use_slot_bottleneck": use_slot_bottleneck,
                "slot_recon_loss_weight": alpha,
                "eval_success_rates": {
                    str(step): summary["final_success_rate"] for step, summary in eval_summaries.items()
                },
            }

        _write_json(phase2_results_path, phase2_payload)
        _write_json(
            final_report_path,
            {
                "alpha_plan_path": str(alpha_plan_path),
                "phase1_results_path": str(phase1_results_path),
                "phase2_results_path": str(phase2_results_path),
                "best_alpha": best_alpha,
                "baseline_eval_success_rates": phase2_payload["runs"]["baseline"]["eval_success_rates"],
                "ocl_eval_success_rates": phase2_payload["runs"]["ocl"]["eval_success_rates"],
            },
        )


if __name__ == "__main__":
    run_slot_ablation()
