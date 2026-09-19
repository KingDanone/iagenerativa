"""Loop de treinamento: AdamW + warmup/cosseno + AMP + acumulacao + checkpoints."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import ensure_dir, lr_schedule, save_checkpoint, vram_snapshot


def unwrap_amp_dtype(device: torch.device, use_amp: bool) -> tuple[bool, torch.dtype, bool]:
    """Retorna (amp_ok, dtype, use_scaler). Prefere bf16 (sem scaler, estavel)."""
    if not use_amp or device.type != "cuda":
        return False, torch.float32, False
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def evaluate(model, loader, device, eval_iters: int = 50) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_loop(cfg: dict, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None, tokenizer=None) -> dict:
    device = torch.device(cfg.get("_device_resolved", "cpu"))
    model = model.to(device)
    model.train()

    max_steps = int(cfg["max_steps"])
    base_lr = float(cfg["learning_rate"])
    min_lr = float(cfg.get("min_lr", base_lr * 0.1))
    warmup = int(cfg.get("warmup_steps", 0))
    accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    eval_interval = int(cfg.get("eval_interval", 250))
    eval_iters = int(cfg.get("eval_iters", 50))
    log_interval = int(cfg.get("log_interval", 25))
    save_interval = int(cfg.get("save_interval", 500))
    patience = cfg.get("early_stopping_patience")

    try:
        opt = torch.optim.AdamW(
            model.parameters(), lr=base_lr,
            weight_decay=float(cfg.get("weight_decay", 0.01)),
            betas=(0.9, 0.95), fused=True,
        )
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(
            model.parameters(), lr=base_lr,
            weight_decay=float(cfg.get("weight_decay", 0.01)),
            betas=(0.9, 0.95),
        )
    start_step = int(cfg.get("_start_step", 0))
    if "_opt_state" in cfg and cfg["_opt_state"] is not None:
        try:
            opt.load_state_dict(cfg["_opt_state"])
        except Exception:
            pass
    # scheduler manual via lr_schedule (mais explicavel que LambdaLR)
    amp_ok, amp_dtype, use_scaler = unwrap_amp_dtype(device, bool(cfg.get("use_amp", True)))
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    ckpt_dir = ensure_dir(cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    log_dir = ensure_dir(cfg.get("log_dir", "logs"))
    log_path = Path(log_dir) / "train.log"

    best_val = float(cfg.get("_best_val", "inf"))
    no_improve = 0
    t0 = time.time()
    tokens_done = 0
    micro_bs = int(cfg.get("batch_size", 8))
    block = int(cfg.get("block_size", 256))
    step = start_step
    train_iter = iter(train_loader)
    max_vram = 0.0

    def get_batch():
        nonlocal train_iter
        try:
            b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            b = next(train_iter)
        return b

    history = []
    model.train()
    opt.zero_grad(set_to_none=True)
    while step < max_steps:
        step_start = time.time()
        acc_loss = 0.0
        for _ in range(accum):
            x, y = get_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_ok else contextlib.nullcontext()
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            acc_loss += loss.item() * accum
            tokens_done += x.numel()
        acc_loss /= accum
        if grad_clip and grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = lr_schedule(step, max_steps=max_steps, base_lr=base_lr, min_lr=min_lr, warmup_steps=warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

        dt = time.time() - step_start
        eff_bs = micro_bs * accum
        tps = (eff_bs * block) / max(dt, 1e-6)
        snap = vram_snapshot()
        if snap.get("max_allocated_gb"):
            max_vram = max(max_vram, snap["max_allocated_gb"])
        step += 1

        if step % log_interval == 0 or step == 1:
            msg = (f"step={step}/{max_steps} train_loss={acc_loss:.4f} lr={lr:.2e} "
                   f"tok/s={tps:.0f} eff_bs={eff_bs} vram={snap}")
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        do_eval = (step % eval_interval == 0 or step == max_steps) and val_loader is not None
        val_loss = None
        if do_eval:
            val_loss = evaluate(model, val_loader, device, eval_iters)
            ppl = _safe_ppl(val_loss)
            msg = f"[eval] step={step} val_loss={val_loss:.4f} ppl={ppl:.2f}"
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            history.append({"step": step, "train_loss": acc_loss, "val_loss": val_loss})
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1
            ckpt_common = dict(
                model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                step=step, config={k: v for k, v in cfg.items() if not k.startswith("_")},
                train_loss=acc_loss, val_loss=val_loss, best_val=best_val,
            )
            save_checkpoint(ckpt_dir / "latest.pt", ckpt_common)
            if improved:
                save_checkpoint(ckpt_dir / "best.pt", ckpt_common)
            if patience and no_improve >= patience:
                print(f"early stopping (patience={patience})", flush=True)
                break
        elif step % save_interval == 0:
            save_checkpoint(ckpt_dir / "latest.pt", dict(
                model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                step=step, config={k: v for k, v in cfg.items() if not k.startswith("_")},
                train_loss=acc_loss, val_loss=None, best_val=best_val,
            ))

    total_time = time.time() - t0
    save_checkpoint(ckpt_dir / "latest.pt", dict(
        model_state=model.state_dict(), optimizer_state=opt.state_dict(),
        step=step, config={k: v for k, v in cfg.items() if not k.startswith("_")},
        train_loss=acc_loss, val_loss=val_loss if 'val_loss' in locals() else None, best_val=best_val,
    ))
    return {
        "steps": step, "best_val": best_val,
        "total_time_s": round(total_time, 1),
        "tokens_done": tokens_done,
        "avg_tokens_per_s": round(tokens_done / max(total_time, 1e-6), 1),
        "max_vram_gb": max_vram,
        "history": history,
    }


def _safe_ppl(loss: float) -> float:
    import math
    try:
        return math.exp(loss) if loss < 20 else float("inf")
    except OverflowError:
        return float("inf")
