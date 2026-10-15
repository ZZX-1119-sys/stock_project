r"""Train and evaluate one stock forecasting model on the agreed CSV splits.

Example (from the repository root, in PowerShell):
    .\.venv\Scripts\python.exe scripts\train_evaluate.py --data E:\a_stock_data.zip --model mambats --window 96

The full test split is evaluated only after validation-based checkpoint
selection. Optional batch limits are intended for pipeline smoke runs; any
limited run is marked as smoke-only in its result row.
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_FEATURE_COLS, StockDataset
from src.models import create_model, list_models
from src.models.base import count_parameters


RESULT_FIELDS = [
    "run_id",
    "model",
    "lookback_len",
    "pred_len",
    "feature_count",
    "train_fraction",
    "sample_seed",
    "train_samples",
    "val_samples",
    "test_samples",
    "batch_size",
    "threads",
    "seed",
    "requested_epochs",
    "patience",
    "learning_rate",
    "weight_decay",
    "best_epoch",
    "best_val_rmse",
    "test_rmse",
    "test_mae",
    "train_time_sec",
    "validation_time_sec",
    "fit_time_sec",
    "test_forward_time_sec",
    "test_forward_ms_per_sample",
    "total_parameters",
    "trainable_parameters",
    "device",
    "smoke_only",
    "checkpoint",
]


class TrainingPaused(Exception):
    """A planned pause after writing a resumable checkpoint."""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    data_path,
    split,
    window,
    pred_len,
    batch_size,
    seed=None,
    train_fraction=1.0,
    sample_seed=42,
):
    dataset = StockDataset(
        data_path=data_path,
        split=split,
        lookback_len=window,
        pred_len=pred_len,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col="close",
    )
    if split == "train" and train_fraction < 1.0:
        by_code = defaultdict(list)
        for index, (code, _) in enumerate(dataset._samples):
            by_code[code].append(index)
        sampling_generator = torch.Generator().manual_seed(sample_seed)
        selected = []
        for indices in by_code.values():
            count = max(1, round(len(indices) * train_fraction))
            positions = torch.randperm(
                len(indices), generator=sampling_generator
            )[:count].tolist()
            selected.extend(indices[position] for position in positions)
        dataset = Subset(dataset, sorted(selected))
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        generator=generator,
        pin_memory=False,
    )
    return dataset, loader


def _take_batches(loader, max_batches):
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


def evaluate(
    model,
    loader,
    device,
    max_batches=None,
    measure_forward=False,
    progress_every=100,
    label="eval",
):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    value_count = 0
    forward_seconds = 0.0
    sample_count = 0

    total_batches = min(len(loader), max_batches or len(loader))
    start_all = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (x, y) in enumerate(_take_batches(loader, max_batches), 1):
            x = x.to(device)
            target = y.to(device).unsqueeze(-1)
            start = time.perf_counter()
            prediction = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if measure_forward:
                forward_seconds += time.perf_counter() - start

            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"Prediction shape {tuple(prediction.shape)} does not match "
                    f"target shape {tuple(target.shape)}"
                )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Model produced NaN or infinite predictions")

            difference = prediction.to(torch.float64) - target.to(torch.float64)
            squared_error += difference.square().sum().item()
            absolute_error += difference.abs().sum().item()
            value_count += difference.numel()
            sample_count += x.shape[0]
            if batch_index % progress_every == 0 or batch_index == total_batches:
                print(
                    f"{label} batch={batch_index}/{total_batches} "
                    f"elapsed_sec={time.perf_counter() - start_all:.1f}",
                    flush=True,
                )

    if value_count == 0:
        raise RuntimeError("Evaluation processed no samples")
    return {
        "rmse": math.sqrt(squared_error / value_count),
        "mae": absolute_error / value_count,
        "forward_seconds": forward_seconds,
        "forward_ms_per_sample": (
            forward_seconds * 1000 / sample_count if sample_count else float("nan")
        ),
        "samples": sample_count,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    max_batches=None,
    progress_every=100,
    save_every=100,
    resume_batch=0,
    epoch_generator_state=None,
    resume_rng_state=None,
    running_squared_error=0.0,
    running_value_count=0,
    checkpoint_callback=None,
    pause_after_batch=None,
):
    model.train()
    squared_error = running_squared_error
    value_count = running_value_count
    total_batches = min(len(loader), max_batches or len(loader))
    if not 0 <= resume_batch <= total_batches:
        raise ValueError("resume batch is outside the training epoch")
    if epoch_generator_state is None:
        epoch_generator_state = loader.generator.get_state().clone()
    else:
        loader.generator.set_state(epoch_generator_state)

    epoch_start = time.perf_counter()
    iterator = iter(loader)
    for batch_index in range(total_batches):
        x, y = next(iterator)
        if batch_index < resume_batch:
            continue
        if batch_index == resume_batch and resume_rng_state is not None:
            torch.set_rng_state(resume_rng_state)
        x = x.to(device)
        target = y.to(device).unsqueeze(-1)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became NaN or infinite")
        loss.backward()
        optimizer.step()
        squared_error += loss.detach().item() * target.numel()
        value_count += target.numel()
        completed = batch_index + 1
        if completed % progress_every == 0 or completed == total_batches:
            elapsed = time.perf_counter() - epoch_start
            processed = completed - resume_batch
            remaining = (total_batches - completed) * elapsed / max(processed, 1)
            print(
                f"epoch={epoch} train_batch={completed}/{total_batches} "
                f"elapsed_sec={elapsed:.1f} eta_sec={remaining:.1f}",
                flush=True,
            )
        if checkpoint_callback is not None and (
            completed % save_every == 0
            or completed == total_batches
            or completed == pause_after_batch
        ):
            checkpoint_callback(
                next_batch=completed,
                epoch_generator_state=epoch_generator_state,
                squared_error=squared_error,
                value_count=value_count,
            )
        if completed == pause_after_batch:
            raise TrainingPaused(f"paused after training batch {completed}")

    if resume_batch == total_batches and resume_rng_state is not None:
        torch.set_rng_state(resume_rng_state)

    if value_count == 0:
        raise RuntimeError("Training processed no samples")
    return math.sqrt(squared_error / value_count)


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            previous_fields = reader.fieldnames
            if previous_fields != RESULT_FIELDS:
                previous_rows = list(reader)
        if previous_fields != RESULT_FIELDS:
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=RESULT_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for previous in previous_rows:
                    writer.writerow({name: previous.get(name, "") for name in RESULT_FIELDS})
            os.replace(temporary, path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_progress(path, payload):
    """Replace the prior progress file only after the new file is complete."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(r"E:\a_stock_data.zip"))
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--resume", type=Path, help="Resume from a .last.pt file")
    parser.add_argument("--progress-every-batches", type=int, default=100)
    parser.add_argument("--save-every-batches", type=int, default=100)
    parser.add_argument(
        "--pause-after-batches",
        type=int,
        help="Stop after this batch in the current epoch; resume with --resume",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Optional smoke-run limit; limited runs are not thesis results",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Optional smoke-run limit; limited runs are not thesis results",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        help="Optional smoke-run limit; limited runs are not thesis results",
    )
    args = parser.parse_args()

    if args.window < 16:
        parser.error("window must be at least 16 for the default patch length")
    if args.pred_len != 1:
        parser.error("this task currently predicts exactly one next-day close")
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("batch-size, epochs, and patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.threads < 1:
        parser.error("learning-rate must be positive; weight-decay nonnegative; threads positive")
    if args.progress_every_batches < 1 or args.save_every_batches < 1:
        parser.error("progress-every-batches and save-every-batches must be positive")
    if not 0 < args.train_fraction <= 1:
        parser.error("train-fraction must be greater than 0 and at most 1")
    if args.pause_after_batches is not None and args.pause_after_batches < 1:
        parser.error("pause-after-batches must be positive")
    for name in ("max_train_batches", "max_val_batches", "max_test_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if not args.data.is_file() and not args.data.is_dir():
        parser.error(f"data path does not exist: {args.data}")

    if args.resume is not None:
        if not args.resume.is_file():
            parser.error(f"resume checkpoint does not exist: {args.resume}")
        args.output_dir = args.resume.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = args.output_dir / ".write_probe"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        parser.error(f"output directory is not writable: {args.output_dir}: {exc}")

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_count = len(DEFAULT_FEATURE_COLS)
    model = create_model(
        args.model,
        seq_len=args.window,
        pred_len=args.pred_len,
        enc_in=feature_count,
        target_dim=1,
        target_idx=3,
    ).to(device)
    parameter_info = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    run_config = {
        "data": str(args.data.resolve()),
        "model": args.model,
        "window": args.window,
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "train_fraction": args.train_fraction,
        "sample_seed": args.sample_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "threads": args.threads,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
    }
    progress = None
    if args.resume is not None:
        progress = torch.load(args.resume, map_location=device, weights_only=False)
        if progress.get("version") != 1 or progress.get("config") != run_config:
            parser.error("resume checkpoint does not match this run's configuration")
        model.load_state_dict(progress["model_state_dict"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        torch.set_rng_state(progress["torch_rng_state"])
        print(
            f"resuming run={progress['run_id']} epoch={progress['epoch']} "
            f"next_batch={progress['next_batch']}",
            flush=True,
        )

    run_id = (
        progress["run_id"]
        if progress is not None
        else f"{args.model}_L{args.window}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    last_path = args.resume or args.output_dir / f"{run_id}.last.pt"

    train_dataset, train_loader = make_loader(
        args.data,
        "train",
        args.window,
        args.pred_len,
        args.batch_size,
        args.seed,
        train_fraction=args.train_fraction,
        sample_seed=args.sample_seed,
    )
    val_dataset, val_loader = make_loader(
        args.data, "val", args.window, args.pred_len, args.batch_size
    )
    if progress is not None:
        train_loader.generator.set_state(progress["loader_generator_state"])
    print(
        f"model={args.model} device={device} train_samples={len(train_dataset)} "
        f"val_samples={len(val_dataset)} parameters={parameter_info['total_parameters']}",
        flush=True,
    )

    best_val_rmse = progress["best_val_rmse"] if progress else float("inf")
    best_epoch = progress["best_epoch"] if progress else 0
    best_state = progress["best_state"] if progress else None
    epochs_without_improvement = (
        progress["epochs_without_improvement"] if progress else 0
    )
    train_seconds = progress["train_seconds"] if progress else 0.0
    validation_seconds = progress["validation_seconds"] if progress else 0.0
    fit_seconds_base = progress["fit_seconds"] if progress else 0.0
    start_epoch = progress["epoch"] if progress else 1
    initial_batch = progress["next_batch"] if progress else 0
    initial_generator_state = progress["epoch_generator_state"] if progress else None
    initial_rng_state = progress["torch_rng_state"] if progress else None
    initial_squared_error = progress["running_squared_error"] if progress else 0.0
    initial_value_count = progress["running_value_count"] if progress else 0
    training_complete = progress["training_complete"] if progress else False
    fit_start = time.perf_counter()

    def write_checkpoint(
        epoch,
        next_batch,
        epoch_generator_state,
        squared_error,
        value_count,
        completed=False,
        train_seconds_at_save=None,
    ):
        payload = {
            "version": 1,
            "config": run_config,
            "run_id": run_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "epoch": epoch,
            "next_batch": next_batch,
            "epoch_generator_state": epoch_generator_state,
            "loader_generator_state": train_loader.generator.get_state().clone(),
            "running_squared_error": squared_error,
            "running_value_count": value_count,
            "best_val_rmse": best_val_rmse,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "epochs_without_improvement": epochs_without_improvement,
            "train_seconds": (
                train_seconds if train_seconds_at_save is None else train_seconds_at_save
            ),
            "validation_seconds": validation_seconds,
            "fit_seconds": fit_seconds_base + time.perf_counter() - fit_start,
            "training_complete": completed,
        }
        save_progress(last_path, payload)

    if progress and device.type == "cuda" and progress["cuda_rng_states"] is not None:
        torch.cuda.set_rng_state_all(progress["cuda_rng_states"])

    if not training_complete:
        for epoch in range(start_epoch, args.epochs + 1):
            resume_this_epoch = epoch == start_epoch and progress is not None
            resume_batch = initial_batch if resume_this_epoch else 0
            train_start = time.perf_counter()

            def save_training_batch(
                next_batch, epoch_generator_state, squared_error, value_count
            ):
                write_checkpoint(
                    epoch=epoch,
                    next_batch=next_batch,
                    epoch_generator_state=epoch_generator_state,
                    squared_error=squared_error,
                    value_count=value_count,
                    train_seconds_at_save=(
                        train_seconds + time.perf_counter() - train_start
                    ),
                )

            try:
                train_rmse = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    epoch=epoch,
                    max_batches=args.max_train_batches,
                    progress_every=args.progress_every_batches,
                    save_every=args.save_every_batches,
                    resume_batch=resume_batch,
                    epoch_generator_state=(
                        initial_generator_state if resume_this_epoch else None
                    ),
                    resume_rng_state=initial_rng_state if resume_this_epoch else None,
                    running_squared_error=(
                        initial_squared_error if resume_this_epoch else 0.0
                    ),
                    running_value_count=(
                        initial_value_count if resume_this_epoch else 0
                    ),
                    checkpoint_callback=save_training_batch,
                    pause_after_batch=args.pause_after_batches,
                )
            except TrainingPaused as exc:
                print(f"{exc}; resume_checkpoint={last_path}", flush=True)
                return
            train_seconds += time.perf_counter() - train_start
            validation_start = time.perf_counter()
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.max_val_batches,
                progress_every=args.progress_every_batches,
                label=f"epoch={epoch} val",
            )
            validation_seconds += time.perf_counter() - validation_start
            print(
                f"epoch={epoch}/{args.epochs} train_rmse={train_rmse:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}",
                flush=True,
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            training_complete = (
                epochs_without_improvement >= args.patience or epoch == args.epochs
            )
            write_checkpoint(
                epoch=epoch + 1,
                next_batch=0,
                epoch_generator_state=None,
                squared_error=0.0,
                value_count=0,
                completed=training_complete,
            )
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    fit_time = fit_seconds_base + time.perf_counter() - fit_start
    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was selected")
    model.load_state_dict(best_state)
    train_sample_count = len(train_dataset)
    val_sample_count = len(val_dataset)
    del train_loader, train_dataset, val_loader, val_dataset

    test_dataset, test_loader = make_loader(
        args.data, "test", args.window, args.pred_len, args.batch_size
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        args.max_test_batches,
        measure_forward=True,
        progress_every=args.progress_every_batches,
        label="test",
    )

    checkpoint_path = args.output_dir / f"{run_id}.pt"
    torch.save(
        {
            "model": args.model,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "seed": args.seed,
            "lookback_len": args.window,
            "pred_len": args.pred_len,
            "feature_cols": DEFAULT_FEATURE_COLS,
            "target_col": "close",
            "target_idx": 3,
            "batch_size": args.batch_size,
            "train_fraction": args.train_fraction,
            "sample_seed": args.sample_seed,
            "threads": args.threads,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        checkpoint_path,
    )
    smoke_only = any(
        getattr(args, name) is not None
        for name in ("max_train_batches", "max_val_batches", "max_test_batches")
    )
    row = {
        "run_id": run_id,
        "model": args.model,
        "lookback_len": args.window,
        "pred_len": args.pred_len,
        "feature_count": feature_count,
        "train_fraction": args.train_fraction,
        "sample_seed": args.sample_seed,
        "train_samples": train_sample_count,
        "val_samples": val_sample_count,
        "test_samples": len(test_dataset),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "seed": args.seed,
        "requested_epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "train_time_sec": round(train_seconds, 4),
        "validation_time_sec": round(validation_seconds, 4),
        "fit_time_sec": round(fit_time, 4),
        "test_forward_time_sec": round(test_metrics["forward_seconds"], 6),
        "test_forward_ms_per_sample": round(test_metrics["forward_ms_per_sample"], 6),
        **parameter_info,
        "device": str(device),
        "smoke_only": smoke_only,
        "checkpoint": str(checkpoint_path),
    }
    result_path = args.output_dir / "results.csv"
    append_result(result_path, row)
    last_path.unlink(missing_ok=True)

    print(f"test_samples={test_metrics['samples']}", flush=True)
    print(f"test_rmse={test_metrics['rmse']:.6f} test_mae={test_metrics['mae']:.6f}")
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_results={result_path}")
    if smoke_only:
        print("WARNING: batch-limited smoke run; do not use these metrics as paper results.")


if __name__ == "__main__":
    main()
