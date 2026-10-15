"""Windowed stock forecasting dataset for pre-split CSV or ZIP files."""

from pathlib import Path
import csv
from datetime import datetime
import io
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pctChg",
    "macd",
    "macd_signal",
    "rsi",
    "boll_upper",
    "boll_lower",
    "log_ret",
    "volatility_60",
]

_SPLITS = ("train", "val", "test")
_SPLIT_ID = {name: index for index, name in enumerate(_SPLITS)}
_SPLIT_FILENAMES = {
    "train": ("train.csv", "us_train.csv"),
    "val": ("val.csv", "us_val.csv"),
    "test": ("test.csv", "us_test.csv"),
}


class StockDataset(Dataset):
    """Return ``(x, y)`` windows, with shapes ``[lookback, features]`` and
    ``[pred_len]``. For validation/test inputs, preceding splits supply
    historical context; labels always remain inside the requested split.

    ``data_path`` may be an extracted directory containing split CSV files or
    a ZIP archive such as ``a_stock_data.zip``. The archive is read in place.
    """

    def __init__(
        self,
        data_path,
        split="train",
        lookback_len=96,
        pred_len=1,
        feature_cols=None,
        target_col="close",
    ):
        if split not in _SPLIT_ID:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        if lookback_len < 1 or pred_len < 1:
            raise ValueError("lookback_len and pred_len must be positive")

        self.data_path = Path(data_path)
        self.split = split
        self.lookback_len = int(lookback_len)
        self.pred_len = int(pred_len)
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.target_col = target_col
        self.num_features = len(self.feature_cols)
        context_splits = _SPLITS[:_SPLIT_ID[split] + 1]

        rows = []
        for source_split in context_splits:
            split_rows = self._read_split(source_split)
            for row in split_rows:
                row["_split_id"] = _SPLIT_ID[source_split]
            rows.extend(split_rows)

        if not rows:
            raise ValueError("No rows found in requested split files")
        required_cols = ["date", "code", *self.feature_cols, self.target_col]
        missing = [name for name in required_cols if name not in rows[0]]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")

        seen = set()
        by_code = {}
        numeric_cols = list(dict.fromkeys(self.feature_cols + [self.target_col]))
        for row in rows:
            try:
                date_value = datetime.fromisoformat(row["date"].strip())
                code = row["code"].strip()
                numeric_values = [float(row[name]) for name in numeric_cols]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid CSV value in row {row}") from exc
            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    f"Feature/target data contains NaN or infinite values: {code} {row['date']}"
                )
            key = (code, date_value)
            if key in seen:
                raise ValueError(f"Duplicate (code, date) row found: {code} {row['date']}")
            seen.add(key)
            row["_date_value"] = date_value
            row["_numeric_values"] = dict(zip(numeric_cols, numeric_values))
            by_code.setdefault(code, []).append(row)

        self._series = {}
        self._samples = []
        target_split_id = _SPLIT_ID[split]
        for code, group in by_code.items():
            group.sort(key=lambda row: row["_date_value"])
            x_values = np.asarray(
                [
                    [row["_numeric_values"][name] for name in self.feature_cols]
                    for row in group
                ],
                dtype=np.float32,
            )
            y_values = np.asarray(
                [row["_numeric_values"][self.target_col] for row in group],
                dtype=np.float32,
            )
            split_ids = np.asarray([row["_split_id"] for row in group], dtype=np.int8)
            self._series[code] = (x_values, y_values)

            target_positions = np.flatnonzero(split_ids == target_split_id)
            for target_start in target_positions:
                target_end = target_start + self.pred_len
                if target_start < self.lookback_len or target_end > len(group):
                    continue
                if not np.all(split_ids[target_start:target_end] == target_split_id):
                    continue
                self._samples.append((code, int(target_start)))

        if not self._samples:
            raise ValueError(
                f"No windows available for split={split!r}; check data and window lengths"
            )

    def _read_split(self, split):
        candidates = _SPLIT_FILENAMES[split]
        if self.data_path.is_dir():
            for filename in candidates:
                path = self.data_path / filename
                if path.is_file():
                    with path.open("r", encoding="utf-8-sig", newline="") as stream:
                        return list(csv.DictReader(stream))
            raise FileNotFoundError(
                f"Could not find a {split} CSV in {self.data_path}; "
                f"tried {candidates}"
            )

        if self.data_path.is_file() and self.data_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(self.data_path) as archive:
                entries = {Path(name).name: name for name in archive.namelist()}
                for filename in candidates:
                    if filename in entries:
                        with archive.open(entries[filename]) as binary_stream:
                            text_stream = io.TextIOWrapper(
                                binary_stream, encoding="utf-8-sig", newline=""
                            )
                            return list(csv.DictReader(text_stream))
            raise FileNotFoundError(
                f"Could not find a {split} CSV in {self.data_path}; "
                f"tried {candidates}"
            )

        raise FileNotFoundError(
            f"data_path must be a ZIP archive or directory: {self.data_path}"
        )

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        code, target_start = self._samples[index]
        x_values, y_values = self._series[code]
        x = x_values[target_start - self.lookback_len:target_start]
        y = y_values[target_start:target_start + self.pred_len]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())