#!/usr/bin/env python3
"""Convert a LeRobot v3 RoboDojo EE dataset from 16D to X-VLA 20D."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.xvla_ee import ee16_to_xvla20, xvla20_to_ee16  # noqa: E402

NUMERIC_FEATURES = ("observation.state", "action")
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
XVLA_NAMES = [
    "l_x",
    "l_y",
    "l_z",
    "l_rot6d_0",
    "l_rot6d_1",
    "l_rot6d_2",
    "l_rot6d_3",
    "l_rot6d_4",
    "l_rot6d_5",
    "l_g",
    "r_x",
    "r_y",
    "r_z",
    "r_rot6d_0",
    "r_rot6d_1",
    "r_rot6d_2",
    "r_rot6d_3",
    "r_rot6d_4",
    "r_rot6d_5",
    "r_g",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output dataset root (default: a sibling directory named <source>_6d)",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _validate_source(source: Path) -> dict[str, Any]:
    info_path = source / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = _load_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    for feature in NUMERIC_FEATURES:
        shape = info.get("features", {}).get(feature, {}).get("shape")
        if shape != [16]:
            raise ValueError(f"{feature} must have shape [16], got {shape}")
    if not list((source / "data").rglob("*.parquet")):
        raise FileNotFoundError(f"No parquet files under {source / 'data'}")
    return info


def _fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), 20
    )


def _update_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    encoded = metadata.get(b"huggingface")
    if encoded is not None:
        huggingface = json.loads(encoded)
        features = huggingface.get("info", {}).get("features", {})
        for feature in NUMERIC_FEATURES:
            if feature in features:
                features[feature]["length"] = 20
        huggingface.pop("fingerprint", None)
        metadata[b"huggingface"] = json.dumps(
            huggingface, separators=(",", ":")
        ).encode()
    return table.replace_schema_metadata(metadata)


def _transform_data_table(table: pa.Table) -> tuple[pa.Table, dict[str, np.ndarray]]:
    converted: dict[str, np.ndarray] = {}
    for feature in NUMERIC_FEATURES:
        values = np.asarray(table[feature].to_pylist(), dtype=np.float32)
        converted[feature] = ee16_to_xvla20(values)
        column_index = table.schema.get_field_index(feature)
        table = table.set_column(
            column_index, feature, _fixed_size_list(converted[feature])
        )
    return _update_huggingface_metadata(table), converted


def _numeric_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 20 or values.shape[0] == 0:
        raise ValueError(f"Expected non-empty (N, 20) values, got {values.shape}")
    quantiles = np.quantile(values, (0.01, 0.10, 0.50, 0.90, 0.99), axis=0)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _replace_episode_stats(
    table: pa.Table, memory_maps: dict[str, np.memmap]
) -> tuple[pa.Table, int]:
    from_indices = np.asarray(table["dataset_from_index"].to_pylist(), dtype=np.int64)
    to_indices = np.asarray(table["dataset_to_index"].to_pylist(), dtype=np.int64)
    for feature in NUMERIC_FEATURES:
        rows = [
            _numeric_stats(memory_maps[feature][start:end])
            for start, end in zip(from_indices, to_indices)
        ]
        for stat_name in STAT_NAMES:
            column_name = f"stats/{feature}/{stat_name}"
            column_index = table.schema.get_field_index(column_name)
            if column_index < 0:
                raise ValueError(f"Episode metadata is missing {column_name}")
            table = table.set_column(
                column_index,
                column_name,
                pa.array(
                    [row[stat_name] for row in rows],
                    type=table.schema.field(column_index).type,
                ),
            )
    return table, len(from_indices)


def _symlink_file(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(source.resolve())


def _symlink_tree_files(source: Path, output: Path) -> int:
    if not source.exists():
        return 0
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _symlink_file(path, output / path.relative_to(source))
            count += 1
    return count


def _validate_output(
    output: Path, info: dict[str, Any], memory_maps: dict[str, np.memmap]
) -> None:
    total_frames = int(info["total_frames"])
    for feature in NUMERIC_FEATURES:
        values = memory_maps[feature]
        if values.shape != (total_frames, 20):
            raise ValueError(f"{feature} converted shape mismatch: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{feature} contains non-finite converted values")
        grippers = values[:, [9, 19]]
        if np.any(grippers < -1e-5) or np.any(grippers > 1 + 1e-5):
            raise ValueError(f"{feature} grippers fall outside [0, 1]")

    sample_indices = np.linspace(
        0, total_frames - 1, min(total_frames, 1024), dtype=np.int64
    )
    for feature in NUMERIC_FEATURES:
        sample = np.asarray(memory_maps[feature][sample_indices])
        restored = xvla20_to_ee16(sample)
        reconstructed = ee16_to_xvla20(restored)
        error = float(np.max(np.abs(sample - reconstructed)))
        if error > 1e-4:
            raise ValueError(
                f"{feature} rotation round-trip error is too high: {error}"
            )

    output_info = _load_json(output / "meta/info.json")
    output_stats = _load_json(output / "meta/stats.json")
    for feature in NUMERIC_FEATURES:
        if output_info["features"][feature]["shape"] != [20]:
            raise ValueError(f"Output info.json still has an invalid {feature} shape")
        for stat_name in STAT_NAMES:
            expected_length = 1 if stat_name == "count" else 20
            if len(output_stats[feature][stat_name]) != expected_length:
                raise ValueError(f"Output stats.json has invalid {feature}/{stat_name}")


def convert_dataset(source: Path, output: Path | None = None) -> dict[str, Any]:
    """Create a converted dataset at ``output`` without modifying ``source``."""

    source = source.resolve()
    output = (output or source.with_name(f"{source.name}_6d")).resolve()
    info = _validate_source(source)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    total_frames = int(info["total_frames"])
    memory_maps = {
        feature: np.memmap(
            temporary / f".{feature.replace('.', '_')}.mmap",
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, 20),
        )
        for feature in NUMERIC_FEATURES
    }

    try:
        shutil.copytree(source / "meta", temporary / "meta", copy_function=shutil.copy2)
        video_count = _symlink_tree_files(source / "videos", temporary / "videos")

        offset = 0
        data_files = sorted((source / "data").rglob("*.parquet"))
        for source_path in data_files:
            table = pq.read_table(source_path)
            if "index" in table.column_names:
                indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
                expected = np.arange(offset, offset + table.num_rows, dtype=np.int64)
                if not np.array_equal(indices, expected):
                    raise ValueError(f"Non-contiguous global index in {source_path}")
            table, converted = _transform_data_table(table)
            end = offset + table.num_rows
            if end > total_frames:
                raise ValueError("Parquet rows exceed info.json total_frames")
            for feature in NUMERIC_FEATURES:
                memory_maps[feature][offset:end] = converted[feature]
            target_path = temporary / "data" / source_path.relative_to(source / "data")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target_path, compression="zstd")
            offset = end
        if offset != total_frames:
            raise ValueError(f"Converted {offset} rows, expected {total_frames}")
        for mmap in memory_maps.values():
            mmap.flush()

        output_info = _load_json(temporary / "meta/info.json")
        for feature in NUMERIC_FEATURES:
            output_info["features"][feature]["shape"] = [20]
            output_info["features"][feature]["names"] = [XVLA_NAMES]
        _write_json(temporary / "meta/info.json", output_info)

        output_stats = _load_json(temporary / "meta/stats.json")
        for feature in NUMERIC_FEATURES:
            output_stats[feature] = _numeric_stats(memory_maps[feature])
        _write_json(temporary / "meta/stats.json", output_stats)

        episode_count = 0
        episode_files = sorted((temporary / "meta/episodes").rglob("*.parquet"))
        if not episode_files:
            raise FileNotFoundError("No episode metadata parquet files found")
        for episode_path in episode_files:
            episode_table = pq.read_table(episode_path)
            episode_table, rows = _replace_episode_stats(episode_table, memory_maps)
            pq.write_table(episode_table, episode_path, compression="zstd")
            episode_count += rows
        if episode_count != int(info["total_episodes"]):
            raise ValueError(
                f"Converted {episode_count} episodes, expected {info['total_episodes']}"
            )

        _validate_output(temporary, output_info, memory_maps)
        for mmap in memory_maps.values():
            mmap._mmap.close()
        for mmap_path in temporary.glob(".*.mmap"):
            mmap_path.unlink()
        os.replace(temporary, output)
    except Exception:
        for mmap in memory_maps.values():
            if getattr(mmap, "_mmap", None) is not None:
                mmap._mmap.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "source": str(source),
        "output": str(output),
        "frames": total_frames,
        "episodes": int(info["total_episodes"]),
        "fps": info["fps"],
        "data_files": len(data_files),
        "video_files": video_count,
        "video_mode": "symlink",
    }


def main() -> None:
    args = parse_args()
    summary = convert_dataset(args.source, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
