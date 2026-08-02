import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.convert_lerobot_ee16_to_xvla20 import (
    NUMERIC_FEATURES,
    STAT_NAMES,
    convert_dataset,
)
from utils.xvla_ee import ee16_to_xvla20


def _stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    quantiles = np.quantile(values, (0.01, 0.10, 0.50, 0.90, 0.99), axis=0)
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _write_fixture(root: Path) -> tuple[np.ndarray, np.ndarray]:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.cam_high/chunk-000").mkdir(parents=True)

    state = np.zeros((4, 16), dtype=np.float32)
    action = np.zeros((4, 16), dtype=np.float32)
    state[:, 3] = state[:, 11] = 1.0
    action[:, 3] = action[:, 11] = 1.0
    state[:, 0] = np.arange(4)
    action[:, 8] = np.arange(4)
    state[:, [7, 15]] = [[0, 1], [0.25, 0.75], [0.5, 0.5], [1, 0]]
    action[:, [7, 15]] = state[:, [7, 15]]

    list16 = pa.list_(pa.float32(), 16)
    data_table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=list16),
            "action": pa.array(action.tolist(), type=list16),
            "timestamp": pa.array(np.arange(4, dtype=np.float32) / 25),
            "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
            "index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    hf_metadata = {
        "info": {
            "features": {
                feature: {
                    "feature": {"dtype": "float32", "_type": "Value"},
                    "length": 16,
                    "_type": "List",
                }
                for feature in NUMERIC_FEATURES
            }
        },
        "fingerprint": "old",
    }
    data_table = data_table.replace_schema_metadata(
        {b"huggingface": json.dumps(hf_metadata).encode()}
    )
    pq.write_table(data_table, root / "data/chunk-000/file-000.parquet")

    episode_columns: dict[str, pa.Array] = {
        "episode_index": pa.array([0, 1], type=pa.int64()),
        "tasks": pa.array([["task"], ["task"]], type=pa.list_(pa.string())),
        "length": pa.array([2, 2], type=pa.int64()),
        "dataset_from_index": pa.array([0, 2], type=pa.int64()),
        "dataset_to_index": pa.array([2, 4], type=pa.int64()),
    }
    for feature, values in (("observation.state", state), ("action", action)):
        rows = [_stats(values[:2]), _stats(values[2:])]
        for stat_name in STAT_NAMES:
            item_type = pa.int64() if stat_name == "count" else pa.float64()
            episode_columns[f"stats/{feature}/{stat_name}"] = pa.array(
                [row[stat_name] for row in rows], type=pa.list_(item_type)
            )
    pq.write_table(
        pa.table(episode_columns), root / "meta/episodes/chunk-000/file-000.parquet"
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unified_robot",
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "fps": 25,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": [["old"] * 16],
            },
            "action": {"dtype": "float32", "shape": [16], "names": [["old"] * 16]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(
        json.dumps(
            {
                feature: _stats(values)
                for feature, values in (
                    ("observation.state", state),
                    ("action", action),
                )
            }
        )
    )
    (root / "meta/tasks.parquet").write_bytes(b"unchanged metadata")
    (root / "videos/observation.images.cam_high/chunk-000/file-000.mp4").write_bytes(
        b"video"
    )
    return state, action


def test_convert_dataset_updates_values_and_all_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source_state, source_action = _write_fixture(source)

    summary = convert_dataset(source, output)

    assert summary["frames"] == 4
    assert summary["episodes"] == 2
    assert summary["video_files"] == 1
    assert json.loads((source / "meta/info.json").read_text())["features"]["action"][
        "shape"
    ] == [16]

    output_info = json.loads((output / "meta/info.json").read_text())
    output_stats = json.loads((output / "meta/stats.json").read_text())
    assert output_info["features"]["observation.state"]["shape"] == [20]
    assert output_info["features"]["action"]["shape"] == [20]
    assert len(output_info["features"]["action"]["names"][0]) == 20
    assert len(output_stats["action"]["mean"]) == 20

    data_path = output / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    converted_state = np.asarray(
        data["observation.state"].to_pylist(), dtype=np.float32
    )
    converted_action = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(converted_state, ee16_to_xvla20(source_state))
    np.testing.assert_allclose(converted_action, ee16_to_xvla20(source_action))
    embedded = json.loads(data.schema.metadata[b"huggingface"])
    assert embedded["info"]["features"]["action"]["length"] == 20
    assert "fingerprint" not in embedded

    episodes = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")
    for feature in NUMERIC_FEATURES:
        for stat_name in STAT_NAMES:
            values = episodes[f"stats/{feature}/{stat_name}"].to_pylist()
            expected_length = 1 if stat_name == "count" else 20
            assert all(len(row) == expected_length for row in values)

    source_video = source / "videos/observation.images.cam_high/chunk-000/file-000.mp4"
    output_video = output / "videos/observation.images.cam_high/chunk-000/file-000.mp4"
    assert os.stat(source_video).st_ino == os.stat(output_video).st_ino
