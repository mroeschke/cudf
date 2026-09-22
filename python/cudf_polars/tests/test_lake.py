# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``polars.scan_iceberg`` and ``polars.scan_delta``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal

if TYPE_CHECKING:
    from pathlib import Path

# Polars resolves an Iceberg scan on a worker thread, and importing these
# extension modules from there crashes the interpreter. Importing them here
# means the worker always finds them in ``sys.modules``.
pyiceberg = pytest.importorskip("pyiceberg")
pytest.importorskip("pyiceberg.avro.decoder_fast")
pytest.importorskip("pyiceberg.io.pyarrow")
pytest.importorskip("pyiceberg.table")
deltalake = pytest.importorskip("deltalake")


def iceberg_catalog(tmp_path: Path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=f"file://{warehouse}",
    )
    catalog.create_namespace_if_not_exists("ns")
    return catalog


@pytest.fixture
def flat_iceberg(tmp_path: Path) -> str:
    """An Iceberg table of two data files that share one schema."""
    catalog = iceberg_catalog(tmp_path)
    table = pa.table(
        {
            "a": pa.array([1, 2, 3], type=pa.int64()),
            "b": pa.array(["x", "y", "z"], type=pa.string()),
        }
    )
    tbl = catalog.create_table("ns.flat", schema=table.schema)
    tbl.append(table)
    tbl.append(
        pa.table(
            {
                "a": pa.array([4, 5], type=pa.int64()),
                "b": pa.array(["p", "q"], type=pa.string()),
            }
        )
    )
    return tbl.metadata_location


@pytest.fixture
def evolved_iceberg(tmp_path: Path) -> str:
    """An Iceberg table whose data files disagree on names, types and columns."""
    from pyiceberg.types import DoubleType, LongType

    catalog = iceberg_catalog(tmp_path)
    first = pa.table(
        {
            "a": pa.array([1, 2, 3], type=pa.int32()),
            "b": pa.array(["x", "y", "z"], type=pa.string()),
        }
    )
    tbl = catalog.create_table("ns.evolved", schema=first.schema)
    tbl.append(first)

    with tbl.update_schema() as update:
        update.rename_column("b", "b_renamed")
    with tbl.update_schema() as update:
        update.update_column("a", field_type=LongType())
    with tbl.update_schema() as update:
        update.add_column("c", DoubleType())

    tbl.append(
        pa.table(
            {
                "a": pa.array([4, 5], type=pa.int64()),
                "b_renamed": pa.array(["p", "q"], type=pa.string()),
                "c": pa.array([1.5, 2.5], type=pa.float64()),
            }
        )
    )
    return tbl.metadata_location


@pytest.fixture
def partitioned_iceberg(tmp_path: Path) -> str:
    """An Iceberg table with an identity-transformed partition field."""
    import pyiceberg.schema
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField, StringType

    catalog = iceberg_catalog(tmp_path)
    schema = pyiceberg.schema.Schema(
        NestedField(1, "a", LongType(), required=False),
        NestedField(2, "part", StringType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=2, field_id=1000, transform=IdentityTransform(), name="part"
        )
    )
    tbl = catalog.create_table("ns.partitioned", schema=schema, partition_spec=spec)
    tbl.append(
        pa.table(
            {
                "a": pa.array([1, 2, 3, 4], type=pa.int64()),
                "part": pa.array(["u", "u", "v", "v"], type=pa.string()),
            }
        )
    )
    return tbl.metadata_location


@pytest.fixture
def flat_delta(tmp_path: Path) -> str:
    """A Delta table of two data files."""
    path = tmp_path / "delta"
    deltalake.write_deltalake(
        str(path),
        pa.table(
            {
                "a": pa.array([1, 2, 3], type=pa.int64()),
                "b": pa.array(["x", "y", "z"], type=pa.string()),
            }
        ),
        mode="overwrite",
    )
    deltalake.write_deltalake(
        str(path),
        pa.table(
            {
                "a": pa.array([4, 5], type=pa.int64()),
                "b": pa.array(["p", "q"], type=pa.string()),
            }
        ),
        mode="append",
    )
    return str(path)


@pytest.fixture
def deletion_vector_delta(tmp_path: Path) -> str:
    """A Delta table that records deleted rows in a deletion vector."""
    path = tmp_path / "delta_dv"
    deltalake.write_deltalake(
        str(path),
        pa.table(
            {
                "a": pa.array(list(range(10)), type=pa.int64()),
                "b": pa.array([f"s{i}" for i in range(10)], type=pa.string()),
            }
        ),
        mode="overwrite",
        configuration={"delta.enableDeletionVectors": "true"},
    )
    deltalake.DeltaTable(str(path)).delete("a % 3 == 0")
    return str(path)


def test_scan_iceberg(engine: pl.GPUEngine, flat_iceberg: str) -> None:
    assert_gpu_result_equal(pl.scan_iceberg(flat_iceberg).sort("a"), engine=engine)


def test_scan_delta(engine: pl.GPUEngine, flat_delta: str) -> None:
    assert_gpu_result_equal(pl.scan_delta(flat_delta).sort("a"), engine=engine)


def test_scan_iceberg_schema_evolution(
    engine: pl.GPUEngine, evolved_iceberg: str
) -> None:
    assert_gpu_result_equal(pl.scan_iceberg(evolved_iceberg).sort("a"), engine=engine)


@pytest.mark.parametrize(
    "columns", [["a"], ["b_renamed"], ["c"], ["c", "a"], ["a", "b_renamed", "c"]]
)
def test_scan_iceberg_schema_evolution_projection(
    engine: pl.GPUEngine, evolved_iceberg: str, columns: list[str]
) -> None:
    assert_gpu_result_equal(
        pl.scan_iceberg(evolved_iceberg).select(columns).sort(columns),
        engine=engine,
    )


def test_scan_iceberg_schema_evolution_predicate(
    engine: pl.GPUEngine, evolved_iceberg: str
) -> None:
    assert_gpu_result_equal(
        pl.scan_iceberg(evolved_iceberg).filter(pl.col("a") > 2).sort("a"),
        engine=engine,
    )


@pytest.mark.parametrize("n_rows", [1, 3, 5])
def test_scan_iceberg_schema_evolution_slice(
    engine: pl.GPUEngine, evolved_iceberg: str, n_rows: int
) -> None:
    assert_gpu_result_equal(
        pl.scan_iceberg(evolved_iceberg).sort("a").head(n_rows), engine=engine
    )


def test_scan_iceberg_identity_partition(
    engine: pl.GPUEngine, partitioned_iceberg: str
) -> None:
    assert_gpu_result_equal(
        pl.scan_iceberg(partitioned_iceberg).sort("a"), engine=engine
    )


def test_scan_iceberg_row_index(engine: pl.GPUEngine, evolved_iceberg: str) -> None:
    assert_gpu_result_equal(
        pl.scan_iceberg(evolved_iceberg).with_row_index("idx"), engine=engine
    )


def test_scan_delta_deletion_vectors(
    engine: pl.GPUEngine, deletion_vector_delta: str
) -> None:
    assert_gpu_result_equal(
        pl.scan_delta(deletion_vector_delta).sort("a"), engine=engine
    )


def test_scan_delta_deletion_vectors_count(
    engine: pl.GPUEngine, deletion_vector_delta: str
) -> None:
    assert_gpu_result_equal(
        pl.scan_delta(deletion_vector_delta).select(pl.len()), engine=engine
    )


@pytest.fixture
def position_deletes(tmp_path: Path) -> tuple[str, tuple[str, ...]]:
    """Three data files of four rows, and the deletes of two of them."""
    data = tmp_path / "data"
    for index in range(3):
        path = data / f"part={index}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"a": [4 * index + i for i in range(4)]}).write_parquet(path)

    deletes = tmp_path / "deletes"
    deletes.mkdir()
    paths = []
    for index, positions in enumerate([[0, 3], [1]]):
        path = deletes / f"{index}.parquet"
        pl.DataFrame({"file_path": [""] * len(positions), "pos": positions}).select(
            "file_path", pl.col("pos").cast(pl.Int64)
        ).write_parquet(path)
        paths.append(str(path))
    return str(data), tuple(paths)


def test_scan_position_deletes(
    engine: pl.GPUEngine, position_deletes: tuple[str, tuple[str, ...]]
) -> None:
    data, deletes = position_deletes
    assert_gpu_result_equal(
        pl.scan_parquet(
            data,
            hive_partitioning=False,
            _deletion_files=("iceberg", ({0: [deletes[0]], 2: [deletes[1]]}, {})),
        ).sort("a"),
        engine=engine,
    )


def test_scan_position_deletes_row_index(
    engine: pl.GPUEngine, position_deletes: tuple[str, tuple[str, ...]]
) -> None:
    data, deletes = position_deletes
    assert_gpu_result_equal(
        pl.scan_parquet(
            data,
            hive_partitioning=False,
            _deletion_files=("iceberg", ({0: [deletes[0]], 2: [deletes[1]]}, {})),
        ).with_row_index("idx"),
        engine=engine,
    )


def test_scan_position_deletes_hive(
    engine: pl.GPUEngine, position_deletes: tuple[str, tuple[str, ...]]
) -> None:
    data, deletes = position_deletes
    assert_gpu_result_equal(
        pl.scan_parquet(
            data,
            hive_partitioning=True,
            _deletion_files=("iceberg", ({0: [deletes[0]], 2: [deletes[1]]}, {})),
        ).sort("a"),
        engine=engine,
    )


def test_scan_position_deletes_count(
    engine: pl.GPUEngine, position_deletes: tuple[str, tuple[str, ...]]
) -> None:
    data, deletes = position_deletes
    assert_gpu_result_equal(
        pl.scan_parquet(
            data,
            hive_partitioning=False,
            _deletion_files=("iceberg", ({0: [deletes[0]], 2: [deletes[1]]}, {})),
        ).select(pl.len()),
        engine=engine,
    )
