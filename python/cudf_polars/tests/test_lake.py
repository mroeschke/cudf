# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``polars.scan_iceberg`` and ``polars.scan_delta``."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    first = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = catalog.create_table("ns.flat", schema=first.to_arrow().schema)
    first.lazy().sink_iceberg(tbl, mode="append")
    pl.LazyFrame({"a": [4, 5], "b": ["p", "q"]}).sink_iceberg(tbl, mode="append")
    return tbl.metadata_location


@pytest.fixture
def evolved_iceberg(tmp_path: Path) -> str:
    """
    An Iceberg table whose data files disagree on names, types and columns.

    Sinking a wider frame with ``schema_mode="merge"`` promotes ``a`` to
    Int64 and adds ``c``, so the first data file ends up disagreeing with
    the table schema on both, as well as on the name of field 2.
    """
    catalog = iceberg_catalog(tmp_path)
    first = pl.DataFrame(
        {"a": pl.Series([1, 2, 3], dtype=pl.Int32), "b": ["x", "y", "z"]}
    )
    tbl = catalog.create_table("ns.evolved", schema=first.to_arrow().schema)
    first.lazy().sink_iceberg(tbl, mode="append")

    with tbl.update_schema() as update:
        update.rename_column("b", "b_renamed")
    tbl.refresh()

    pl.LazyFrame(
        {
            "a": pl.Series([4, 5], dtype=pl.Int64),
            "b_renamed": ["p", "q"],
            "c": [1.5, 2.5],
        }
    ).sink_iceberg(tbl, mode="append", schema_mode="merge")
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
    pl.LazyFrame({"a": [1, 2, 3, 4], "part": ["u", "u", "v", "v"]}).sink_iceberg(
        tbl, mode="append"
    )
    return tbl.metadata_location


@pytest.fixture
def flat_delta(tmp_path: Path) -> str:
    """A Delta table of two data files."""
    path = tmp_path / "delta"
    pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).sink_delta(
        str(path), mode="overwrite"
    )
    pl.LazyFrame({"a": [4, 5], "b": ["p", "q"]}).sink_delta(str(path), mode="append")
    return str(path)


@pytest.fixture
def deleted_rows_delta(tmp_path: Path) -> str:
    """
    A Delta table some of whose rows have been deleted.

    Deletion vectors are requested but delta-rs does not write them, so it
    rewrites the data files instead. The scan of a real deletion vector is
    covered by polars' own ``test_delta_deletion_vector.py``, which hand
    writes one.
    """
    path = tmp_path / "delta_deleted"
    pl.LazyFrame({"a": list(range(10)), "b": [f"s{i}" for i in range(10)]}).sink_delta(
        str(path),
        mode="overwrite",
        delta_write_options={"configuration": {"delta.enableDeletionVectors": "true"}},
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


def test_scan_delta_deleted_rows(engine: pl.GPUEngine, deleted_rows_delta: str) -> None:
    assert_gpu_result_equal(pl.scan_delta(deleted_rows_delta).sort("a"), engine=engine)


def test_scan_delta_deleted_rows_count(
    engine: pl.GPUEngine, deleted_rows_delta: str
) -> None:
    assert_gpu_result_equal(
        pl.scan_delta(deleted_rows_delta).select(pl.len()), engine=engine
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
