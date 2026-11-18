# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("null_behavior", ["ignore", "drop"])
@pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
def test_list_diff(
    engine: pl.GPUEngine, n: int, null_behavior: Literal["ignore", "drop"]
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5], [1, 4, 9, 16]]})
    q = lf.select(pl.col("a").list.diff(n, null_behavior=null_behavior))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [3, -3, 5, 2**40, -(2**40)])
def test_list_diff_n_out_of_bounds(engine: pl.GPUEngine, n: int) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5], [1, 4, 9, 16]]})
    q = lf.select(pl.col("a").list.diff(n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ],
)
@pytest.mark.parametrize("null_behavior", ["ignore", "drop"])
def test_list_diff_integer_overflow(
    engine: pl.GPUEngine, dtype: pl.DataType, null_behavior: Literal["ignore", "drop"]
) -> None:
    info = np.iinfo(str(dtype).lower())
    values = [
        int(info.min),
        int(info.max),
        int(info.min),
        None,
        int(info.max) // 2,
        int(info.max) // 2 + 1,
        0,
    ]
    lf = pl.LazyFrame({"a": pl.Series([values, None, [1]], dtype=pl.List(dtype))})
    q = lf.select(pl.col("a").list.diff(1, null_behavior=null_behavior))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_list_diff_float(engine: pl.GPUEngine, dtype: pl.DataType) -> None:
    values = [1.5, float("nan"), float("inf"), float("inf"), None, -float("inf"), 0.0]
    lf = pl.LazyFrame({"a": pl.Series([values, [], None], dtype=pl.List(dtype))})
    q = lf.select(pl.col("a").list.diff(-1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("time_zone", [None, "UTC"])
@pytest.mark.parametrize("n", [1, -2])
def test_list_diff_datetime(
    engine: pl.GPUEngine,
    time_unit: Literal["ms", "us", "ns"],
    time_zone: str | None,
    n: int,
) -> None:
    values = [0, 5, None, -(2**63) + 1, 2**63 - 1, 2]
    lf = pl.LazyFrame(
        {
            "a": pl.Series([values, None, []], dtype=pl.List(pl.Int64)).cast(
                pl.List(pl.Datetime(time_unit, time_zone))
            )
        }
    )
    q = lf.select(pl.col("a").list.diff(n))
    assert_gpu_result_equal(q, engine=engine)


def test_list_diff_duration(
    engine: pl.GPUEngine, time_unit: Literal["ms", "us", "ns"]
) -> None:
    values = [0, 5, None, -(2**63) + 1, 2**63 - 1, 2]
    lf = pl.LazyFrame(
        {
            "a": pl.Series([values, None, []], dtype=pl.List(pl.Int64)).cast(
                pl.List(pl.Duration(time_unit))
            )
        }
    )
    q = lf.select(pl.col("a").list.diff(1, null_behavior="drop"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("null_behavior", ["ignore", "drop"])
def test_list_diff_date(
    engine: pl.GPUEngine, null_behavior: Literal["ignore", "drop"]
) -> None:
    days = [
        0,
        2,
        None,
        106_751_991,
        -106_751_991,
        106_751_992,
        0,
        -(2**31),
        10**8,
        -(10**8),
    ]
    lf = pl.LazyFrame(
        {
            "a": pl.Series([days, None, [1]], dtype=pl.List(pl.Int32)).cast(
                pl.List(pl.Date)
            )
        }
    )
    q = lf.select(pl.col("a").list.diff(1, null_behavior=null_behavior))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("null_behavior", ["ignore", "drop"])
def test_list_diff_empty(
    engine: pl.GPUEngine, null_behavior: Literal["ignore", "drop"]
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.diff(1, null_behavior=null_behavior))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[], [None, None]])
def test_list_diff_without_lists_follows_schema(
    engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"a": pl.Series(values, dtype=pl.List(pl.UInt64))})
    q = lf.select(pl.col("a").list.diff())
    assert_gpu_result_equal(q, engine=engine, check_dtypes=False)
    assert q.collect(engine=engine).schema == q.collect_schema()


@pytest.mark.parametrize("null_behavior", ["ignore", "drop"])
@pytest.mark.parametrize("n", [1, -1])
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_diff_sliced(
    engine: pl.GPUEngine,
    n: int,
    null_behavior: Literal["ignore", "drop"],
    how: str,
    *,
    with_nulls: bool,
) -> None:
    null_list = None if with_nulls else [6, 7, 9]
    a = pl.Series([[3, 4, 8], null_list, [2, None, 5], [1, 7]])
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": a}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.diff(n, null_behavior=null_behavior))
    assert_gpu_result_equal(q, engine=engine)


def test_list_diff_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.diff())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_diff_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.diff())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_diff_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([1, 4, None, 9]).list.diff())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Boolean, pl.String, pl.Null, pl.Decimal(10, 2), pl.List(pl.Int64)]
)
def test_list_diff_dtype_not_supported(
    in_memory_engine: pl.GPUEngine, dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([[None, None], None], dtype=pl.List(dtype))})
    q = lf.select(pl.col("a").list.diff())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("n", [3, -3])
def test_list_diff_drop_large_n_not_supported(
    in_memory_engine: pl.GPUEngine, n: int
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 4, 7], [1, 2]]})
    q = lf.select(pl.col("a").list.diff(n, null_behavior="drop"))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "inner_dtype,data",
    [
        (pl.Int64, [[1, 2], [10, 2, 1]]),
        (pl.Datetime("us"), [[1, 2]]),
        (pl.Date, [[1, 2]]),
        (pl.UInt64, [[1, 2]]),
        (pl.UInt32, [[1, 2]]),
        (pl.UInt8, [[1, 2]]),
        (pl.Int8, [[1, 2]]),
        (pl.Float32, [[1, 2]]),
    ],
    ids=[
        "values",
        "schema_datetime",
        "schema_date",
        "schema_uint64",
        "schema_uint32",
        "schema_uint8",
        "schema_int8",
        "schema_float32",
    ],
)
def test_list_diff_polars_cases(
    engine: pl.GPUEngine, inner_dtype: pl.DataType, data: list
) -> None:
    lf = pl.LazyFrame({"a": pl.Series(data).cast(pl.List(inner_dtype))})
    q = lf.select(pl.col("a").list.diff(1))
    assert_gpu_result_equal(q, engine=engine)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
