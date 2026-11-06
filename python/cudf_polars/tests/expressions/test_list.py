# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(params=[False, True], ids=["ascending", "descending"])
def descending(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[False, True], ids=["nulls_first", "nulls_last"])
def nulls_last(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.mark.parametrize(
    "inner", [pl.Int64, pl.UInt64, pl.Float64, pl.Boolean, pl.String]
)
def test_list_sort(
    engine: pl.GPUEngine, inner: pl.DataType, *, descending: bool, nulls_last: bool
) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.sort(descending=descending, nulls_last=nulls_last))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Float32, pl.Float64])
def test_list_sort_float_nan(
    engine: pl.GPUEngine, inner: pl.DataType, *, descending: bool, nulls_last: bool
) -> None:
    nan = float("nan")
    a = pl.Series(
        [
            [1.0, nan, -1.0, None, float("inf"), float("-inf"), 0.0],
            [nan, None, nan],
            [],
            None,
            [-0.0, 2.0],
        ],
        dtype=pl.List(inner),
    )
    lf = pl.LazyFrame({"a": a})
    q = lf.select(pl.col("a").list.sort(descending=descending, nulls_last=nulls_last))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "inner",
    [pl.Decimal(10, 2), pl.Date, pl.Datetime("us"), pl.Duration("ms")],
    ids=str,
)
def test_list_sort_decimal_temporal(engine: pl.GPUEngine, inner: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[3, 1, 2], [None, 3], [], None, [5]], dtype=pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.sort())
    assert_gpu_result_equal(q, engine=engine)


def test_list_sort_nested_list(
    engine: pl.GPUEngine, *, descending: bool, nulls_last: bool
) -> None:
    lf = pl.LazyFrame(
        {"a": [[[2, 1], [1], None, [], [1, None], [None]], None, [], [[3], [3]]]}
    )
    q = lf.select(pl.col("a").list.sort(descending=descending, nulls_last=nulls_last))
    assert_gpu_result_equal(q, engine=engine)


def test_list_sort_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.sort())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String, pl.List(pl.Int64)], ids=str)
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_sort_sliced(
    engine: pl.GPUEngine,
    inner: pl.DataType,
    how: str,
    *,
    with_nulls: bool,
) -> None:
    rows = [[3, None, 4], None if with_nulls else [6, 5], [2, None, 1], [1, 0]]
    if inner == pl.List(pl.Int64):
        a = pl.Series([None if r is None else [[x] for x in r] for r in rows])
    else:
        a = pl.Series(rows).cast(pl.List(inner))
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": a}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.sort(nulls_last=True))
    assert_gpu_result_equal(q, engine=engine)


def test_list_sort_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(
        pl.col("a").list.sort(descending=True)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_sort_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([3, None, 1]).list.sort())
    assert_gpu_result_equal(q, engine=engine)


def test_list_sort_ordering(engine: pl.GPUEngine, *, nulls_last: bool) -> None:
    lf = pl.LazyFrame({"a": [[2, 1], [1, 3, 2], [None, 1, 2], [-1, None, 9]]})
    q = lf.select(pl.col("a").list.sort(nulls_last=nulls_last))
    assert_gpu_result_equal(q, engine=engine)


def test_list_sort_bool_nulls(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"a": [[], [None], [None, None, False], [None, True, None, False, True]]}
    )
    q = lf.select(pl.col("a").list.sort())
    assert_gpu_result_equal(q, engine=engine)


def test_list_sort_after_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"k": [0, 1, 0, 1, 0, 1], "v": [2, 1, 0, None, 1, 1]})
    q = (
        lf.group_by("k")
        .agg(pl.col("v"))
        .sort("k")
        .with_columns(pl.col("v").list.sort())
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("nesting", ["struct", "list_of_struct", "struct_of_list"])
def test_list_sort_struct_not_supported(
    in_memory_engine: pl.GPUEngine, nesting: str
) -> None:
    data: list[object] = [[{"x": 1}, {"x": None}], None]
    if nesting == "list_of_struct":
        data = [[row] for row in data]
    elif nesting == "struct_of_list":
        data = [[{"x": [1]}, {"x": None}], None]
    lf = pl.LazyFrame({"a": data})
    q = lf.select(pl.col("a").list.sort())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_sort_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.sort())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_sort_implode_in_rolling_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"t": [1, 2, 3], "a": values})
    q = lf.rolling("t", period="2i").agg(pl.col("a").implode().list.sort())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
