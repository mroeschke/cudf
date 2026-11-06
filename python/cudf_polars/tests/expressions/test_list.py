# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize(
    "inner", [pl.Int8, pl.Int64, pl.UInt64, pl.Float64, pl.Boolean, pl.String]
)
def test_list_first(engine: pl.GPUEngine, inner: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.first())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Decimal(10, 2), pl.Datetime("us")])
def test_list_first_decimal_datetime(engine: pl.GPUEngine, inner: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]], dtype=pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.first())
    assert_gpu_result_equal(q, engine=engine)


def test_list_first_nested_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[[1], [2, 3]], None, [], [None], [[]]]})
    q = lf.select(pl.col("a").list.first())
    assert_gpu_result_equal(q, engine=engine)


def test_list_first_struct(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
                [{"a": None, "b": 6}],
                [],
                None,
                [None, {"a": 7, "b": 8}],
            ]
        }
    )
    q = lf.select(pl.col("a").list.first())
    assert_gpu_result_equal(q, engine=engine)


def test_list_first_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.first())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String, pl.List(pl.Int64)], ids=str)
def test_list_first_sliced(
    engine: pl.GPUEngine, inner: pl.DataType, *, with_nulls: bool
) -> None:
    rows = [[3, 4], None if with_nulls else [6], [2], [1]]
    if inner == pl.List(pl.Int64):
        a = pl.Series([None if r is None else [[x] for x in r] for r in rows])
    else:
        a = pl.Series(rows).cast(pl.List(inner))
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": a})
    q = lf.sort("k").slice(1, 3).select(pl.col("a").list.first())
    assert_gpu_result_equal(q, engine=engine)


def test_list_first_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.first())
    assert_gpu_result_equal(q, engine=engine)


def test_list_first_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([4, 5]).list.first())
    assert_gpu_result_equal(q, engine=engine)


def test_list_first_literal_series(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(
        pl.lit(pl.Series("b", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])).list.first()
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_first_when_then(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"cars": [[1, 2, 3], [2, 3], [4], []]})
    q = lf.select(
        pl.col("cars").list.first().alias("cars_first"),
        pl.when(pl.col("cars").list.first() == 2).then(1).otherwise(3).alias("x"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "index,index_dtype",
    [(0, None), (0, pl.UInt8), (0, pl.Int16), (0.0, None), (False, None)],
)
def test_list_first_get_zero(
    engine: pl.GPUEngine, index: float, index_dtype: pl.DataType | None
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(
        pl.col("a").list.get(pl.lit(index, dtype=index_dtype), null_on_oob=True)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "index,null_on_oob",
    [(0, False), (1, True), (-1, True), (True, True), (None, True), ("i", True)],
)
def test_list_first_get_not_supported(
    in_memory_engine: pl.GPUEngine, index: int | str | None, *, null_on_oob: bool
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None], "i": [0, 1]})
    index_expr = pl.col(index) if isinstance(index, str) else pl.lit(index)
    q = lf.select(pl.col("a").list.get(index_expr, null_on_oob=null_on_oob))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_first_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.first())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_first_implode_in_rolling_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"t": [1, 2, 3], "a": values})
    q = lf.rolling("t", period="2i").agg(pl.col("a").implode().list.first())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
