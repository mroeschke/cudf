# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String])
@pytest.mark.parametrize("how", ["last", "get", "float_index"])
def test_list_last(engine: pl.GPUEngine, inner: pl.DataType, how: str) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]]).cast(pl.List(inner))}
    )
    a = pl.col("a").list
    expr = {
        "last": a.last(),
        "get": a.get(-1, null_on_oob=True),
        "float_index": a.get(pl.lit(-1.0), null_on_oob=True),
    }[how]
    q = lf.select(expr)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.UInt8,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
        pl.Date,
    ],
)
def test_list_last_dtypes(engine: pl.GPUEngine, dtype: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1], [1, 0, 1], [0, None], None, []]).cast(pl.List(dtype))}
    )
    q = lf.select(pl.col("a").list.last())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", ["list", "struct"])
def test_list_last_nested(engine: pl.GPUEngine, inner: str) -> None:
    data: list
    if inner == "list":
        data = [[[1], [2, 3]], None, [], [[4], None]]
    else:
        data = [
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
            [{"a": 1, "b": 2}, None],
            [],
            None,
        ]
    lf = pl.LazyFrame({"a": data})
    q = lf.select(pl.col("a").list.last())
    assert_gpu_result_equal(q, engine=engine)


def test_list_last_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.String))})
    q = lf.select(pl.col("a").list.last())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String])
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_last_sliced(
    engine: pl.GPUEngine, inner: pl.DataType, how: str, *, with_nulls: bool
) -> None:
    null_list = None if with_nulls else [6, 7]
    a = pl.Series([[5], [4, None], null_list, [None, 3], null_list]).cast(
        pl.List(inner)
    )
    lf = pl.LazyFrame({"k": [4, 3, 2, 1, 0], "a": a}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.last())
    assert_gpu_result_equal(q, engine=engine)


def test_list_last_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.last())
    assert_gpu_result_equal(q, engine=engine)


def test_list_last_of_aggregated_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"g": [1, 1, 1, 2, 2, 3], "x": [1, 2, 3, 4, None, 6]})
    q = (
        lf.group_by("g", maintain_order=True)
        .agg(pl.col("x"))
        .select("g", last=pl.col("x").list.last())
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_last_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([1, 2]).list.last())
    assert_gpu_result_equal(q, engine=engine)


def test_list_last_literal_series(engine: pl.GPUEngine) -> None:
    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    q = pl.LazyFrame().select(pl.lit(a).list.last())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "how", ["first", "positive_index", "column_index", "null_index", "raise_on_oob"]
)
def test_list_last_get_not_supported(in_memory_engine: pl.GPUEngine, how: str) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None], "i": [-1, -1]})
    a = pl.col("a").list
    expr = {
        "first": a.first(),
        "positive_index": a.get(1, null_on_oob=True),
        "column_index": a.get(pl.col("i"), null_on_oob=True),
        "null_index": a.get(pl.lit(None, dtype=pl.Int64), null_on_oob=True),
        "raise_on_oob": a.get(-1, null_on_oob=False),
    }[how]
    q = lf.select(expr)
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_last_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.last())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
