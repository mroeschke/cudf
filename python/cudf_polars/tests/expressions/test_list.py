# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("inner", [pl.Int64, pl.Float64, pl.String])
def test_list_drop_nulls(engine: pl.GPUEngine, inner: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, None, 2], [None], [], None]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", ["list", "struct"])
def test_list_drop_nulls_nested(engine: pl.GPUEngine, inner: str) -> None:
    data: list[object]
    if inner == "list":
        data = [[[1], None, [], [None]], None, [], [None]]
    else:
        data = [[{"x": 1}, None, {"x": None}], None, [], [None]]
    lf = pl.LazyFrame({"a": data})
    q = lf.select(pl.col("a").list.drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


def test_list_drop_nulls_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String])
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_drop_nulls_sliced(
    engine: pl.GPUEngine, inner: pl.DataType, how: str, *, with_nulls: bool
) -> None:
    null_list = None if with_nulls else [5, None]
    a = pl.Series([[3, None, 4], null_list, [None, 2], [1, None]]).cast(pl.List(inner))
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": a}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


def test_list_drop_nulls_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_drop_nulls_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.drop_nulls())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_drop_nulls_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([1, None, 2]).list.drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "data",
    [
        [[1, None, 2, None], [None, None], [1, 2], None],
        [[None, 1, None, 2], [None], [3, 4]],
        [None, ["value"]],
        [None, ["value", None]],
        [None, [None, "value", None]],
        [[1.0, float("nan"), None], [None, float("nan")], None],
        [[1, 2], [], None, [3]],
    ],
    ids=[
        "eager_series",
        "eager_frame",
        "lazy_no_null_elements",
        "lazy_trailing_null",
        "lazy_surrounding_nulls",
        "keeps_nan",
        "no_null_elements",
    ],
)
def test_list_drop_nulls_polars_cases(engine: pl.GPUEngine, data: list) -> None:
    lf = pl.LazyFrame({"a": data})
    q = lf.with_columns(pl.col("a").list.drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
