# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("item", ["literal", "null", "column", "scalar", "null_scalar"])
@pytest.mark.parametrize("nulls_equal", [True, False])
def test_list_contains(engine: pl.GPUEngine, item: str, *, nulls_equal: bool) -> None:
    lf = pl.LazyFrame(
        {"a": [[1, 2, 2], [None, 3], [], None, [5]], "x": [2, None, 3, 4, 5]}
    )
    needle = {
        "literal": pl.lit(2),
        "null": pl.lit(None),
        "column": pl.col("x"),
        "scalar": pl.col("x").last(),
        "null_scalar": pl.col("x").get(1),
    }[item]
    q = lf.select(pl.col("a").list.contains(needle, nulls_equal=nulls_equal))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("item", ["literal", "column"])
def test_list_contains_string(engine: pl.GPUEngine, item: str) -> None:
    lf = pl.LazyFrame(
        {
            "a": [["a", "b"], [None, "c"], [], None, ["c"]],
            "x": ["b", None, "c", "a", "d"],
        }
    )
    needle = pl.lit("c") if item == "literal" else pl.col("x")
    q = lf.select(pl.col("a").list.contains(needle))
    assert_gpu_result_equal(q, engine=engine)


def test_list_contains_float(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[float("nan")], [0.0], [-0.0], None, [1.0]],
            "x": [float("nan"), -0.0, 0.0, 1.0, float("nan")],
        }
    )
    q = lf.select(
        pl.col("a").list.contains(pl.col("x")).alias("column"),
        pl.col("a").list.contains(float("nan")).alias("nan"),
        pl.col("a").list.contains(-0.0).alias("zero"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", ["list", "struct"])
def test_list_contains_nested(engine: pl.GPUEngine, inner: str) -> None:
    a: list[object]
    x: list[object]
    if inner == "list":
        a = [[[1], [2, None]], [None, [3]], [], None, [[2, None]]]
        x = [[2, None], None, [1], [1], [2]]
    else:
        a = [[{"f": 1}, {"f": None}], [None, {"f": 3}], [], None, [{"f": 2}]]
        x = [{"f": None}, None, {"f": 1}, {"f": 1}, {"f": 2}]
    lf = pl.LazyFrame({"a": a, "x": x})
    q = lf.select(pl.col("a").list.contains(pl.col("x")))
    assert_gpu_result_equal(q, engine=engine)


def test_list_contains_widened_item(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], [3], None], dtype=pl.List(pl.Int64)),
            "x": pl.Series([1, 2, 3], dtype=pl.Int8),
        }
    )
    q = lf.select(pl.col("a").list.contains(pl.col("x")))
    assert_gpu_result_equal(q, engine=engine)


def test_list_contains_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([], dtype=pl.List(pl.Int64)),
            "x": pl.Series([], dtype=pl.Int64),
        }
    )
    q = lf.select(
        pl.col("a").list.contains(pl.col("x")).alias("column"),
        pl.col("a").list.contains(None, nulls_equal=False).alias("null"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("item", ["literal", "null", "column"])
@pytest.mark.parametrize("how", ["slice", "tail"])
@pytest.mark.parametrize("nulls_equal", [True, False])
def test_list_contains_sliced(
    engine: pl.GPUEngine, item: str, how: str, *, nulls_equal: bool, with_nulls: bool
) -> None:
    null_list = None if with_nulls else [2]
    lf = pl.LazyFrame(
        {
            "k": [4, 3, 2, 1, 0],
            "a": [[3, None], [2], null_list, [2, None], [1]],
            "x": [None, 2, 1, None, 1],
        }
    ).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    needle = {"literal": pl.lit(2), "null": pl.lit(None), "column": pl.col("x")}[item]
    q = lf.select(pl.col("a").list.contains(needle, nulls_equal=nulls_equal))
    assert_gpu_result_equal(q, engine=engine)


def test_list_contains_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.contains(2))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_contains_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.contains(pl.lit(None)))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("item", ["literal", "column"])
def test_list_contains_literal(engine: pl.GPUEngine, item: str) -> None:
    lf = pl.LazyFrame({"x": [1, None, 4]})
    needle = pl.lit(4) if item == "literal" else pl.col("x")
    q = lf.select(pl.lit([1, None, 4]).list.contains(needle))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("item", ["literal", "column"])
@pytest.mark.parametrize("dtypes", ["decimal_scale", "time_zone"])
def test_list_contains_mismatched_dtype_not_supported(
    in_memory_engine: pl.GPUEngine, item: str, dtypes: str
) -> None:
    if dtypes == "decimal_scale":
        value: object = Decimal("1.50")
        item_dtype: pl.DataType = pl.Decimal(3, 2)
        inner: pl.DataType = pl.Decimal(3, 1)
    else:
        value = datetime(2020, 1, 1)
        item_dtype = pl.Datetime("us", "UTC")
        inner = pl.Datetime("us", "America/New_York")
    lf = pl.LazyFrame(
        {
            "a": pl.Series([[value], None], dtype=pl.List(inner)),
            "x": pl.Series([value, value], dtype=item_dtype),
        }
    )
    needle = pl.lit(value, item_dtype) if item == "literal" else pl.col("x")
    q = lf.select(pl.col("a").list.contains(needle))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
