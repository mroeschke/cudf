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
    "inner", [pl.Int64, pl.String, pl.Date, pl.Datetime("us")], ids=str
)
def test_list_concat(engine: pl.GPUEngine, inner: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2, 2], [None, 3], [], None, [5], None]).cast(
                pl.List(inner)
            ),
            "b": pl.Series([[2], [3], [4], [5], None, None]).cast(pl.List(inner)),
        }
    )
    q = lf.select(pl.col("a").list.concat(pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_three_inputs(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "b": [[2], [], [4], [5], [6]],
            "c": [[], [None], None, [7], [8, 9]],
        }
    )
    q = lf.select(pl.col("a").list.concat([pl.col("b"), pl.col("c")]))
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_concat_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"a": [[1, 2, 2], [None, 3], [], None, [5]], "b": [[2], [3], [4], [5], None]}
    )
    q = lf.select(pl.concat_list("a", "b"), pl.concat_list("a").alias("c"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("literal_first", [False, True])
@pytest.mark.parametrize("kind", ["list", "empty", "null", "series"])
def test_list_concat_literal(
    engine: pl.GPUEngine, kind: str, *, literal_first: bool
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    values = {"list": [7, None], "empty": [], "null": None}
    value = pl.Series([[1, 2]]) if kind == "series" else values[kind]
    literal = pl.lit(value, dtype=pl.List(pl.Int64))
    exprs = [literal, pl.col("a")] if literal_first else [pl.col("a"), literal]
    q = lf.select(pl.concat_list(exprs).alias("a"))
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_all_literals(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.lit([1, 2]).list.concat(pl.lit([3])))
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_nested_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[[1], [2, 3]], None, [], [None], [[]]],
            "b": [[[4]], [[5]], None, [[6], None], []],
        }
    )
    q = lf.select(pl.col("a").list.concat("b"))
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_struct(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[{"a": 1, "b": 2}, {"a": 3, "b": 4}], [{"a": 1, "b": 2}], None],
            "b": [[{"a": 6, "b": 7}, None], [{"a": 6, "b": None}], [{"a": 8, "b": 9}]],
        }
    )
    q = lf.select(pl.col("a").list.concat("b"))
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([], dtype=pl.List(pl.Int64)),
            "b": pl.Series([], dtype=pl.List(pl.Int64)),
        }
    )
    q = lf.select(pl.col("a").list.concat([pl.col("b"), pl.lit([1])]))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String])
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_concat_sliced(
    engine: pl.GPUEngine, inner: pl.DataType, how: str, *, with_nulls: bool
) -> None:
    null_list = None if with_nulls else [6]
    lf = pl.LazyFrame(
        {
            "k": [3, 2, 1, 0],
            "a": pl.Series([[3, 4], null_list, [2], [1]]).cast(pl.List(inner)),
            "b": pl.Series([[30], [40], null_list, [10]]).cast(pl.List(inner)),
            "c": pl.Series([[7], [8], [9], []]).cast(pl.List(inner)),
        }
    ).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(
        pl.col("a").list.concat(pl.col("b")),
        pl.col("c").list.concat("a"),
        pl.col("a").alias("a2"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_sliced_literal(engine: pl.GPUEngine, *, with_nulls: bool) -> None:
    null_list = None if with_nulls else [6]
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": [[3, 4], null_list, [2], [1]]})
    q = lf.sort("k").slice(1, 3).select(pl.col("a").list.concat(pl.lit([9])))
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_when_then(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [["123", "456"], None], "b": [["789"], ["zzz"]]})
    q = lf.with_columns(
        pl.when(pl.col("a").is_null())
        .then(pl.col("a").list.concat(pl.col("b")))
        .otherwise(pl.col("b"))
        .alias("x"),
        pl.when(pl.col("a").is_null())
        .then(pl.col("b"))
        .otherwise(pl.col("a").list.concat(pl.col("b")))
        .alias("y"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_concat_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "k": [1, 1, 2, 2, 3],
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "b": [[2], [3], [4], [5], None],
        }
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.concat("b"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "other",
    [pl.col("c"), pl.lit(1), pl.lit(None)],
    ids=["column", "scalar", "null"],
)
def test_list_concat_non_list_not_supported(
    in_memory_engine: pl.GPUEngine, other: pl.Expr
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None], "c": [1, 2]})
    q = lf.select(pl.concat_list(pl.col("a"), other))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("other", ["b", "s", "e"])
def test_list_concat_different_dtypes_not_supported(
    in_memory_engine: pl.GPUEngine, other: str
) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], None], dtype=pl.List(pl.Int64)),
            "b": pl.Series([[3], [4]], dtype=pl.List(pl.Int32)),
            "s": [["x"], None],
            "e": [[], []],
        }
    )
    q = lf.select(pl.col("a").list.concat(pl.col(other)))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_concat_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.concat(pl.col("a").implode()))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_concat_implode_in_rolling_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"t": [1, 2, 3], "a": values})
    q = lf.rolling("t", period="2i").agg(
        pl.col("a").implode().list.concat(pl.col("a").implode())
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
