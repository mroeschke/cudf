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


@pytest.mark.parametrize("scalar_first", [False, True])
@pytest.mark.parametrize(
    "scalar,other", [("i", "a"), ("j", "a"), ("s", "as"), ("st", "ast")]
)
def test_concat_list_scalar_and_list(
    engine: pl.GPUEngine, scalar: str, other: str, *, scalar_first: bool
) -> None:
    lf = pl.LazyFrame(
        {
            "i": pl.Series([1, None, 3, 4, 5], dtype=pl.Int32),
            "j": [10, 20, None, 40, 50],
            "s": ["x", None, "z", "w", "v"],
            "st": [{"x": 1}, None, {"x": None}, {"x": 4}, {"x": 5}],
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "as": [["1", "2", "2"], [None, "3"], [], None, ["5"]],
            "ast": [[{"x": 1}, None], [{"x": None}], [], None, [{"x": 5}]],
        }
    )
    exprs = [scalar, other]
    q = lf.select(pl.concat_list(exprs[::-1] if scalar_first else exprs).alias("a"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "left,right", [("i", "j"), ("i", "u"), ("f", "d"), ("s", "s"), ("j", "j")]
)
def test_concat_list_scalars(engine: pl.GPUEngine, left: str, right: str) -> None:
    lf = pl.LazyFrame(
        {
            "i": pl.Series([1, None, 3], dtype=pl.Int32),
            "u": pl.Series([None, 2, 3], dtype=pl.UInt32),
            "j": [10, 20, None],
            "f": pl.Series([1.5, None, 3.0], dtype=pl.Float32),
            "d": [0.1, 0.2, None],
            "s": ["x", None, "z"],
        }
    )
    q = lf.select(pl.concat_list(left, right), pl.concat_list(right).alias("c"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("literal_first", [False, True])
@pytest.mark.parametrize("value", [1, None])
@pytest.mark.parametrize("column", ["j", "a"])
def test_concat_list_scalar_literal(
    engine: pl.GPUEngine, column: str, value: int | None, *, literal_first: bool
) -> None:
    lf = pl.LazyFrame(
        {"j": [10, 20, None, 40, 50], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    exprs = [pl.lit(value), pl.col(column)]
    q = lf.select(pl.concat_list(exprs if literal_first else exprs[::-1]).alias("a"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("column", ["s", "st", "ll"])
def test_concat_list_null_literal_nested(engine: pl.GPUEngine, column: str) -> None:
    lf = pl.LazyFrame(
        {
            "s": ["x", None, "z"],
            "st": [{"x": [1]}, None, {"x": None}],
            "ll": [[[1]], None, [[2], None]],
        }
    )
    q = lf.select(pl.concat_list(pl.lit(None), column))
    assert_gpu_result_equal(q, engine=engine)


def test_concat_list_all_scalar_literals(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(
        pl.concat_list(pl.lit(1), pl.lit(2)),
        pl.concat_list(pl.lit(None), pl.lit(None, dtype=pl.Int64)).alias("b"),
        pl.concat_list(pl.lit(1, dtype=pl.Int8), pl.lit([2, 3])).alias("c"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_concat_list_scalar_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "i": pl.Series([], dtype=pl.Int32),
            "a": pl.Series([], dtype=pl.List(pl.Int64)),
        }
    )
    q = lf.select(pl.concat_list("i", "a", pl.lit(None), pl.lit(1)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("how", ["slice", "tail"])
def test_concat_list_scalar_sliced(
    engine: pl.GPUEngine, how: str, *, with_nulls: bool
) -> None:
    null_value = None if with_nulls else 6
    lf = pl.LazyFrame(
        {
            "k": [3, 2, 1, 0],
            "i": pl.Series([3, null_value, 2, 1], dtype=pl.Int32),
            "s": ["c", "b", None if with_nulls else "n", "a"],
            "a": [[30], [40], None if with_nulls else [6], [10]],
            "st": [{"x": 3}, {"x": 2}, None if with_nulls else {"x": 6}, {"x": 1}],
        }
    ).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(
        pl.concat_list("i", "a"),
        pl.concat_list("s", "s").alias("s2"),
        pl.concat_list("i", pl.lit(9)).alias("i2"),
        pl.concat_list("st", "st").alias("st2"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_concat_list_scalar_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "i": [1, None, 3, 4, 5], "a": [[1], None, [], [2], [3]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(
        pl.concat_list("i", "a"), pl.concat_list("i", "i").alias("b")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "left,right",
    [
        ("j", "f"),
        ("f", "float_lit"),
        ("b", "a"),
        ("j", "float_lit"),
        ("i8", "int_lit"),
    ],
)
def test_concat_list_scalar_different_dtypes_not_supported(
    in_memory_engine: pl.GPUEngine, left: str, right: str
) -> None:
    lf = pl.LazyFrame(
        {
            "j": [1, 2],
            "f": pl.Series([1.5, None], dtype=pl.Float32),
            "b": [True, None],
            "a": [[1], None],
            "i8": pl.Series([1, None], dtype=pl.Int8),
        }
    )
    literals = {"float_lit": pl.lit(2.5), "int_lit": pl.lit(1)}
    other = literals.get(right, pl.col(right))
    q = lf.select(pl.concat_list(pl.col(left), other))
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
