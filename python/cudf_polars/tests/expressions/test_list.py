# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.list import SetOperation
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize(
    "inner",
    [
        pl.Int64,
        pl.UInt8,
        pl.Float32,
        pl.String,
        pl.Date,
        pl.Duration("ms"),
        pl.Decimal(10, 2),
    ],
)
def test_list_set_union(engine: pl.GPUEngine, inner: pl.DataType) -> None:
    dtype = pl.List(inner)
    a = pl.Series([[3, 1, 1, None], [], None, [1], None, [2, 2], [1, 2, 2]])
    b = pl.Series([[1, 4, None], [], [1], None, None, [], [None, 3]])
    lf = pl.LazyFrame({"a": a.cast(dtype), "b": b.cast(dtype)})
    q = lf.select(pl.col("a").list.set_union(pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_union_float(engine: pl.GPUEngine) -> None:
    nan = float("nan")
    lf = pl.LazyFrame(
        {
            "a": [[nan, 1.0, -0.0], [0.0, nan], [None, nan], [-0.0, 0.0]],
            "b": [[nan, 0.0], [-0.0, -nan], [nan, None], [0.0]],
        }
    )
    q = lf.select(pl.col("a").list.set_union(pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_union_multiple_columns(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 3], [1, 1, 1], [4]],
            "b": [[4, 2, 1], [2, 1, 12], [4]],
            "c": [[1, 2], [2, 1, 76], [8, 9]],
        }
    )
    q = lf.select(pl.col("a", "b").list.set_union("c"))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_union_empty(engine: pl.GPUEngine) -> None:
    dtype = pl.List(pl.String)
    lf = pl.LazyFrame(
        {"a": pl.Series([], dtype=dtype), "b": pl.Series([], dtype=dtype)}
    )
    q = lf.select(pl.col("a").list.set_union(pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("how", ["slice", "tail"])
@pytest.mark.parametrize("swap", [False, True])
def test_list_set_union_sliced(
    engine: pl.GPUEngine, how: str, *, swap: bool, with_nulls: bool
) -> None:
    null_list = None if with_nulls else [8]
    lf = pl.LazyFrame(
        {
            "k": [3, 2, 1, 0],
            "a": [[3, 4], null_list, [2], [1]],
            "b": [[5, 3], [6], null_list, [1, 7]],
        }
    ).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    left, right = ("b", "a") if swap else ("a", "b")
    q = lf.select(pl.col(left).list.set_union(pl.col(right)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("other", [[], [1, 2], [1, None]])
@pytest.mark.parametrize("swap", [False, True])
def test_list_set_union_literal(
    engine: pl.GPUEngine, other: list[int | None], *, swap: bool
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 3], [], [None, 3], None, [5, 6, 7]]})
    column, literal = pl.col("a"), pl.lit(pl.Series([other], dtype=pl.List(pl.Int64)))
    q = lf.select(
        literal.list.set_union(column) if swap else column.list.set_union(literal)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_union_literal_python_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[2, 3, 3], [3, 1], None, [1, 2, 3]]})
    q = lf.select(pl.col("a").list.set_union([1, 2]))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_union_all_literals(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.lit([2, 5]).list.set_union(pl.lit([5, 1])))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_union_null_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [], None]})
    q = lf.select(pl.col("a").list.set_union(pl.lit(None)))
    assert_frame_equal(q.collect(engine=engine), q.collect())


@pytest.mark.parametrize("swap", [False, True])
def test_list_set_union_broadcast_first(engine: pl.GPUEngine, *, swap: bool) -> None:
    lf = pl.LazyFrame(
        {
            "a": [["2", "3", "3"], ["3", "1"], None, ["1", "2", "3"]],
            "b": [["1", "2"], ["4"], ["5"], None],
        }
    )
    column, first = pl.col("a"), pl.col("b").first()
    q = lf.select(
        first.list.set_union(column) if swap else column.list.set_union(first)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_union_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "k": [1, 1, 2, 2],
            "a": [[1, 2, 2], [None, 3], [], None],
            "b": [[3, 4], [3], [5], [6]],
        }
    )
    q = lf.group_by("k", maintain_order=True).agg(
        pl.col("a").list.set_union(pl.col("b"))
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_set_union_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(
        pl.col("a").implode().list.set_union(pl.col("a").implode())
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "operation", ["set_intersection", "set_difference", "set_symmetric_difference"]
)
def test_list_set_union_other_operation_not_supported(
    in_memory_engine: pl.GPUEngine, operation: str
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None], "b": [[2, 3], [1]]})
    q = lf.select(getattr(pl.col("a").list, operation)(pl.col("b")))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("inner", ["bool", "list", "struct", "null"])
def test_list_set_union_dtype_not_supported(
    in_memory_engine: pl.GPUEngine, inner: str
) -> None:
    data: list[object]
    if inner == "bool":
        data = [[True, None], [False]]
    elif inner == "list":
        data = [[[1], [2, 3]], [None]]
    elif inner == "struct":
        data = [[{"x": 1}], [None, {"x": 2}]]
    else:
        data = [[None], []]
    lf = pl.LazyFrame({"a": data, "b": data[::-1]})
    q = lf.select(pl.col("a").list.set_union(pl.col("b")))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_set_union_reconstruct() -> None:
    dtype = DataType(pl.List(pl.Int64()))
    node = expr.ListFunction(
        dtype,
        expr.ListFunction.Name.SetOperation,
        ("union",),
        expr.Col(dtype, "a"),
        expr.Col(dtype, "b"),
    )
    assert node.options == (SetOperation.UNION,)
    assert node.reconstruct(node.children) == node


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
