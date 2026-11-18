# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import polars as pl

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
def test_list_set_symmetric_difference(
    engine: pl.GPUEngine, inner: pl.DataType
) -> None:
    dtype = pl.List(inner)
    a = pl.Series(
        [[3, 1, 1, None], [], None, [1], None, [2, 2], [1, 2, 2], [2, 1, 2, 3, 5, 5]]
    )
    b = pl.Series(
        [[1, 4, None], [], [1], None, None, [], [None, 3], [3, 3, 7, 2, 6, 7]]
    )
    lf = pl.LazyFrame({"a": a.cast(dtype), "b": b.cast(dtype)})
    q = lf.select(pl.col("a").list.set_symmetric_difference(pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_symmetric_difference_float(engine: pl.GPUEngine) -> None:
    nan = float("nan")
    lf = pl.LazyFrame(
        {
            "a": [[nan, 1.0, -0.0], [0.0, nan], [None, nan], [-0.0, 0.0], [4.0]],
            "b": [[2.0, 0.0], [-0.0, -nan], [nan, None], [5.0], [0.0, -0.0]],
        }
    )
    q = lf.select(pl.col("a").list.set_symmetric_difference(pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)
    got = q.collect(engine=engine)["a"].explode().drop_nulls().drop_nans()
    expect = q.collect()["a"].explode().drop_nulls().drop_nans()
    np.testing.assert_array_equal(
        np.signbit(got.to_numpy()), np.signbit(expect.to_numpy())
    )


def test_list_set_symmetric_difference_empty(engine: pl.GPUEngine) -> None:
    dtype = pl.List(pl.String)
    lf = pl.LazyFrame(
        {"a": pl.Series([], dtype=dtype), "b": pl.Series([], dtype=dtype)}
    )
    q = lf.select(pl.col("a").list.set_symmetric_difference(pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("how", ["slice", "tail"])
@pytest.mark.parametrize("swap", [False, True])
def test_list_set_symmetric_difference_sliced(
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
    q = lf.select(pl.col(left).list.set_symmetric_difference(pl.col(right)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("other", [[], [2, 1], [1, None]])
@pytest.mark.parametrize("form", ["series", "python_list", "swap"])
def test_list_set_symmetric_difference_literal(
    engine: pl.GPUEngine, other: list[int | None], form: str
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 3], [], [None, 3], None, [5, 6, 7]]})
    column = pl.col("a")
    literal = pl.lit(pl.Series([other], dtype=pl.List(pl.Int64)))
    if form == "series":
        q = lf.select(column.list.set_symmetric_difference(literal))
    elif form == "python_list":
        q = lf.select(column.list.set_symmetric_difference(other))
    else:
        q = lf.select(literal.list.set_symmetric_difference(column))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_symmetric_difference_all_literals(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.lit([2, 5]).list.set_symmetric_difference(pl.lit([5, 1])))
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_symmetric_difference_null_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [], None]})
    q = lf.select(pl.col("a").list.set_symmetric_difference(pl.lit(None)))
    # The CPU in-memory engine panics on a null list literal.
    assert_gpu_result_equal(
        q,
        engine=engine,
        polars_collect_kwargs={"engine": "streaming"},  # type: ignore[dict-item]
    )


@pytest.mark.parametrize("swap", [False, True])
def test_list_set_symmetric_difference_broadcast_first(
    engine: pl.GPUEngine, *, swap: bool
) -> None:
    lf = pl.LazyFrame(
        {
            "a": [["2", "3", "3"], ["3", "1"], None, ["1", "2", "3"]],
            "b": [["1", "2"], ["4"], ["5"], None],
        }
    )
    column, first = pl.col("a"), pl.col("b").first()
    q = lf.select(
        first.list.set_symmetric_difference(column)
        if swap
        else column.list.set_symmetric_difference(first)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_set_symmetric_difference_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "k": [1, 1, 2, 2],
            "a": [[1, 2, 2], [None, 3], [], None],
            "b": [[3, 4], [3], [5], [6]],
        }
    )
    q = lf.group_by("k", maintain_order=True).agg(
        pl.col("a").list.set_symmetric_difference(pl.col("b"))
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_set_symmetric_difference_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(
        pl.col("a").implode().list.set_symmetric_difference(pl.col("a").implode())
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "operation", ["set_union", "set_intersection", "set_difference"]
)
def test_list_set_symmetric_difference_other_operation_not_supported(
    in_memory_engine: pl.GPUEngine, operation: str
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None], "b": [[2, 3], [1]]})
    q = lf.select(getattr(pl.col("a").list, operation)(pl.col("b")))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "inner",
    [pl.Boolean, pl.List(pl.Int64), pl.Struct({"x": pl.Int64}), pl.Null],
)
def test_list_set_symmetric_difference_dtype_not_supported(
    in_memory_engine: pl.GPUEngine, inner: pl.DataType
) -> None:
    data = pl.Series([[None], []], dtype=pl.List(inner))
    lf = pl.LazyFrame({"a": data, "b": data})
    q = lf.select(pl.col("a").list.set_symmetric_difference(pl.col("b")))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_set_symmetric_difference_reconstruct() -> None:
    dtype = DataType(pl.List(pl.Int64()))
    node = expr.ListFunction(
        dtype,
        expr.ListFunction.Name.SetOperation,
        ("symmetric_difference",),
        expr.Col(dtype, "a"),
        expr.Col(dtype, "b"),
    )
    assert node.options == (SetOperation.SYMMETRIC_DIFFERENCE,)
    assert node.reconstruct(node.children) == node


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
