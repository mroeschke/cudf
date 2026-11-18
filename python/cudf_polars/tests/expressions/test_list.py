# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("fields", [[""], ["p", "q"], ["p", "q", "r", "s"]])
@pytest.mark.parametrize(
    "inner",
    [
        pl.Int8,
        pl.Int64,
        pl.UInt64,
        pl.Float64,
        pl.Boolean,
        pl.String,
        pl.Decimal(10, 2),
        pl.Datetime("us"),
    ],
    ids=str,
)
def test_list_to_struct(
    engine: pl.GPUEngine, inner: pl.DataType, fields: list[str]
) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.to_struct(fields))
    assert_gpu_result_equal(q, engine=engine)


def test_list_to_struct_nested_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[[1], [2, 3]], None, [], [None], [[]]]})
    q = lf.select(pl.col("a").list.to_struct(["p", "q"]))
    assert_gpu_result_equal(q, engine=engine)


def test_list_to_struct_struct(engine: pl.GPUEngine) -> None:
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
    q = lf.select(pl.col("a").list.to_struct(["p", "q"]))
    assert_gpu_result_equal(q, engine=engine)


def test_list_to_struct_all_null(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([None, None], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.to_struct(["p", "q", "r"]))
    assert_gpu_result_equal(q, engine=engine)


def test_list_to_struct_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.to_struct(["p", "q"]))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String, pl.List(pl.Int64)], ids=str)
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_to_struct_sliced(
    engine: pl.GPUEngine, inner: pl.DataType, how: str, *, with_nulls: bool
) -> None:
    rows = [[3, 4], None if with_nulls else [6], [2], [1]]
    if inner == pl.List(pl.Int64):
        a = pl.Series([None if r is None else [[x] for x in r] for r in rows])
    else:
        a = pl.Series(rows).cast(pl.List(inner))
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": a}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.to_struct(["p", "q"]))
    assert_gpu_result_equal(q, engine=engine)


def test_list_to_struct_field(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1.0, 2.0], [1.5], None]})
    q = lf.select(pl.col("a").list.to_struct(["p", "q"])).select(
        pl.col("a").struct.field("q")
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_to_struct_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(
        pl.col("a").list.to_struct(["p", "q"]).struct.field("q")
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_to_struct_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([4, 5]).list.to_struct(["p", "q"]))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("fields", [[], ["p", "p"], ["p", "q", "p"]])
def test_list_to_struct_fields_not_supported(
    in_memory_engine: pl.GPUEngine, fields: list[str]
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.to_struct(fields))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_to_struct_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.to_struct(["p", "q"]))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_to_struct_implode_in_rolling_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"t": [1, 2, 3], "a": values})
    q = lf.rolling("t", period="2i").agg(
        pl.col("a").implode().list.to_struct(["p", "q"])
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
