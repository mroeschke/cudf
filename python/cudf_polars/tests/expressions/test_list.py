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
@pytest.mark.parametrize("n,offset", [(2, 0), (2, 1), (1, 0), (3, 5)])
def test_list_gather_every(
    engine: pl.GPUEngine, inner: pl.DataType, n: int, offset: int
) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.gather_every(n, offset))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", ["list", "struct"])
def test_list_gather_every_nested(engine: pl.GPUEngine, inner: str) -> None:
    data: list[object]
    if inner == "list":
        data = [[[1], [2, 3], None, []], None, [], [None]]
    else:
        data = [[{"x": 1}, None, {"x": None}], None, [], [None, {"x": 2}]]
    lf = pl.LazyFrame({"a": data})
    q = lf.select(pl.col("a").list.gather_every(2, 1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [2, pl.lit(None)])
def test_list_gather_every_empty(engine: pl.GPUEngine, n: int | pl.Expr) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.gather_every(n, 1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String])
@pytest.mark.parametrize("how", ["slice", "tail"])
@pytest.mark.parametrize("n,offset", [(2, 1), (1, 0)])
def test_list_gather_every_sliced(
    engine: pl.GPUEngine,
    inner: pl.DataType,
    how: str,
    n: int,
    offset: int,
    *,
    with_nulls: bool,
) -> None:
    null_list = None if with_nulls else [5, 6]
    a = pl.Series([[3, 4], null_list, [2], [1]]).cast(pl.List(inner))
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": a}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.gather_every(n, offset))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n,offset",
    [
        (pl.lit(2, dtype=pl.Int8), pl.lit(1, dtype=pl.Int8)),
        (pl.lit(2, dtype=pl.Int16), pl.lit(1, dtype=pl.Int16)),
        (pl.lit(2, dtype=pl.Int64), pl.lit(1, dtype=pl.Int64)),
        (pl.lit(2, dtype=pl.UInt8), pl.lit(1, dtype=pl.UInt8)),
        (pl.lit(2, dtype=pl.UInt32), pl.lit(1, dtype=pl.UInt32)),
        (pl.lit(2, dtype=pl.UInt64), pl.lit(1, dtype=pl.UInt64)),
        (pl.lit(2**32 - 1, dtype=pl.UInt32), 0),
        (1, pl.lit(2**32 - 1, dtype=pl.UInt32)),
        (pl.lit(2**31, dtype=pl.UInt32), 1),
        (1, 2**31 - 1),
        (pl.lit(None), 1),
        (2, pl.lit(None)),
        (pl.lit(None, dtype=pl.Int64), 1),
        (2, pl.lit(None, dtype=pl.String)),
        (pl.lit(value=True), 1),
        (2, pl.lit(value=True)),
        (2, pl.lit(value=False)),
    ],
    ids=[
        "int8",
        "int16",
        "int64",
        "uint8",
        "uint32",
        "uint64",
        "n_uint32_max",
        "offset_uint32_max",
        "n_above_int32",
        "offset_int32_max",
        "null_n",
        "null_offset",
        "typed_null_n",
        "string_null_offset",
        "bool_n",
        "bool_offset",
        "false_offset",
    ],
)
def test_list_gather_every_arguments(
    engine: pl.GPUEngine, n: int | pl.Expr, offset: int | pl.Expr
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.gather_every(n, offset))
    assert_gpu_result_equal(q, engine=engine)


def test_list_gather_every_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.gather_every(2, 1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_gather_every_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.gather_every(2, 1))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_gather_every_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([1, None, 2]).list.gather_every(2, 0))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n,offset",
    [
        (0, 0),
        (pl.lit(value=False), 0),
        (-1, 0),
        (2, -1),
        (2**32, 0),
        (1, 2**32),
        (pl.lit(2**63, dtype=pl.UInt64), 0),
        (pl.lit(2.0), 0),
        (pl.col("n"), 0),
        (2, pl.col("offset")),
        (pl.col("n"), pl.col("offset")),
    ],
    ids=[
        "n_zero",
        "n_false",
        "negative_n",
        "negative_offset",
        "n_above_uint32",
        "offset_above_uint32",
        "n_above_int64",
        "float_n",
        "column_n",
        "column_offset",
        "column_n_and_offset",
    ],
)
def test_list_gather_every_argument_not_supported(
    in_memory_engine: pl.GPUEngine, n: int | pl.Expr, offset: int | pl.Expr
) -> None:
    lf = pl.LazyFrame({"a": [["a"], ["eb", "d"]], "n": [2, None], "offset": [None, 1]})
    q = lf.select(pl.col("a").list.gather_every(n, offset))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
