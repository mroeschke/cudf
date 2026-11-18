# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import decimal

import pyarrow as pa
import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("n", [0, 1, -1, 5, -5])
def test_list_shift(engine: pl.GPUEngine, n: int) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.shift(n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "values",
    [
        [["a", "bb", None], ["c"], [], None],
        [[decimal.Decimal("1.25"), None], [], None, [decimal.Decimal("3.50")]],
        [[[1], None, [2, 3]], [[]], [], None, [None]],
        [[{"x": 1, "y": "a"}, None], [{"x": None, "y": "b"}], [], None],
        [[[{"x": 1}], None, []], [[{"x": None}, None]], [], None],
    ],
)
@pytest.mark.parametrize("n", [1, -2])
def test_list_shift_dtypes(engine: pl.GPUEngine, values: list, n: int) -> None:
    lf = pl.LazyFrame({"a": values})
    q = lf.select(pl.col("a").list.shift(n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n,dtype",
    [
        (1, pl.Int8),
        (-1, pl.Int8),
        (2, pl.UInt8),
        (-(2**63), pl.Int64),
        (2**63 - 1, pl.UInt64),
        (2**63, pl.UInt64),
        (True, pl.Boolean),
        (False, pl.Boolean),
    ],
)
def test_list_shift_n_dtype(engine: pl.GPUEngine, n: int, dtype: pl.DataType) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.shift(pl.lit(n, dtype=dtype)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [None, pl.Int64, pl.UInt8, pl.String])
def test_list_shift_null_n(engine: pl.GPUEngine, dtype: pl.DataType | None) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.shift(pl.lit(None, dtype=dtype)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[], [None, None], [[], None, []]])
def test_list_shift_without_elements(engine: pl.GPUEngine, values: list) -> None:
    lf = pl.LazyFrame({"a": pl.Series(values, dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.shift(1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [1, -2])
def test_list_shift_sliced(engine: pl.GPUEngine, n: int, *, with_nulls: bool) -> None:
    null_list = None if with_nulls else [6, 7, 9]
    a = pl.Series([[3, 4, 8], null_list, [2, None, 5], [1, 7]])
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": a}).sort("k").slice(1, 3)
    q = lf.select(pl.col("a").list.shift(n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [1, -2])
def test_list_shift_sliced_null_rows_with_elements(
    in_memory_engine: pl.GPUEngine, n: int
) -> None:
    a = pa.ListArray.from_arrays(
        pa.array([0, 3, 5, 8, 10], type=pa.int32()),
        pa.array([3, 4, 8, 6, 7, 2, None, 5, 1, 7]),
        mask=pa.array([False, True, False, False]),
    )
    df = pl.from_arrow(pa.table({"a": a}))
    assert isinstance(df, pl.DataFrame)
    q = pl.concat([df.lazy(), df.lazy()]).tail(6).select(pl.col("a").list.shift(n))
    assert_gpu_result_equal(q, engine=in_memory_engine)


def test_list_shift_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.shift(1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_shift_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.shift(1))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_shift_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([1, 4, None, 9]).list.shift(-1))
    assert_gpu_result_equal(q, engine=engine)


def test_list_shift_column_n_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2]], "n": [1]})
    q = lf.select(pl.col("a").list.shift(pl.col("n")))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("n,dtype", [(1.0, pl.Float64), ("1", pl.String)])
def test_list_shift_n_not_supported(
    in_memory_engine: pl.GPUEngine, n: float | str, dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.shift(pl.lit(n, dtype=dtype)))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
