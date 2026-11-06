# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(
    params=[
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]
)
def integer_dtype(request: pytest.FixtureRequest) -> pl.DataType:
    return request.param


def test_bitwise_trailing_zeros(
    engine: pl.GPUEngine, integer_dtype: pl.DataType
) -> None:
    info = np.iinfo(str(integer_dtype).lower())
    patterns = (
        0xFCEF_0123,
        0xFFFF_FFFF,
        0xFFF0_FFE1_ABCD_EF01,
        0xAAAA_AAAA_AAAA_AAAA,
        1 << (info.bits - 1),
    )
    masked = (v & ((1 << info.bits) - 1) for v in patterns)
    values = [
        0,
        1,
        6,
        8,
        info.min,
        info.max,
        None,
        *(v - (1 << info.bits) if v > info.max else v for v in masked),
    ]
    lf = pl.LazyFrame({"a": pl.Series(values, dtype=integer_dtype)})
    q = lf.select(pl.col("a").bitwise_trailing_zeros())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_trailing_zeros_boolean(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [True, False, None]})
    q = lf.select(pl.col("a").bitwise_trailing_zeros())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_trailing_zeros_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.Int16)})
    q = lf.select(pl.col("a").bitwise_trailing_zeros())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_trailing_zeros_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.lit(8, dtype=pl.Int8).bitwise_trailing_zeros())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_trailing_zeros_with_column(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([4, 0, None, -2], dtype=pl.Int32)})
    q = lf.with_columns(b=pl.col("a").bitwise_trailing_zeros())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128])
def test_bitwise_trailing_zeros_128_bit_not_supported(
    in_memory_engine: pl.GPUEngine, dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([1, 0, None], dtype=dtype)})
    q = lf.select(pl.col("a").bitwise_trailing_zeros())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "func",
    [
        "bitwise_count_ones",
        "bitwise_count_zeros",
        "bitwise_leading_ones",
        "bitwise_leading_zeros",
        "bitwise_trailing_ones",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    ],
)
def test_bitwise_function_not_supported(
    in_memory_engine: pl.GPUEngine, func: str
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([1, 2, None], dtype=pl.Int32)})
    q = lf.select(getattr(pl.col("a"), func)())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_bitwise_trailing_zeros_dtype_not_supported() -> None:
    child = expr.Col(DataType(pl.Float64()), "a")
    with pytest.raises(NotImplementedError, match="Float64"):
        expr.BitwiseFunction(
            DataType(pl.UInt32()), expr.BitwiseFunction.Name.TrailingZeros, child
        )
