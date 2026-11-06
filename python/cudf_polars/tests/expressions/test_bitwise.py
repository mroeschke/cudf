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


def test_bitwise_leading_ones(engine: pl.GPUEngine, integer_dtype: pl.DataType) -> None:
    info = np.iinfo(str(integer_dtype).lower())
    patterns = (
        1 << (info.bits - 1),
        (1 << info.bits) - 1,
        (1 << info.bits) - 2,
        0b1110 << (info.bits - 4),
        0xFCEF_0123,
        0xFFFF_FFFF,
        0xFFF0_FFE1_ABCD_EF01,
        0xAAAA_AAAA_AAAA_AAAA,
    )
    masked = (v & ((1 << info.bits) - 1) for v in patterns)
    values = [
        0,
        1,
        6,
        info.min,
        info.max,
        None,
        *(v - (1 << info.bits) if v > info.max else v for v in masked),
    ]
    lf = pl.LazyFrame({"a": pl.Series(values, dtype=integer_dtype)})
    q = lf.select(pl.col("a").bitwise_leading_ones())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_leading_ones_boolean(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [True, False, None]})
    q = lf.select(pl.col("a").bitwise_leading_ones())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_leading_ones_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.Int16)})
    q = lf.select(pl.col("a").bitwise_leading_ones())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_leading_ones_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.lit(-3, dtype=pl.Int8).bitwise_leading_ones())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_leading_ones_with_column(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([1, -1, -8, None], dtype=pl.Int32)})
    q = lf.with_columns(
        b=pl.col("a").bitwise_leading_ones(),
        c=pl.lit(0xF0, dtype=pl.UInt8).bitwise_leading_ones(),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "func",
    [
        "bitwise_count_ones",
        "bitwise_count_zeros",
        "bitwise_leading_zeros",
        "bitwise_trailing_ones",
        "bitwise_trailing_zeros",
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


def test_bitwise_leading_ones_int128_not_supported(
    in_memory_engine: pl.GPUEngine,
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([1, -1], dtype=pl.Int128)})
    q = lf.select(pl.col("a").bitwise_leading_ones())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_bitwise_leading_ones_dtype_not_supported() -> None:
    child = expr.Col(DataType(pl.Float64()), "a")
    with pytest.raises(NotImplementedError, match="Float64"):
        expr.BitwiseFunction(
            DataType(pl.UInt32()), expr.BitwiseFunction.Name.LeadingOnes, child
        )
