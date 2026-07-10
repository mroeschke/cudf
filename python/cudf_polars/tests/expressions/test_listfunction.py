# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_list_concat(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2], [], None, [None]],
            "b": [[3], [4, 5], [6], None],
        }
    )
    query = ldf.select(pl.col("a").list.concat("b"))
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_list_contains(engine: pl.GPUEngine, nulls_equal) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2, None], [], [None], None],
            "item": [1, None, None, 1],
        }
    )
    query = ldf.select(
        pl.col("a").list.contains(pl.col("item"), nulls_equal=nulls_equal)
    )
    assert_gpu_result_equal(query, engine=engine)


def test_list_drop_nulls(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame({"a": [[None, 1, None, 2], [None], [], None, [3, 4]]})
    query = ldf.select(pl.col("a").list.drop_nulls())
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_get(engine: pl.GPUEngine, null_on_oob) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2], [3], [4], None],
            "index": [0, -1, 0, 0],
        }
    )
    query = ldf.select(pl.col("a").list.get(pl.col("index"), null_on_oob=null_on_oob))
    assert_gpu_result_equal(query, engine=engine)


def test_list_get_oob_raises(engine: pl.GPUEngine) -> None:
    query = pl.LazyFrame({"a": [[1], []]}).select(pl.col("a").list.get(0))
    with pytest.raises(pl.exceptions.ComputeError):
        query.collect(engine=engine)


def test_list_first(engine: pl.GPUEngine) -> None:
    query = pl.LazyFrame({"a": [[1, 2], [], None, [None]]}).select(
        pl.col("a").list.first()
    )
    assert_gpu_result_equal(query, engine=engine)


def test_list_last(engine: pl.GPUEngine) -> None:
    query = pl.LazyFrame({"a": [[1, 2], [], None, [None]]}).select(
        pl.col("a").list.last()
    )
    assert_gpu_result_equal(query, engine=engine)
