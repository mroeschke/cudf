# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

plrs = operator.attrgetter("polars")(pl)
pytestmark = pytest.mark.skipif(
    not hasattr(plrs._expr_nodes, "ListFunction")
    or not hasattr(plrs._expr_nodes, "Explode"),
    reason="List expression nodes are not exposed by Polars",
)


def test_list_concat(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2], [], None, [None]],
            "b": [[3], [4, 5], [6], None],
        }
    )
    query = ldf.select(pl.col("a").list.concat("b"))
    assert_gpu_result_equal(query, engine=engine)


def test_concat_list(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2], [], None, [None]],
            "b": [[3], [4, 5], [6], None],
            "c": [7, 8, 9, 10],
        }
    )
    query = ldf.select(pl.concat_list("a", "b", "c"))
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


def test_list_len(engine: pl.GPUEngine) -> None:
    query = pl.LazyFrame({"a": [[1, 2, None], [], None, [3]]}).select(
        pl.col("a").list.len()
    )
    assert_gpu_result_equal(query, engine=engine)


def test_list_explode(engine: pl.GPUEngine) -> None:
    query = pl.LazyFrame({"a": [[1, 2], [], None, [None], [3]]}).select(
        pl.col("a").list.explode()
    )
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize(
    "empty_as_null,keep_nulls",
    [(False, True), (True, False)],
)
def test_list_explode_unsupported_options(
    engine: pl.GPUEngine, empty_as_null, keep_nulls
) -> None:
    query = pl.LazyFrame({"a": [[1], [], None]}).select(
        pl.col("a").list.explode(empty_as_null=empty_as_null, keep_nulls=keep_nulls)
    )
    assert_ir_translation_raises(query, engine, NotImplementedError)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_list_sort(engine: pl.GPUEngine, descending, nulls_last) -> None:
    query = pl.LazyFrame({"a": [[3, 1, 2], [None, 2, 1], [], None]}).select(
        pl.col("a").list.sort(descending=descending, nulls_last=nulls_last)
    )
    assert_gpu_result_equal(query, engine=engine)


def test_list_set_difference(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2, 2, 3], [], [None, 3], None],
            "b": [[2, 4], [3], [3, None], [1]],
        }
    )
    query = ldf.select(
        columns=pl.col("a").list.set_difference("b"),
        literal=pl.col("a").list.set_difference([2, 3]),
    )
    assert_gpu_result_equal(query, engine=engine)


def test_list_set_intersection(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2, 2, 3], [], [None, 3], None],
            "b": [[2, 4], [3], [3, None], [1]],
        }
    )
    query = ldf.select(pl.col("a").list.set_intersection("b"))
    assert_gpu_result_equal(query, engine=engine)


def test_list_set_union(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2, 2, 3], [], [None, 3], None],
            "b": [[2, 4], [3], [3, None], [1]],
        }
    )
    query = ldf.select(pl.col("a").list.set_union("b"))
    assert_gpu_result_equal(query, engine=engine)


def test_list_set_symmetric_difference(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {
            "a": [[1, 2, 2, 3], [], [None, 3], None],
            "b": [[2, 4], [3], [3, None], [1]],
        }
    )
    query = ldf.select(pl.col("a").list.set_symmetric_difference("b"))
    assert_gpu_result_equal(query, engine=engine)
