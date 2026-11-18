# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(params=[True, False], ids=["ignore_nulls", "keep_nulls"])
def ignore_nulls(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.mark.parametrize("separator", ["literal", "column"])
def test_list_join(engine: pl.GPUEngine, separator: str, *, ignore_nulls: bool) -> None:
    lf = pl.LazyFrame(
        {
            "a": [
                ["ab", "c", "d"],
                ["e", "f"],
                ["g"],
                [],
                None,
                ["a", None, "b", None],
                None,
                [None, None],
                ["c", "d"],
                [],
                ["", ""],
            ],
            "sep": ["&", None, "*", "_", "*", "-", "&", " ", "@", "/", ""],
        }
    )
    sep = "-" if separator == "literal" else pl.col("sep")
    q = lf.select(pl.col("a").list.join(sep, ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine)


def test_list_join_null_separator(engine: pl.GPUEngine, *, ignore_nulls: bool) -> None:
    lf = pl.LazyFrame({"a": [["a", None, "b"], [], None]})
    q = lf.select(
        pl.col("a").list.join(pl.lit(None, dtype=pl.String), ignore_nulls=ignore_nulls)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_join_separator_expression(
    engine: pl.GPUEngine, *, ignore_nulls: bool
) -> None:
    lf = pl.LazyFrame({"a": [["a", None, "b"], [], None], "sep": ["-", None, "+"]})
    q = lf.select(
        pl.col("a").list.join(
            pl.col("sep").str.to_uppercase(), ignore_nulls=ignore_nulls
        )
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("separator", ["literal", "column"])
def test_list_join_empty(engine: pl.GPUEngine, separator: str) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([], dtype=pl.List(pl.String)),
            "sep": pl.Series([], dtype=pl.String),
        }
    )
    sep = "-" if separator == "literal" else pl.col("sep")
    q = lf.select(pl.col("a").list.join(sep))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("separator", ["literal", "column"])
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_join_sliced(
    engine: pl.GPUEngine,
    separator: str,
    how: str,
    *,
    ignore_nulls: bool,
    with_nulls: bool,
) -> None:
    null_list = None if with_nulls else ["e"]
    null_sep = None if with_nulls else "*"
    lf = pl.LazyFrame(
        {
            "k": [3, 2, 1, 0],
            "a": [["c", None, "d"], null_list, ["b", None], ["a"]],
            "sep": ["-", "+", null_sep, ","],
        }
    ).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    sep = "-" if separator == "literal" else pl.col("sep")
    q = lf.select(pl.col("a").list.join(sep, ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine)


def test_list_join_literal(engine: pl.GPUEngine, *, ignore_nulls: bool) -> None:
    lf = pl.LazyFrame({"sep": ["-", None, "+"]})
    q = lf.select(
        pl.lit(["p", None, "q"]).list.join("-", ignore_nulls=ignore_nulls).alias("x"),
        pl.lit(["p", None, "q"])
        .list.join(pl.col("sep").first(), ignore_nulls=ignore_nulls)
        .alias("y"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_join_scalar_separator_expression(
    engine: pl.GPUEngine, *, ignore_nulls: bool
) -> None:
    lf = pl.LazyFrame({"a": [["a", None, "b"], [], None], "sep": ["-", None, "+"]})
    q = lf.select(
        pl.col("a").list.join(pl.col("sep").last(), ignore_nulls=ignore_nulls)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_join_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "k": [1, 1, 2, 2, 3],
            "a": [["a", "b"], [None, "c"], [], None, ["d"]],
            "sep": ["-", "+", None, ",", ""],
        }
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.join(pl.col("sep")))
    assert_gpu_result_equal(q, engine=engine)


def test_list_join_after_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"x": [0, 0, 1], "y": [0, 1, 2]})
    q = (
        lf.group_by("x", maintain_order=True)
        .agg(("'" + pl.col("y").cast(pl.String) + "'").alias("quoted_ys"))
        .with_columns(pl.col("quoted_ys").cast(pl.List(pl.String())).list.join(", "))
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_join_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine,
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": ["a", "b", "c"]})
    q = lf.group_by("k").agg(pl.col("a").implode().list.join("-"))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_join_non_string_separator_not_supported(
    in_memory_engine: pl.GPUEngine,
) -> None:
    lf = pl.LazyFrame({"a": [["a", "b"]], "sep": [1]})
    q = lf.select(pl.col("a").list.join(pl.col("sep")))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
