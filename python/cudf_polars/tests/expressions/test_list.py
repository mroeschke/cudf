# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("item", [2, 7, None])
def test_list_count_matches(engine: pl.GPUEngine, item: int | None) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(
        pl.col("a").list.count_matches(item).alias("untyped"),
        pl.col("a").list.count_matches(pl.lit(item, pl.Int64)).alias("typed"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("item", ["b", None])
def test_list_count_matches_string(engine: pl.GPUEngine, item: str | None) -> None:
    lf = pl.LazyFrame({"a": [["a", "b", "b"], [None, "c"], [], None, ["b"]]})
    q = lf.select(pl.col("a").list.count_matches(item))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("item", [float("nan"), -0.0, 1.5, 1, None])
def test_list_count_matches_float(
    engine: pl.GPUEngine, inner: pl.DataType, item: float | None
) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series(
                [
                    [float("nan"), 1.0, float("nan")],
                    [0.0, -0.0, 1.5],
                    [None, 1.5],
                    None,
                ],
                dtype=pl.List(inner),
            )
        }
    )
    q = lf.select(pl.col("a").list.count_matches(item))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "inner,item",
    [
        (pl.Int8, pl.lit(2)),
        (pl.UInt8, pl.lit(2)),
        (pl.Int64, pl.lit(2, pl.UInt64)),
        (pl.Int64, pl.lit(2, pl.Int8)),
        (pl.Float64, pl.lit(2)),
        (pl.Float32, pl.lit(2.0)),
        (pl.Decimal(10, 1), pl.lit(Decimal("2.00"))),
        (pl.Decimal(10, 1), pl.lit(2)),
    ],
    ids=[
        "int8_dyn_int",
        "uint8_dyn_int",
        "int64_uint64",
        "int64_int8",
        "float64_dyn_int",
        "float32_float64",
        "decimal_other_scale",
        "decimal_dyn_int",
    ],
)
def test_list_count_matches_converted_item(
    engine: pl.GPUEngine, inner: pl.DataType, item: pl.Expr
) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.count_matches(item))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "inner,item_dtype",
    [
        (pl.Int64, pl.Float64),
        (pl.Int64, pl.String),
        (pl.Int64, pl.Date),
        (pl.Int64, pl.Boolean),
        (pl.Int64, pl.Decimal(10, 2)),
        (pl.String, pl.Int8),
    ],
)
def test_list_count_matches_typed_null(
    engine: pl.GPUEngine, inner: pl.DataType, item_dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, None, None], [2], [], None]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.count_matches(pl.lit(None, item_dtype)))
    assert_gpu_result_equal(q, engine=engine)


def test_list_count_matches_temporal(
    engine: pl.GPUEngine, time_unit: Literal["ms", "us", "ns"]
) -> None:
    dt = datetime(2020, 1, 1)
    lf = pl.LazyFrame(
        {
            "dt": pl.Series(
                [[dt, None, dt], None], dtype=pl.List(pl.Datetime(time_unit))
            ),
            "d": [[date(2020, 1, 1), date(2020, 1, 2)], []],
        }
    )
    q = lf.select(
        pl.col("dt").list.count_matches(pl.lit(dt, pl.Datetime(time_unit))),
        pl.col("d").list.count_matches(date(2020, 1, 1)),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("item", [2, None])
def test_list_count_matches_empty(engine: pl.GPUEngine, item: int | None) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.count_matches(item))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("item", [3, None])
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_count_matches_sliced(
    engine: pl.GPUEngine, item: int | None, how: str, *, with_nulls: bool
) -> None:
    null_list = None if with_nulls else [None, 3]
    lf = pl.LazyFrame({"k": [3, 2, 1, 0], "a": [[3, 4], null_list, [2], [1]]}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.count_matches(item))
    assert_gpu_result_equal(q, engine=engine)


def test_list_count_matches_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.count_matches(2))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_count_matches_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.count_matches(1))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("item", [1, None])
def test_list_count_matches_literal(engine: pl.GPUEngine, item: int | None) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([1, None, 1]).list.count_matches(item))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "data,item",
    [
        ({"a": [[True, None, False]]}, True),
        ({"a": [[1, 2]], "b": [[3, 4]]}, 3),
    ],
    ids=["boolean_nulls_9141", "wildcard_expansion"],
)
def test_list_count_matches_polars_cases(
    engine: pl.GPUEngine, data: dict[str, list], item: int
) -> None:
    lf = pl.LazyFrame(data)
    q = lf.select(pl.all().list.count_matches(item))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "item", [pl.col("x"), pl.col("x").first()], ids=["column", "scalar"]
)
def test_list_count_matches_non_literal_item_not_supported(
    in_memory_engine: pl.GPUEngine, item: pl.Expr
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [2]], "x": [2, 1]})
    q = lf.select(pl.col("a").list.count_matches(item))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("inner", ["list", "struct"])
def test_list_count_matches_nested_not_supported(
    in_memory_engine: pl.GPUEngine, inner: str
) -> None:
    data: list[object]
    if inner == "list":
        data = [[[1], None], None]
    else:
        data = [[{"f": 1}, None], None]
    lf = pl.LazyFrame({"a": data})
    q = lf.select(pl.col("a").list.count_matches(None))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "inner,item",
    [
        (pl.Int64, pl.lit(2.0)),
        (pl.Int8, pl.lit(300)),
        (pl.Float32, pl.lit(0.1)),
        (pl.Decimal(10, 1), pl.lit(Decimal("2.05"))),
        (pl.Datetime("us"), pl.lit(datetime(2020, 1, 1), pl.Datetime("ms"))),
    ],
    ids=[
        "float_in_int",
        "int8_out_of_range",
        "float32_inexact",
        "decimal_inexact",
        "datetime_time_unit",
    ],
)
def test_list_count_matches_item_dtype_not_supported(
    in_memory_engine: pl.GPUEngine, inner: pl.DataType, item: pl.Expr
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([[], None], dtype=pl.List(inner))})
    q = lf.select(pl.col("a").list.count_matches(item))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
