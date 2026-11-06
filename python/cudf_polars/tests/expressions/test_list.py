# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine


def assert_get_out_of_bounds_raises(q: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    match = "get index is out of bounds"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.ComputeError, match=match)
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            q.collect(engine=engine)


@pytest.mark.parametrize("index", [0, 1, 2, -1, -2, -3, 5, -5, 2**40, -(2**40)])
def test_list_get_literal_null_on_oob(engine: pl.GPUEngine, index: int) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.get(index, null_on_oob=True))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("index", [0, -1])
def test_list_get_literal_in_bounds(engine: pl.GPUEngine, index: int) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], None, [5]]})
    q = lf.select(pl.col("a").list.get(index))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("index", [0, 1, -2, 5, 2**40, -(2**40)])
@pytest.mark.parametrize("all_null", [False, True])
def test_list_get_literal_out_of_bounds(
    engine: pl.GPUEngine, index: int, *, all_null: bool
) -> None:
    data = [None, None] if all_null else [[1, 2, 2], [None, 3], [], None, [5]]
    lf = pl.LazyFrame({"a": pl.Series(data, dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.get(index))
    if all_null:
        assert_gpu_result_equal(q, engine=engine)
    else:
        assert_get_out_of_bounds_raises(q, engine)


@pytest.mark.parametrize("null_on_oob", [False, True])
@pytest.mark.parametrize(
    "value,dtype", [(None, pl.Int64), (2**63, pl.UInt64), (2**64 - 1, pl.UInt64)]
)
def test_list_get_literal_null_index(
    engine: pl.GPUEngine, value: int | None, dtype: pl.DataType, *, null_on_oob: bool
) -> None:
    lf = pl.LazyFrame({"a": [[{"x": 1}], [{"x": None}, None], [], None]})
    q = lf.select(
        pl.col("a").list.get(pl.lit(value, dtype=dtype), null_on_oob=null_on_oob)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_get_first_last(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [["x", "y"], [None, "z"], [], None]})
    q = lf.select(
        pl.col("a").list.first().alias("first"), pl.col("a").list.last().alias("last")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64],
)
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_get_column(
    engine: pl.GPUEngine, dtype: pl.DataType, *, null_on_oob: bool
) -> None:
    indices = [2, -2, None, 7, -1] if dtype.is_signed_integer() else [2, 0, None, 7, 0]
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "i": pl.Series(indices, dtype=dtype),
        }
    )
    q = lf.select(pl.col("a").list.get(pl.col("i"), null_on_oob=null_on_oob))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "indices", [[0, 0, 0, None, 0], [-4, 0, None, 0, 0], [3, 0, None, 0, 0]]
)
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_get_column_out_of_bounds(
    engine: pl.GPUEngine, indices: list, *, null_on_oob: bool
) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "i": pl.Series(indices, dtype=pl.Int32),
        }
    )
    q = lf.select(pl.col("a").list.get(pl.col("i"), null_on_oob=null_on_oob))
    if null_on_oob:
        assert_gpu_result_equal(q, engine=engine)
    else:
        assert_get_out_of_bounds_raises(q, engine)


@pytest.mark.parametrize(
    "indices,dtype,expect_raise",
    [
        ([2**40, -(2**40), 2**31, -(2**31), 2**31 - 1], pl.Int64, True),
        ([2**32 - 1, 2**31, 2**31 - 1, None, 0], pl.UInt32, True),
        ([2**63 - 1, 2**32, 2**31, None, 0], pl.UInt64, True),
        ([2**63, 1, 2**64 - 1, 7, 0], pl.UInt64, False),
    ],
)
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_get_column_outside_size_type(
    engine: pl.GPUEngine,
    indices: list,
    dtype: pl.DataType,
    *,
    expect_raise: bool,
    null_on_oob: bool,
) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "i": pl.Series(indices, dtype=dtype),
        }
    )
    q = lf.select(pl.col("a").list.get(pl.col("i"), null_on_oob=null_on_oob))
    if expect_raise and not null_on_oob:
        assert_get_out_of_bounds_raises(q, engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


def test_list_get_column_expression_index(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "i": pl.Series([0, -1, 0, 1, None], dtype=pl.Int64),
        }
    )
    q = lf.select(
        pl.col("a").list.get(pl.col("i") + 1, null_on_oob=True).alias("expr"),
        pl.col("a").list.get(pl.col("i").first(), null_on_oob=True).alias("scalar"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_get_scalar_list_column_index(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"i": pl.Series([0, -1, 3, None], dtype=pl.Int64)})
    q = lf.select(
        pl.lit(pl.Series([[1, 2, 3]])).first().list.get(pl.col("i"), null_on_oob=True)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_get_zero_rows(engine: pl.GPUEngine, *, null_on_oob: bool) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([], dtype=pl.List(pl.String)),
            "i": pl.Series([], dtype=pl.Int64),
        }
    )
    q = lf.select(
        pl.col("a").list.get(3, null_on_oob=null_on_oob).alias("literal"),
        pl.col("a").list.get(pl.col("i"), null_on_oob=null_on_oob).alias("column"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("window", ["slice", "tail"])
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_get_sliced(
    engine: pl.GPUEngine, window: str, *, with_nulls: bool, null_on_oob: bool
) -> None:
    lists: list[list[int | None] | None] = [
        [9, 8],
        [1, 2, 2],
        [3, 4],
        [4],
        [6, None],
        [],
    ]
    indices: list[int | None] = [0, 0, -1, 0, 1, 7]
    if with_nulls:
        lists[1] = None
        lists[3] = None
        indices[4] = None
    lf = pl.LazyFrame(
        {
            "k": [5, 4, 3, 2, 1, 0],
            "a": lists,
            "i": pl.Series(indices, dtype=pl.Int32),
        }
    ).sort("k")
    lf = lf.tail(3) if window == "tail" else lf.slice(1, 3)
    q = lf.select(
        pl.col("a").list.get(0, null_on_oob=null_on_oob).alias("literal"),
        pl.col("a").list.get(-1, null_on_oob=null_on_oob).alias("negative"),
        pl.col("a").list.get(pl.col("i"), null_on_oob=null_on_oob).alias("column"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("window", ["slice", "tail"])
def test_list_get_sliced_out_of_bounds_raises(
    engine: pl.GPUEngine, window: str
) -> None:
    lf = pl.LazyFrame(
        {
            "k": [5, 4, 3, 2, 1, 0],
            "a": [[9, 8], [1, 2, 2], [None, 3], [4], [6, 7], [5]],
            "i": pl.Series([0, 0, 5, 0, 0, 0], dtype=pl.Int32),
        }
    ).sort("k")
    lf = lf.tail(3) if window == "tail" else lf.slice(1, 3)
    q = lf.select(pl.col("a").list.get(pl.col("i")))
    assert_get_out_of_bounds_raises(q, engine)


@pytest.mark.parametrize("dtype", [pl.Float64, pl.Boolean, pl.String])
def test_list_get_non_integer_index_not_supported(
    in_memory_engine: pl.GPUEngine, dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [3]], "i": pl.Series([0, 1]).cast(dtype)})
    q = lf.select(pl.col("a").list.get(pl.col("i"), null_on_oob=True))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_get_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(
        pl.col("a").list.get(-1, null_on_oob=True)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_get_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.get(0, null_on_oob=True))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
