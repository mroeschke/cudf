# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.utils.versions import POLARS_VERSION_LT_21


def assert_gather_out_of_bounds_raises(q: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    match = "gather indices are out of bounds"
    with pytest.raises(pl.exceptions.OutOfBoundsError, match=match):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.OutOfBoundsError, match=match)
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.OutOfBoundsError, match=match):
            q.collect(engine=engine)


# CollectKwargs has no "engine" key, so polars_collect_kwargs can't select it
def assert_gpu_result_equal_default_cpu_engine(
    q: pl.LazyFrame, engine: pl.GPUEngine
) -> None:
    """Compare with the default CPU engine, which differs from in-memory on Polars 2.0."""
    assert_frame_equal(q.collect(), q.collect(engine=engine))


@pytest.mark.parametrize(
    "indices",
    [[0, -1], [0, 1], [-1, 1], [0, 5], [-3, 2, 0], [0, None], [2, 2, -1, -4]],
)
def test_list_gather_literal_null_on_oob(engine: pl.GPUEngine, indices: list) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.gather(indices, null_on_oob=True))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("indices", [[0, -1], [1, -2, None], [0, 0]])
def test_list_gather_literal_in_bounds(engine: pl.GPUEngine, indices: list) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], None, [5, 6]]})
    q = lf.select(pl.col("a").list.gather(indices))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("indices", [[0, 2], [1, 2], [-3], [5]])
def test_list_gather_literal_out_of_bounds(engine: pl.GPUEngine, indices: list) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], None, [5, 6]]})
    q = lf.select(pl.col("a").list.gather(indices))
    assert_gather_out_of_bounds_raises(q, engine)


def test_list_gather_literal_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.gather(pl.lit([], dtype=pl.List(pl.Int64))))
    assert_gpu_result_equal_default_cpu_engine(q, engine)


@pytest.mark.skipif(POLARS_VERSION_LT_21, reason="Polars 2.0 panics on CPU")
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_gather_null_literal(engine: pl.GPUEngine, *, null_on_oob: bool) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(
        pl.col("a").list.gather(
            pl.lit(None, dtype=pl.List(pl.Int64)), null_on_oob=null_on_oob
        )
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("indices", [[0], [-1], [0, None]])
def test_list_gather_empty_list_out_of_bounds(
    engine: pl.GPUEngine, indices: list
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], []]})
    q = lf.select(pl.col("a").list.gather(indices))
    if POLARS_VERSION_LT_21:
        assert_gpu_result_equal_default_cpu_engine(q, engine)
    else:
        assert_gather_out_of_bounds_raises(q, engine)


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64],
)
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_gather_column(
    engine: pl.GPUEngine, dtype: pl.DataType, *, null_on_oob: bool
) -> None:
    indices = (
        [[2, -3, None], [-1, 1], [None], [-7], None, [0, 1]]
        if dtype.is_signed_integer()
        else [[2, 0, None], [1], [None], [7], None, [0, 1]]
    )
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5], [6, 7]],
            "i": pl.Series(indices, dtype=pl.List(dtype)),
        }
    )
    q = lf.select(pl.col("a").list.gather(pl.col("i"), null_on_oob=null_on_oob))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "index,dtype",
    [
        (3, pl.Int8),
        (-4, pl.Int8),
        (2**31 - 1, pl.Int32),
        (-(2**31), pl.Int32),
        (2**40, pl.Int64),
        (-(2**40), pl.Int64),
        (2**32 - 1, pl.UInt32),
        (2**63, pl.UInt64),
        (2**64 - 1, pl.UInt64),
    ],
)
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_gather_column_out_of_bounds(
    engine: pl.GPUEngine, index: int, dtype: pl.DataType, *, null_on_oob: bool
) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "i": pl.Series([[0, index], [1], [], [index], [0]], dtype=pl.List(dtype)),
        }
    )
    q = lf.select(pl.col("a").list.gather(pl.col("i"), null_on_oob=null_on_oob))
    if null_on_oob:
        assert_gpu_result_equal(q, engine=engine)
    else:
        assert_gather_out_of_bounds_raises(q, engine)


@pytest.mark.parametrize("inner", ["list", "struct", "string", "datetime"])
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_gather_nested(
    engine: pl.GPUEngine, inner: str, *, null_on_oob: bool
) -> None:
    data: list[object]
    if inner == "list":
        data = [[[1], [2, 3]], None, [None, [4]]]
    elif inner == "struct":
        data = [[{"x": 1}, {"x": None}], None, [None, {"x": 2}]]
    elif inner == "string":
        data = [["a", "b"], None, [None, "c"]]
    else:
        data = [[datetime(2020, 1, 1), None], None, [None, datetime(2021, 1, 1)]]
    lf = pl.LazyFrame({"a": data, "i": [[1, 0, -1], [0], [None, -2]]})
    q = lf.select(
        pl.col("a").list.gather([-1, 0], null_on_oob=null_on_oob).alias("literal"),
        pl.col("a").list.gather(pl.col("i"), null_on_oob=null_on_oob).alias("column"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_gather_string_null_on_oob(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [["A", "B", "C"], ["A"], ["B"], ["1", "2"], ["e"]]})
    q = lf.select(
        pl.col("a").list.gather([0, 2], null_on_oob=True).alias("two"),
        pl.col("a").list.gather([0, 1, 2], null_on_oob=True).alias("three"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_gather_column_expression_index(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "k": [0, 1, 2, 3, 4],
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "i": [[0, -1], [1], [5], None, [0]],
        }
    )
    q = lf.select(
        pl.col("a").list.gather(pl.col("i").first(), null_on_oob=True).alias("first"),
        pl.col("a")
        .list.gather(pl.when(pl.col("k") != 1).then(pl.col("i")), null_on_oob=True)
        .alias("when"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_gather_literal_list_column_index(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[0.1, 0.2, 0.3]]})
    q = lf.with_columns(pl.lit([0]).alias("c")).with_columns(
        gather=pl.col("a").list.gather(pl.col("c"), null_on_oob=True)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_gather_scalar_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"i": [[0, -1], [3], None, [None, 1]]})
    q = lf.select(
        pl.lit(pl.Series([[1, 2, 3]]))
        .first()
        .list.gather(pl.col("i"), null_on_oob=True)
        .alias("column"),
        pl.lit(pl.Series([[1, 2, 3]])).first().list.gather([2, 0]).alias("literal"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_gather_all_null(engine: pl.GPUEngine, *, null_on_oob: bool) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([None, None], dtype=pl.List(pl.Int64)),
            "i": pl.Series([[0], [0]], dtype=pl.List(pl.UInt32)),
        }
    )
    q = lf.select(
        pl.col("a").list.gather([3], null_on_oob=null_on_oob).alias("literal"),
        pl.col("a").list.gather(pl.col("i"), null_on_oob=null_on_oob).alias("column"),
    )
    assert_gpu_result_equal_default_cpu_engine(q, engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21, reason="Polars 2.0 returns List(Null) for zero rows"
)
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_gather_zero_rows(engine: pl.GPUEngine, *, null_on_oob: bool) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([], dtype=pl.List(pl.Int64)),
            "i": pl.Series([], dtype=pl.List(pl.UInt32)),
        }
    )
    q = lf.select(
        pl.col("a").list.gather([3], null_on_oob=null_on_oob).alias("literal"),
        pl.col("a").list.gather(pl.col("i"), null_on_oob=null_on_oob).alias("column"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21, reason="Polars 2.0 panics on list.gather after sort"
)
@pytest.mark.parametrize("window", ["slice", "tail"])
@pytest.mark.parametrize("null_on_oob", [False, True])
def test_list_gather_sliced(
    engine: pl.GPUEngine, window: str, *, with_nulls: bool, null_on_oob: bool
) -> None:
    lists: list[list[int | None] | None] = [
        [9, 8],
        [1, 2, 2],
        [3, 4],
        [4],
        [6, None],
        [7, 5],
    ]
    indices: list[list[int | None] | None] = [[0], [0, -1], [-1], [0], [1], [5]]
    if with_nulls:
        lists[1] = None
        lists[3] = None
        indices[2] = None
        indices[4] = [None, 0]
    lf = pl.LazyFrame({"k": [5, 4, 3, 2, 1, 0], "a": lists, "i": indices}).sort("k")
    lf = lf.tail(3) if window == "tail" else lf.slice(1, 3)
    q = lf.select(
        pl.col("a").list.gather([0, -1], null_on_oob=null_on_oob).alias("literal"),
        pl.col("a").list.gather(pl.col("i"), null_on_oob=null_on_oob).alias("column"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21, reason="Polars 2.0 panics on list.gather after sort"
)
@pytest.mark.parametrize("window", ["slice", "tail"])
def test_list_gather_sliced_out_of_bounds_raises(
    engine: pl.GPUEngine, window: str
) -> None:
    lf = pl.LazyFrame(
        {
            "k": [5, 4, 3, 2, 1, 0],
            "a": [[9, 8], [1, 2, 2], [None, 3], [4], [6, 7], [5]],
            "i": [[0], [0], [0, 5], None, [0], [0]],
        }
    ).sort("k")
    lf = lf.tail(3) if window == "tail" else lf.slice(1, 3)
    q = lf.select(pl.col("a").list.gather(pl.col("i")))
    assert_gather_out_of_bounds_raises(q, engine)


def test_list_gather_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(
        pl.col("a").list.gather([0, -1], null_on_oob=True)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_gather_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.gather([0], null_on_oob=True))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
