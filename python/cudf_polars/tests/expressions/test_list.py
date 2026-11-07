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
from cudf_polars.utils.versions import POLARS_VERSION_LT_21


@pytest.mark.parametrize(
    "offset,length",
    [
        (1, 1),
        (-2, 5),
        (1, None),
        (10, 2),
        (0, 0),
        (-1, 1),
        (-5, 3),
        (-4, 1),
        (-10, 2),
        (0, -1),
        (1, -2),
        (2**40, 1),
        (-(2**40), 2**40 - 1),
        (-(2**63), 2**63 - 1),
    ],
)
def test_list_slice_literal(
    engine: pl.GPUEngine, offset: int, length: int | None
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.slice(offset, length))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [0, 2, 10, -1])
def test_list_slice_head_tail(engine: pl.GPUEngine, n: int) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(
        pl.col("a").list.head(n).alias("head"), pl.col("a").list.tail(n).alias("tail")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pytest.param(
            pl.UInt64,
            marks=pytest.mark.skipif(
                POLARS_VERSION_LT_21, reason="UInt64 offsets fall back on Polars 2.0"
            ),
        ),
    ],
)
def test_list_slice_column(engine: pl.GPUEngine, dtype: pl.DataType) -> None:
    signed = dtype.is_signed_integer()
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "o": pl.Series(
                [1, -1, 0, 2, -3] if signed else [1, 5, 0, 2, 0], dtype=dtype
            ),
            "l": pl.Series(
                [2, -1, 3, 1, 1] if signed else [2, 1, 3, 1, 0], dtype=dtype
            ),
        }
    )
    q = lf.select(
        pl.col("a").list.slice(pl.col("o"), pl.col("l")).alias("both"),
        pl.col("a").list.slice(pl.col("o")).alias("offset"),
        pl.col("a").list.slice(1, pl.col("l")).alias("length"),
        pl.col("a").list.head(pl.col("l")).alias("head"),
        pl.col("a").list.tail(pl.col("l")).alias("tail"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(POLARS_VERSION_LT_21, reason="Polars 2.0 raises for null offsets")
@pytest.mark.parametrize("offset", ["null", "typed_null", "column", "literal"])
@pytest.mark.parametrize("length", ["null", "typed_null", "column", "literal"])
def test_list_slice_nulls(engine: pl.GPUEngine, offset: str, length: str) -> None:
    lf = pl.LazyFrame(
        {"a": [None, [1, 2, 3], [4, 5, 6]], "o": [1, None, -1], "l": [1, None, 1]}
    )
    offsets = {
        "null": pl.lit(None),
        "typed_null": pl.lit(None, pl.Int64),
        "column": pl.col("o"),
        "literal": pl.lit(1),
    }
    lengths = offsets | {"column": pl.col("l")}
    q = lf.select(pl.col("a").list.slice(offsets[offset], lengths[length]))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(POLARS_VERSION_LT_21, reason="Polars 2.0 raises for null offsets")
def test_list_slice_null_scalar_offset(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2], [3]],
            "b": [1, 2],
            "n": pl.Series([None, 1], dtype=pl.Int64),
        }
    )
    q = lf.select(
        pl.when(pl.lit(value=True))
        .then(None)
        .otherwise(pl.col("a").list.slice(pl.col("b"), 1)),
        pl.col("a").list.slice(pl.col("n").first(), 1).alias("scalar"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21, reason="UInt64 offsets fall back on Polars 2.0"
)
def test_list_slice_uint64_outside_int64(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "o": pl.Series([2**64 - 1, 2**63, 1, 0, 0], dtype=pl.UInt64),
            "l": pl.Series([2**63, 2**64 - 1, 1, None, 2], dtype=pl.UInt64),
        }
    )
    q = lf.select(
        pl.col("a").list.slice(pl.col("o"), 1).alias("offset"),
        pl.col("a").list.slice(0, pl.col("l")).alias("length"),
        pl.col("a")
        .list.slice(pl.lit(2**64 - 1, dtype=pl.UInt64), 1)
        .alias("literal_offset"),
        pl.col("a")
        .list.slice(0, pl.lit(2**64 - 1, dtype=pl.UInt64))
        .alias("literal_length"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    not POLARS_VERSION_LT_21, reason="Polars 2.1 slices to null for null offsets"
)
def test_list_slice_null_column_offset_raises(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [3]], "o": pl.Series([None, 1], dtype=pl.Int64)})
    q = lf.select(pl.col("a").list.slice(pl.col("o"), 1))
    match = "could not extract number"
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


@pytest.mark.skipif(
    not POLARS_VERSION_LT_21, reason="Polars 2.1 supports these offsets"
)
@pytest.mark.parametrize("offset", ["null", "uint64"])
def test_list_slice_offset_not_supported_polars_20(
    in_memory_engine: pl.GPUEngine, offset: str
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [3]], "u": pl.Series([0, 1], dtype=pl.UInt64)})
    q = lf.select(
        pl.col("a").list.slice(pl.lit(None) if offset == "null" else pl.col("u"), 1)
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_slice_broadcast_list(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"o": [0, -1, 5, 1], "l": [1, 3, 2, -1]})
    q = lf.select(
        pl.lit([1, 2, 3]).list.slice(pl.col("o"), pl.col("l")).alias("both"),
        pl.lit([1, 2, 3]).list.slice(-2, pl.col("l")).alias("length"),
        pl.lit(None, dtype=pl.List(pl.Int64)).list.slice(pl.col("o")).alias("null"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("offset", ["literal", "series"])
@pytest.mark.parametrize("length", ["literal", "series"])
def test_list_slice_broadcast_series(
    engine: pl.GPUEngine, offset: str, length: str
) -> None:
    q = pl.LazyFrame().select(
        pl.lit([0, 1, 2]).list.slice(
            pl.lit(pl.Series([0, 1, 2])) if offset == "series" else 1,
            pl.lit(pl.Series([3, 2, 1])) if length == "series" else 2,
        )
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(POLARS_VERSION_LT_21, reason="Polars 2.0 raises for null offsets")
def test_list_slice_broadcast_nulls(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"o": [1, None, -1], "l": [1, None, 1]})
    a = pl.lit(pl.Series([[1, 2, 3]]))
    q = lf.select(
        o=a.list.slice(pl.col("o"), 1),
        l=a.list.slice(1, pl.col("l")),
        ol=a.list.slice(pl.col("o"), pl.col("l")),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", ["string", "list", "struct"])
def test_list_slice_nested(engine: pl.GPUEngine, inner: str) -> None:
    data: list[object]
    if inner == "string":
        data = [["x", "y", None], None, [], ["z"]]
    elif inner == "list":
        data = [[[1], None, [2, 3]], None, [], [[4]]]
    else:
        data = [[{"x": 1}, None, {"x": 2}], None, [], [{"x": None}]]
    lf = pl.LazyFrame({"a": data, "o": pl.Series([1, 0, 0, -1], dtype=pl.Int32)})
    q = lf.select(
        pl.col("a").list.slice(pl.col("o"), 2).alias("slice"),
        pl.col("a").list.head(1).alias("head"),
        pl.col("a").list.tail(1).alias("tail"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_slice_polars_cases(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 3, 4], [10, 2, 1]],
            "o": [1, 2],
            "l": [3, 2],
            "s": [["a", "b", "c"], ["d"]],
        }
    )
    q = lf.select(
        pl.col("a").list.head(2).alias("head"),
        pl.col("a").list.tail(2).alias("tail"),
        pl.col("a").list.tail(200).alias("tail_long"),
        pl.col("a").list.head(200).alias("head_long"),
        pl.col("a").list.slice(1, 2).alias("slice"),
        pl.col("a").list.slice(-5, 2).alias("slice_before_start"),
        pl.col("a").list.slice(1).alias("slice_to_end"),
        pl.col("a").list.slice("o", "l").alias("columns"),
        pl.col("a").list.slice("o", 1).alias("column_offset"),
        pl.col("a").list.slice(-2, "l").alias("column_length"),
        pl.col("s").list.tail(pl.lit(1, pl.UInt32)).alias("tail_uint32"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_slice_list_of_struct(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [
                [{"a": 1, "b": 4}, {"a": 3, "b": 6}],
                [{"a": 10, "b": 40}, {"a": 20, "b": 50}, {"a": 30, "b": 60}],
            ]
        }
    )
    q = lf.select(
        pl.col("a").list.head(1).alias("head"), pl.col("a").list.tail(1).alias("tail")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21,
    reason="Polars 2.0 returns List(Null) for an all-null list column",
)
@pytest.mark.parametrize("offset", [0, "column"])
@pytest.mark.parametrize("length", [1, "column"])
def test_list_slice_all_null(
    engine: pl.GPUEngine, offset: int | str, length: int | str
) -> None:
    lf = pl.LazyFrame(
        {"a": [None, None], "n": [0, 0]},
        schema={"a": pl.List(pl.String), "n": pl.Int64},
    )
    q = lf.select(
        pl.col("a").list.slice(
            pl.col("n") if offset == "column" else offset,
            pl.col("n") if length == "column" else length,
        )
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "argument",
    [
        "literal",
        pytest.param(
            "column",
            marks=pytest.mark.skipif(
                POLARS_VERSION_LT_21,
                reason="Polars 2.0 returns List(Null) for an empty list column",
            ),
        ),
    ],
)
def test_list_slice_zero_rows(engine: pl.GPUEngine, argument: str) -> None:
    lf = pl.LazyFrame(
        {
            "a": pl.Series([], dtype=pl.List(pl.String)),
            "o": pl.Series([], dtype=pl.Int64),
        }
    )
    n = pl.col("o") if argument == "column" else 1
    q = lf.select(pl.col("a").list.slice(n, n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("window", ["slice", "tail"])
def test_list_slice_sliced(
    engine: pl.GPUEngine, window: str, *, with_nulls: bool
) -> None:
    if with_nulls and POLARS_VERSION_LT_21:
        pytest.skip("Polars 2.0 raises for null offsets")
    lists: list[list[int | None] | None] = [
        [9, 8],
        [1, 2, 2],
        [3, 4],
        [4],
        [6, None, 7],
        [],
    ]
    offsets: list[int | None] = [0, 0, -1, 0, 1, 7]
    lengths: list[int | None] = [1, 2, 1, 3, 5, 1]
    if with_nulls:
        lists[1] = None
        lists[3] = None
        offsets[4] = None
        lengths[2] = None
    lf = pl.LazyFrame(
        {
            "k": [5, 4, 3, 2, 1, 0],
            "a": lists,
            "o": pl.Series(offsets, dtype=pl.Int32),
            "l": pl.Series(lengths, dtype=pl.Int32),
        }
    ).sort("k")
    lf = lf.tail(3) if window == "tail" else lf.slice(1, 3)
    q = lf.select(
        pl.col("a").list.slice(1, 1).alias("literal"),
        pl.col("a").list.slice(pl.col("o"), pl.col("l")).alias("column"),
        pl.col("a").list.slice(pl.col("o")).alias("offset"),
        pl.col("a").list.tail(pl.col("l")).alias("tail"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Float64, pl.Boolean, pl.String])
@pytest.mark.parametrize("argument", ["offset", "length"])
def test_list_slice_non_integer_argument_not_supported(
    in_memory_engine: pl.GPUEngine, dtype: pl.DataType, argument: str
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [3]], "n": pl.Series([0, 1]).cast(dtype)})
    q = lf.select(
        pl.col("a").list.slice(pl.col("n"), 1)
        if argument == "offset"
        else pl.col("a").list.slice(0, pl.col("n"))
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_slice_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.slice(-2, 1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_slice_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.slice(0, 1))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
