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
        (-(2**63), None),
        (-(2**63), -1),
    ],
)
def test_list_slice_literal(
    engine: pl.GPUEngine, offset: int, length: int | None
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.slice(offset, length))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [0, 2, 10, -1, -(2**63)])
def test_list_slice_head_tail(engine: pl.GPUEngine, n: int) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(
        pl.col("a").list.head(n).alias("head"), pl.col("a").list.tail(n).alias("tail")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype,minimum",
    [
        (pl.Int8, -(2**7)),
        (pl.UInt8, 0),
        (pl.Int64, -(2**63)),
        pytest.param(
            pl.UInt64,
            0,
            marks=pytest.mark.skipif(
                POLARS_VERSION_LT_21, reason="UInt64 offsets fall back on Polars 2.0"
            ),
        ),
    ],
)
def test_list_slice_column(
    engine: pl.GPUEngine, dtype: pl.DataType, minimum: int
) -> None:
    signed = dtype.is_signed_integer()
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5]],
            "o": pl.Series(
                [minimum, -1, 0, 2, -3] if signed else [1, 5, 0, 2, 0], dtype=dtype
            ),
            "l": pl.Series(
                [-1, -1, 3, 1, 1] if signed else [2, 1, 3, 1, 0], dtype=dtype
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
    lf = pl.LazyFrame({"a": [[1, 2], [3]], "n": pl.Series([None, 1], dtype=pl.Int64)})
    q = lf.select(pl.col("a").list.slice(pl.col("n").first(), 1))
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


@pytest.mark.parametrize("offset", ["literal", "column", "series"])
@pytest.mark.parametrize("length", ["literal", "column", "series"])
def test_list_slice_broadcast(engine: pl.GPUEngine, offset: str, length: str) -> None:
    if {offset, length} == {"column", "series"}:
        pytest.skip("Series literals are not split across ranks like columns")
    if POLARS_VERSION_LT_21 and "column" in (offset, length):
        pytest.skip("Polars 2.0 raises for null offsets and ignores null lengths")
    lf = pl.LazyFrame({"o": [1, None, -1, 5], "l": [1, None, -1, 2]})
    offsets = {
        "literal": pl.lit(-2),
        "column": pl.col("o"),
        "series": pl.lit(pl.Series([0, 1, 2, -1])),
    }
    lengths = {
        "literal": pl.lit(2),
        "column": pl.col("l"),
        "series": pl.lit(pl.Series([3, 2, 1, -1])),
    }
    q = lf.select(
        pl.lit([1, 2, 3]).list.slice(offsets[offset], lengths[length]).alias("literal"),
        pl.lit(None, dtype=pl.List(pl.Int64))
        .list.slice(offsets[offset], lengths[length])
        .alias("null"),
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
        pl.col("a").list.tail(pl.lit(1, pl.UInt32)).alias("tail"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize("offset", [0, "column"])
@pytest.mark.parametrize("length", [1, "column"])
def test_list_slice_empty_or_all_null(
    engine: pl.GPUEngine, rows: int, offset: int | str, length: int | str
) -> None:
    if POLARS_VERSION_LT_21 and "column" in (offset, length):
        pytest.skip("Polars 2.0 returns List(Null) for an empty or all-null column")
    lf = pl.LazyFrame(
        {"a": [None] * rows, "n": [0] * rows},
        schema={"a": pl.List(pl.String), "n": pl.Int64},
    )
    q = lf.select(
        pl.col("a").list.slice(
            pl.col("n") if offset == "column" else offset,
            pl.col("n") if length == "column" else length,
        )
    )
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


def test_list_head_default(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 3, 4, 5, 6, 7], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.head())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n,dtype",
    [(0, pl.UInt32), (2, pl.UInt32), (10, pl.UInt32), (-1, pl.Int16), (-10, pl.Int16)],
)
def test_list_head_typed_literal(
    engine: pl.GPUEngine, n: int, dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.head(pl.lit(n, dtype)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21, reason="Polars 2.0 in-memory slices to null for null lengths"
)
@pytest.mark.parametrize("dtype", [pl.Int16, pl.Int64, pl.UInt32, pl.UInt64])
def test_list_head_column(engine: pl.GPUEngine, dtype: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5], [6, 7]],
            "n": pl.Series(
                [2, None, 1, 0, 5, -1 if dtype.is_signed_integer() else 0],
                dtype=dtype,
            ),
        }
    )
    q = lf.select(pl.col("a").list.head(pl.col("n")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_21, reason="Polars 2.0 in-memory slices to null for null lengths"
)
@pytest.mark.parametrize("inner", ["string", "list", "struct"])
def test_list_head_nested_column(engine: pl.GPUEngine, inner: str) -> None:
    data: list[object]
    if inner == "string":
        data = [["x", "y", None], None, [], ["z"]]
    elif inner == "list":
        data = [[[1], None, [2, 3]], None, [], [[4]]]
    else:
        data = [[{"x": 1}, None, {"x": 2}], None, [], [{"x": None}]]
    lf = pl.LazyFrame({"a": data, "n": pl.Series([2, 1, None, 0], dtype=pl.UInt8)})
    q = lf.select(pl.col("a").list.head(pl.col("n")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [2, "column"])
def test_list_head_zero_rows(engine: pl.GPUEngine, n: int | str) -> None:
    if POLARS_VERSION_LT_21 and n == "column":
        pytest.skip("Polars 2.0 returns List(Null) for an empty or all-null column")
    lf = pl.LazyFrame(schema={"a": pl.List(pl.Int64), "n": pl.UInt32})
    q = lf.select(pl.col("a").list.head(pl.col("n") if n == "column" else n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("window", ["slice", "tail"])
def test_list_head_sliced(
    engine: pl.GPUEngine, window: str, *, with_nulls: bool
) -> None:
    lf = pl.LazyFrame(
        {
            "k": [3, 2, 1, 0],
            "a": [[3, 4], None if with_nulls else [5, 6], [2], [1]],
            "n": pl.Series([1, None if with_nulls else 2, 3, 0], dtype=pl.Int32),
        }
    ).sort("k")
    lf = lf.tail(3) if window == "tail" else lf.slice(1, 3)
    q = lf.select(
        pl.col("a").list.head(1).alias("literal"),
        pl.col("a").list.head(pl.col("n")).alias("column"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_head_implode_over(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"type": ["water", "fire", "water", "earth"]})
    q = lf.select(pl.col("type").implode().list.head(1).over("type"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_head_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.head())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_tail_default(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 3, 4, 5, 6, 7], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.tail())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n,dtype",
    [
        (0, pl.UInt32),
        (2, pl.UInt32),
        (10, pl.UInt32),
        (-1, pl.Int16),
        (-10, pl.Int16),
        pytest.param(
            2**64 - 1,
            pl.UInt64,
            marks=pytest.mark.skipif(
                POLARS_VERSION_LT_21, reason="Polars 2.0 raises for null offsets"
            ),
        ),
    ],
)
def test_list_tail_typed_literal(
    engine: pl.GPUEngine, n: int, dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame({"a": [[1, 2, 2], [None, 3], [], None, [5]]})
    q = lf.select(pl.col("a").list.tail(pl.lit(n, dtype)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(POLARS_VERSION_LT_21, reason="Polars 2.0 raises for null offsets")
@pytest.mark.parametrize(
    "dtype,extreme",
    [
        (pl.Int16, -1),
        (pl.Int64, -(2**63)),
        (pl.UInt32, 2**32 - 1),
        (pl.UInt64, 2**64 - 1),
    ],
)
def test_list_tail_column(
    engine: pl.GPUEngine, dtype: pl.DataType, extreme: int
) -> None:
    lf = pl.LazyFrame(
        {
            "a": [[1, 2, 2], [None, 3], [], None, [5], [6, 7]],
            "n": pl.Series([2, None, 1, 0, 5, extreme], dtype=dtype),
        }
    )
    q = lf.select(pl.col("a").list.tail(pl.col("n")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(POLARS_VERSION_LT_21, reason="Polars 2.0 raises for null offsets")
@pytest.mark.parametrize("inner", ["string", "list", "struct"])
def test_list_tail_nested_column(engine: pl.GPUEngine, inner: str) -> None:
    data: list[object]
    if inner == "string":
        data = [["x", "y", None], None, [], ["z"]]
    elif inner == "list":
        data = [[[1], None, [2, 3]], None, [], [[4]]]
    else:
        data = [[{"x": 1}, None, {"x": 2}], None, [], [{"x": None}]]
    lf = pl.LazyFrame({"a": data, "n": pl.Series([2, 1, None, 0], dtype=pl.UInt8)})
    q = lf.select(pl.col("a").list.tail(pl.col("n")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [2, "column"])
def test_list_tail_zero_rows(engine: pl.GPUEngine, n: int | str) -> None:
    if POLARS_VERSION_LT_21 and n == "column":
        pytest.skip("Polars 2.0 returns List(Null) for an empty or all-null column")
    lf = pl.LazyFrame(schema={"a": pl.List(pl.Int64), "n": pl.UInt32})
    q = lf.select(pl.col("a").list.tail(pl.col("n") if n == "column" else n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("window", ["slice", "tail"])
def test_list_tail_sliced(
    engine: pl.GPUEngine, window: str, *, with_nulls: bool
) -> None:
    if with_nulls and POLARS_VERSION_LT_21:
        pytest.skip("Polars 2.0 raises for null offsets")
    lf = pl.LazyFrame(
        {
            "k": [3, 2, 1, 0],
            "a": [[3, 4], None if with_nulls else [5, 6], [2], [1]],
            "n": pl.Series([1, None if with_nulls else 2, 3, 0], dtype=pl.Int32),
        }
    ).sort("k")
    lf = lf.tail(3) if window == "tail" else lf.slice(1, 3)
    q = lf.select(
        pl.col("a").list.tail(1).alias("literal"),
        pl.col("a").list.tail(pl.col("n")).alias("column"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_list_tail_implode_over(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"type": ["water", "fire", "water", "earth"]})
    q = lf.select(pl.col("type").implode().list.tail(1).over("type"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_tail_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.tail())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
