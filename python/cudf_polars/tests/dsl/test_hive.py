# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.utils.hive import HivePartitions
from cudf_polars.utils.cuda_stream import get_cuda_stream


@pytest.fixture
def partitions() -> HivePartitions:
    return HivePartitions(
        names=("part", "cat"),
        dtypes=(DataType(pl.Int64()), DataType(pl.String())),
        values=((1, 2, 3), ("u", "u", "v")),
    )


def test_from_polars() -> None:
    got = HivePartitions.from_polars(
        pl.DataFrame(
            {"part": [1, 2], "cat": ["u", "v"]},
            schema={"part": pl.Int32, "cat": pl.String},
        )
    )
    assert got == HivePartitions(
        names=("part", "cat"),
        dtypes=(DataType(pl.Int32()), DataType(pl.String())),
        values=((1, 2), ("u", "v")),
    )


def test_from_polars_zero_width() -> None:
    assert HivePartitions.from_polars(pl.DataFrame(height=3)) is None


def test_hashable(partitions: HivePartitions) -> None:
    assert hash(partitions) == hash(
        HivePartitions(
            names=("part", "cat"),
            dtypes=(DataType(pl.Int64()), DataType(pl.String())),
            values=((1, 2, 3), ("u", "u", "v")),
        )
    )


@pytest.mark.parametrize(
    "names,dtypes,values",
    [
        (("a",), (DataType(pl.Int64()), DataType(pl.Int64())), ((1,),)),
        (("a", "b"), (DataType(pl.Int64()), DataType(pl.Int64())), ((1,), (1, 2))),
    ],
)
def test_mismatched_lengths_raise(names, dtypes, values) -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        HivePartitions(names=names, dtypes=dtypes, values=values)


def test_num_paths(partitions: HivePartitions) -> None:
    assert partitions.num_paths == 3


@pytest.mark.parametrize(
    "values,expected",
    [
        (((1, 2, 3), ("u", "u", "v")), False),
        (((1, 1, 1), ("u", "u", "v")), False),
        (((1, 1, 1), ("u", "u", "u")), True),
        (((1,), ("u",)), True),
    ],
)
def test_is_uniform(values, expected) -> None:
    partitions = HivePartitions(
        names=("part", "cat"),
        dtypes=(DataType(pl.Int64()), DataType(pl.String())),
        values=values,
    )
    assert partitions.is_uniform is expected


def test_slice(partitions: HivePartitions) -> None:
    assert partitions.slice(1, 3) == HivePartitions(
        names=("part", "cat"),
        dtypes=(DataType(pl.Int64()), DataType(pl.String())),
        values=((2, 3), ("u", "v")),
    )


def test_broadcast() -> None:
    stream = get_cuda_stream()
    partitions = HivePartitions(
        names=("part", "cat"),
        dtypes=(DataType(pl.Int64()), DataType(pl.String())),
        values=((7,), ("u",)),
    )
    got = DataFrame(partitions.broadcast(3, stream=stream), stream=stream)
    assert_frame_equal(
        got.to_polars(), pl.DataFrame({"part": [7, 7, 7], "cat": ["u", "u", "u"]})
    )


def test_repeat(partitions: HivePartitions) -> None:
    stream = get_cuda_stream()
    got = DataFrame(partitions.repeat([2, 0, 1], stream=stream), stream=stream)
    assert_frame_equal(
        got.to_polars(),
        pl.DataFrame({"part": [1, 1, 3], "cat": ["u", "u", "v"]}),
    )


def test_repeat_empty(partitions: HivePartitions) -> None:
    stream = get_cuda_stream()
    got = DataFrame(partitions.repeat([0, 0, 0], stream=stream), stream=stream)
    assert got.num_rows == 0


def test_gather(partitions: HivePartitions) -> None:
    stream = get_cuda_stream()
    source_index = plc.Column.from_arrow(
        pl.Series(values=[0, 0, 2, 1], dtype=pl.Int32()), stream=stream
    )
    got = DataFrame(partitions.gather(source_index, stream=stream), stream=stream)
    assert_frame_equal(
        got.to_polars(),
        pl.DataFrame({"part": [1, 1, 3, 2], "cat": ["u", "u", "v", "u"]}),
    )


def test_gather_preserves_nulls() -> None:
    stream = get_cuda_stream()
    partitions = HivePartitions(
        names=("part",),
        dtypes=(DataType(pl.Int64()),),
        values=((1, None),),
    )
    source_index = plc.Column.from_arrow(
        pl.Series(values=[1, 0], dtype=pl.Int32()), stream=stream
    )
    got = DataFrame(partitions.gather(source_index, stream=stream), stream=stream)
    assert_frame_equal(got.to_polars(), pl.DataFrame({"part": [None, 1]}))
