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
