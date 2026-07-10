# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
