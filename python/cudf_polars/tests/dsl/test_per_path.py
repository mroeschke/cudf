# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.dsl.utils.per_path import PerPathValues


@pytest.fixture
def per_path() -> PerPathValues:
    return PerPathValues(pl.DataFrame({"part": [1, 2, 3], "cat": ["u", "u", "v"]}))


def test_equal_values_are_equal(per_path: PerPathValues) -> None:
    other = PerPathValues(pl.DataFrame({"part": [1, 2, 3], "cat": ["u", "u", "v"]}))
    assert per_path == other
    assert hash(per_path) == hash(other)


def test_dtypes_distinguish_identical_values() -> None:
    values = {"part": [1, 2]}
    narrow = PerPathValues(pl.DataFrame(values, schema={"part": pl.Int32}))
    wide = PerPathValues(pl.DataFrame(values, schema={"part": pl.Int64}))
    assert narrow != wide
    assert hash(narrow) != hash(wide)


def test_names_distinguish_identical_values() -> None:
    # hash_rows digests the values but not the column they sit in, so the
    # schema is what tells these two apart.
    values = [1, 2]
    part = PerPathValues(pl.DataFrame({"part": values}))
    cat = PerPathValues(pl.DataFrame({"cat": values}))
    assert part != cat
    assert hash(part) != hash(cat)


def test_path_order_matters() -> None:
    forwards = PerPathValues(pl.DataFrame({"part": [1, 2]}))
    backwards = PerPathValues(pl.DataFrame({"part": [2, 1]}))
    assert forwards != backwards
    assert hash(forwards) != hash(backwards)


def test_tall_partitions_are_distinguished() -> None:
    # Polars elides the middle of a tall frame's repr, so identity cannot
    # rest on it. hash_rows sees every row.
    tall = PerPathValues(pl.DataFrame({"part": range(40)}))
    other = PerPathValues(pl.DataFrame({"part": [*range(20), 999, *range(21, 40)]}))
    assert repr(tall.df) == repr(other.df)
    assert tall != other
    assert hash(tall) != hash(other)


def test_not_equal_to_other_types(per_path: PerPathValues) -> None:
    assert per_path != per_path.df


def test_repr(per_path: PerPathValues) -> None:
    assert repr(per_path) == f"PerPathValues(df={per_path.df!r})"
