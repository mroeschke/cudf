# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hive partition values attached to a file scan."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rmm.pylibrmm.stream import Stream

__all__ = ["HivePartitions"]


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class HivePartitions:
    """
    Hive partition values, one row per file in a scan.

    Polars parses hive keys out of the file paths and prunes the file list
    using any predicate it can evaluate against them. What reaches us is the
    surviving file list plus the partition values for those files, which are
    not stored in the parquet files themselves and so must be materialized
    onto the rows we read.

    Scanning a dataset written with ``partition_by=["cat", "part"]`` hands
    over a frame like this for the four surviving paths::

        shape: (4, 2)
        ┌──────┬─────┐
        │ part ┆ cat │
        │ ---  ┆ --- │
        │ i64  ┆ str │
        ╞══════╪═════╡
        │ 1    ┆ u   │
        │ 2    ┆ u   │
        │ 3    ┆ v   │
        │ 4    ┆ v   │
        └──────┴─────┘

    Row ``i`` holds the values for path ``i``, so the frame is as tall as the
    file list. The columns need not be in the order the keys appear in the
    paths, which are ``cat=u/part=1`` and so on here, so it is the names that
    tie a column to a key rather than its position.

    Parameters
    ----------
    df
        Partition values, as polars hands them over.
    """

    df: pl.DataFrame

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> HivePartitions | None:
        """
        Build from the ``hive_parts`` dataframe of a polars ``Scan`` node.

        Parameters
        ----------
        df
            One row per path in the scan, one column per hive key. Polars
            narrows the columns to those the query actually needs, so a
            zero-width frame means no hive columns are required.

        Returns
        -------
        The partition values, or ``None`` if no hive columns are needed.
        """
        if df.width == 0:
            return None
        return cls(df)

    @functools.cached_property
    def names(self) -> tuple[str, ...]:
        """Names of the hive columns."""
        return tuple(self.df.columns)

    @functools.cached_property
    def dtypes(self) -> tuple[DataType, ...]:
        """Datatype of each hive column."""
        return tuple(DataType(dtype) for dtype in self.df.dtypes)

    @property
    def num_paths(self) -> int:
        """Number of paths these partition values describe."""
        return self.df.height

    @functools.cached_property
    def is_uniform(self) -> bool:
        """Whether every path shares the same partition values."""
        return all(series.n_unique() <= 1 for series in self.df.iter_columns())

    def slice(self, start: int, stop: int) -> HivePartitions:
        """
        Restrict to the paths in ``range(start, stop)``.

        Parameters
        ----------
        start
            Index of the first path to keep.
        stop
            Index one past the last path to keep.

        Returns
        -------
        Partition values for the selected paths.
        """
        return type(self)(self.df.slice(start, stop - start))

    def broadcast(self, num_rows: int, *, stream: Stream) -> list[Column]:
        """
        Materialize the partition values as columns of ``num_rows`` equal rows.

        Only valid when :attr:`is_uniform` holds, since every output row is
        given the partition values of the first path.

        Parameters
        ----------
        num_rows
            Length of the returned columns.
        stream
            CUDA stream used for device memory operations and kernel launches.

        Returns
        -------
        One column per hive key.
        """
        return self._to_columns(
            plc.filling.repeat(
                self._value_table(self.df.head(1), stream=stream),
                num_rows,
                stream=stream,
            )
        )

    def repeat(self, rows_per_path: Sequence[int], *, stream: Stream) -> list[Column]:
        """
        Materialize the partition values by repeating each path's values.

        Used when no columns are read from the files at all, so there is no
        source index to gather with and the row counts have to come from the
        file metadata instead.

        Parameters
        ----------
        rows_per_path
            Number of output rows contributed by each path.
        stream
            CUDA stream used for device memory operations and kernel launches.

        Returns
        -------
        One column per hive key, of length ``sum(rows_per_path)``.
        """
        return self._to_columns(
            plc.filling.repeat(
                self._value_table(self.df, stream=stream),
                plc.Column.from_arrow(
                    pl.Series(values=rows_per_path, dtype=pl.Int32()), stream=stream
                ),
                stream=stream,
            )
        )

    def gather(self, source_index: plc.Column, *, stream: Stream) -> list[Column]:
        """
        Materialize the partition values by indexing them with a source index.

        Parameters
        ----------
        source_index
            Column giving, for each output row, the index of the path it was
            read from. This is what ``prepend_source_index_column`` produces,
            and unlike per-source row counts it stays valid when the reader
            applies a filter.
        stream
            CUDA stream used for device memory operations and kernel launches.

        Returns
        -------
        One column per hive key, aligned with ``source_index``.
        """
        return self._to_columns(
            plc.copying.gather(
                self._value_table(self.df, stream=stream),
                source_index,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            )
        )

    @staticmethod
    def _value_table(df: pl.DataFrame, *, stream: Stream) -> plc.Table:
        return plc.Table.from_arrow(df, stream=stream)

    def _to_columns(self, table: plc.Table) -> list[Column]:
        return [
            Column(column, name=name, dtype=dtype)
            for column, name, dtype in zip(
                table.columns(), self.names, self.dtypes, strict=True
            )
        ]

    @functools.cached_property
    def _key(self) -> tuple[Any, ...]:
        # Node digests fall back to repr() for anything that is not a tuple,
        # and polars leaves the middle out of a tall frame's repr, so identity
        # rests on something that spells out every value instead.
        return (tuple(self.df.schema.items()), tuple(self.df.iter_rows()))

    def __repr__(self) -> str:
        """Representation naming every partition value."""
        schema, rows = self._key
        return f"{type(self).__name__}(schema={dict(schema)!r}, values={rows!r})"

    def __hash__(self) -> int:
        """Hash of the partition values."""
        return hash(self._key)

    def __eq__(self, other: Any) -> bool:
        """Whether two sets of partition values agree, dtypes included."""
        return isinstance(other, HivePartitions) and self._key == other._key
