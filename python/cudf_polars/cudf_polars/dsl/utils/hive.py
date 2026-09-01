# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hive partition values attached to a file scan."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rmm.pylibrmm.stream import Stream

__all__ = ["HivePartitions"]


@dataclasses.dataclass(frozen=True, slots=True)
class HivePartitions:
    """
    Hive partition values, one row per file in a scan.

    Polars parses hive keys out of the file paths and prunes the file list
    using any predicate it can evaluate against them. What reaches us is the
    surviving file list plus the partition values for those files, which are
    not stored in the parquet files themselves and so must be materialized
    onto the rows we read.

    Parameters
    ----------
    names
        Names of the hive columns.
    dtypes
        Datatype of each hive column.
    values
        Column-major partition values: ``values[i][j]`` is the value of
        ``names[i]`` for the ``j``th path of the scan.
    """

    names: tuple[str, ...]
    dtypes: tuple[DataType, ...]
    values: tuple[tuple[Any, ...], ...]

    def __post_init__(self) -> None:
        """Validate that the field lengths agree."""
        if not (len(self.names) == len(self.dtypes) == len(self.values)):
            raise ValueError("HivePartitions fields must have matching lengths")
        if len(set(map(len, self.values))) > 1:
            raise ValueError("HivePartitions values must have matching lengths")

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
        return cls(
            tuple(df.columns),
            tuple(DataType(dtype) for dtype in df.dtypes),
            tuple(tuple(series.to_list()) for series in df.iter_columns()),
        )

    @property
    def num_paths(self) -> int:
        """Number of paths these partition values describe."""
        return len(self.values[0])

    @property
    def is_uniform(self) -> bool:
        """Whether every path shares the same partition values."""
        return all(len(set(value)) <= 1 for value in self.values)

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
        return dataclasses.replace(
            self, values=tuple(value[start:stop] for value in self.values)
        )

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
        return [
            Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(value[0], dtype.plc_type, stream=stream),
                    num_rows,
                    stream=stream,
                ),
                name=name,
                dtype=dtype,
            )
            for name, dtype, value in zip(
                self.names, self.dtypes, self.values, strict=True
            )
        ]

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
        repeated = plc.filling.repeat(
            self._value_table(stream=stream),
            plc.Column.from_arrow(
                pl.Series(values=rows_per_path, dtype=pl.Int32()), stream=stream
            ),
            stream=stream,
        )
        return self._to_columns(repeated)

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
                self._value_table(stream=stream),
                source_index,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            )
        )

    def _value_table(self, *, stream: Stream) -> plc.Table:
        return plc.Table(
            [
                plc.Column.from_arrow(
                    pl.Series(values=value, dtype=dtype.polars_type), stream=stream
                )
                for dtype, value in zip(self.dtypes, self.values, strict=True)
            ]
        )

    def _to_columns(self, table: plc.Table) -> list[Column]:
        return [
            Column(column, name=name, dtype=dtype)
            for column, name, dtype in zip(
                table.columns(), self.names, self.dtypes, strict=True
            )
        ]
