# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-level deletions of an Iceberg or Delta Lake scan."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.containers.datatype import DataType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.utils.lake import LakeScanOptions

__all__ = ["apply_deletions", "deletion_mask"]

BOOL = DataType(pl.Boolean())


def deletion_mask(
    paths: Sequence[str],
    rows_per_path: Sequence[int],
    lake_options: LakeScanOptions,
    *,
    stream: Stream,
) -> Column | None:
    """
    Build the mask of rows a lake scan keeps.

    Iceberg and Delta record deleted rows outside the data files, so a scan
    has to read every row and drop the deleted ones afterwards. The
    positions are physical, that is relative to the start of the data file
    they belong to, which is why the mask is assembled per path and then
    concatenated in scan order.

    Parameters
    ----------
    paths
        Data files of the scan, in scan order.
    rows_per_path
        Number of rows each path contributed to the frame.
    lake_options
        Options of the scan.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    Boolean column that is true for the rows to keep, or ``None`` when
    nothing is deleted.
    """
    masks = []
    deleted = False
    for index, (path, num_rows) in enumerate(zip(paths, rows_per_path, strict=True)):
        selection = lake_options.delta_selections.get(index)
        if selection is None:
            positions = _iceberg_positions(index, path, lake_options, stream=stream)
            mask = (
                _keep_all(num_rows, stream=stream)
                if positions is None
                else _mask_from_positions(positions, num_rows, stream=stream)
            )
            deleted |= positions is not None
        else:
            mask = _pad(
                plc.Column.from_arrow(selection.to_arrow(), stream=stream),
                num_rows,
                stream=stream,
            )
            deleted = True
        masks.append(mask)
    if not deleted:
        return None
    return Column(
        plc.concatenate.concatenate(masks, stream=stream)
        if len(masks) > 1
        else masks[0],
        dtype=BOOL,
    )


def apply_deletions(
    df: DataFrame,
    paths: Sequence[str],
    rows_per_path: Sequence[int],
    lake_options: LakeScanOptions,
    *,
    stream: Stream,
) -> DataFrame:
    """
    Drop the rows an Iceberg or Delta scan deletes.

    Parameters
    ----------
    df
        Frame holding the rows of ``paths``, in scan order.
    paths
        Data files of the scan, in scan order.
    rows_per_path
        Number of rows each path contributed to ``df``.
    lake_options
        Options of the scan.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    The frame without the deleted rows.

    Notes
    -----
    A frame of no columns only tracks how many rows it has, so there is
    nothing for a filter to act on and the kept rows have to be counted.
    """
    mask = deletion_mask(paths, rows_per_path, lake_options, stream=stream)
    if mask is None:
        return df
    if not df.columns:
        return DataFrame([], num_rows=_count_kept(mask, stream=stream), stream=stream)
    return df.filter(mask)


def _count_kept(mask: Column, *, stream: Stream) -> int:
    return plc.stream_compaction.apply_retention_mask(
        plc.Table([mask.obj]), mask.obj, stream=stream
    ).num_rows()


def _keep_all(num_rows: int, *, stream: Stream) -> plc.Column:
    return plc.Column.from_scalar(
        plc.Scalar.from_py(
            True,  # noqa: FBT003
            BOOL.plc_type,
            stream=stream,
        ),
        num_rows,
        stream=stream,
    )


def _pad(mask: plc.Column, num_rows: int, *, stream: Stream) -> plc.Column:
    """Match a mask to the rows of its file, keeping any rows it does not cover."""
    if mask.size() > num_rows:  # pragma: no cover; polars never sends a longer mask
        (mask,) = plc.copying.slice(mask, [0, num_rows], stream=stream)
        return mask
    if mask.size() < num_rows:
        return plc.concatenate.concatenate(
            [mask, _keep_all(num_rows - mask.size(), stream=stream)], stream=stream
        )
    return mask


def _mask_from_positions(
    positions: plc.Column, num_rows: int, *, stream: Stream
) -> plc.Column:
    """Turn deleted row positions into a keep-mask of ``num_rows`` rows."""
    positions = plc.unary.cast(
        positions, plc.DataType(plc.types.SIZE_TYPE_ID), stream=stream
    )
    (mask,) = plc.copying.scatter(
        [
            plc.Scalar.from_py(
                False,  # noqa: FBT003
                BOOL.plc_type,
                stream=stream,
            )
        ],
        positions,
        plc.Table([_keep_all(num_rows, stream=stream)]),
        stream=stream,
    ).columns()
    return mask


def _iceberg_positions(
    index: int, path: str, lake_options: LakeScanOptions, *, stream: Stream
) -> plc.Column | None:
    """Positions of the rows deleted from one Iceberg data file."""
    delete_files = lake_options.position_deletes.get(index)
    if delete_files is not None:
        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(list(delete_files))
        ).build()
        options.set_column_names(["pos"])
        (positions,) = plc.io.parquet.read_parquet(options, stream=stream).tbl.columns()
        return positions
    puffin = lake_options.deletion_vectors.get(index)
    if puffin is None:
        return None
    from polars.io.iceberg._utils import load_puffin_deletion_file

    deletions = load_puffin_deletion_file(Path(puffin).read_bytes())
    positions_series = deletions.get(path)
    if positions_series is None:  # pragma: no cover; polars keys on the scan path
        return None
    return plc.Column.from_arrow(positions_series.to_arrow(), stream=stream)
