# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Options that ``scan_iceberg`` and ``scan_delta`` attach to a parquet scan."""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.utils.io import CachedParquetInfo
    from cudf_polars.typing import Schema

__all__ = ["IcebergColumn", "LakeScanOptions", "read_lake_files"]


@dataclasses.dataclass(frozen=True, slots=True)
class IcebergColumn:
    """
    A column of an Iceberg schema, identified by its physical field ID.

    Parameters
    ----------
    name
        Name the column has in the table schema, which may differ from the
        name recorded in any given data file.
    physical_id
        Field ID written into the parquet footer for this column.
    children
        Fields of a nested column, empty for a primitive column.
    """

    name: str
    physical_id: int
    children: tuple[IcebergColumn, ...]

    @classmethod
    def from_polars(cls, spec: tuple[str, int, tuple[Any, ...]]) -> IcebergColumn:
        """
        Build from the nested tuples polars uses for an Iceberg column.

        Parameters
        ----------
        spec
            ``(name, physical_id, type_)`` where ``type_`` is one of:
                * ``("primitive", dtype)``
                * ``("list", column)``
                * ``("fixed-size-list", column, width)``
                * ``("map", key, value)``
                * ``("struct", {physical_id: column})``

        Returns
        -------
        The parsed column.
        """
        name, physical_id, type_ = spec
        kind, *rest = type_
        if kind == "primitive":
            children: tuple[IcebergColumn, ...] = ()
        elif kind == "struct":
            children = tuple(cls.from_polars(child) for child in rest[0].values())
        elif kind in ("list", "fixed-size-list", "map"):
            children = tuple(
                cls.from_polars(child) for child in rest if isinstance(child, tuple)
            )
        else:  # pragma: no cover; polars only produces the kinds above
            raise NotImplementedError(f"Unhandled Iceberg column type {kind!r}")
        return cls(name=name, physical_id=physical_id, children=children)


@dataclasses.dataclass(frozen=True)
class LakeScanOptions:
    """
    The Iceberg and Delta Lake specific parts of a polars ``Scan``.

    Parameters
    ----------
    columns
        Top-level columns of the Iceberg schema, in table order, or ``None``
        when the scan reads columns by name (Delta, and Iceberg tables that
        never evolved).
    position_deletes
        Maps the index of a path in the scan to the Iceberg position delete
        files that apply to it.
    deletion_vectors
        Maps the index of a path in the scan to the Iceberg puffin file
        holding its deletion vector.
    delta_selections
        Maps the index of a path in the scan to the Delta deletion vector
        that applies to it, as a boolean series that is true for the rows
        to keep.
    partition_values
        Maps a physical field ID to the value its identity-transformed
        partition field takes, one row per path.
    initial_defaults
        Maps a physical field ID to the single-element series holding the
        Iceberg V3 initial default for a column absent from a file.
    missing_columns_policy
        ``"insert"`` to fill columns absent from a file, ``"raise"`` to fail.
    extra_columns_policy
        ``"ignore"`` to drop columns a file has but the schema does not,
        ``"raise"`` to fail.
    cast_columns_policy
        Which casts from the physical to the table type are permitted.
    row_count
        ``(total_rows, deleted_rows)`` for the whole scan when the table
        metadata could supply it.
    """

    columns: tuple[IcebergColumn, ...] | None
    position_deletes: Mapping[int, tuple[str, ...]]
    deletion_vectors: Mapping[int, str]
    delta_selections: Mapping[int, pl.Series]
    partition_values: Mapping[int, pl.Series]
    initial_defaults: Mapping[int, pl.Series]
    missing_columns_policy: str
    extra_columns_policy: str
    cast_columns_policy: Mapping[str, Any]
    row_count: tuple[int, int] | None

    def __post_init__(self) -> None:  # noqa: D105
        if self.columns is not None and any(column.children for column in self.columns):
            raise NotImplementedError("Iceberg column mapping of nested columns")
        if self.extra_columns_policy not in ("ignore", "raise"):
            raise NotImplementedError(  # pragma: no cover; only two policies exist
                f"Extra columns policy {self.extra_columns_policy!r}"
            )
        if self.missing_columns_policy not in ("insert", "raise"):
            raise NotImplementedError(  # pragma: no cover; only two policies exist
                f"Missing columns policy {self.missing_columns_policy!r}"
            )
        unknown_defaults = set(self.initial_defaults) - {
            column.physical_id for column in self.columns or ()
        }
        if unknown_defaults:  # pragma: no cover; polars keys defaults by schema id
            raise NotImplementedError(
                f"Iceberg defaults for unmapped fields {sorted(unknown_defaults)}"
            )

    @classmethod
    def from_file_options(
        cls, file_options: Any, paths: Sequence[str]
    ) -> LakeScanOptions | None:
        """
        Build from the ``file_options`` of a polars ``Scan`` node.

        Parameters
        ----------
        file_options
            The ``PyFileOptions`` of the scan.
        paths
            Data files of the scan, named as polars names them.

        Returns
        -------
        The parsed options, or ``None`` for a scan that is not a lake scan.

        Raises
        ------
        NotImplementedError
            If polars supplies a variant we do not understand. Failing here
            keeps us from silently ignoring information that changes the
            result.

        Notes
        -----
        A Delta scan carries no column mapping, deletions or defaults, but a
        table that has evolved still has files that are missing a column of
        the schema, which a plain parquet read cannot express. Such a scan is
        recognised by its missing columns policy.

        A Delta deletion vector arrives as a callback holding a handle to the
        table, which cannot be sent to another process, so it is resolved
        here rather than in the tasks that read the files.
        """
        column_mapping = file_options.column_mapping
        columns: tuple[IcebergColumn, ...] | None = None
        if column_mapping is not None:
            mapping_kind, mapping = column_mapping
            if (
                mapping_kind != "iceberg-column-mapping"
            ):  # pragma: no cover; only kind polars emits
                raise NotImplementedError(f"Unhandled column mapping {mapping_kind!r}")
            columns = tuple(
                IcebergColumn.from_polars(spec) for spec in mapping.values()
            )

        position_deletes: dict[int, tuple[str, ...]] = {}
        deletion_vectors: dict[int, str] = {}
        delta_selections: dict[int, pl.Series] = {}
        if file_options.deletion_files is not None:
            deletion_kind, deletion_payload = file_options.deletion_files
            if deletion_kind == "iceberg":
                raw_position_deletes, raw_deletion_vectors = deletion_payload
                position_deletes = {
                    index: tuple(delete_files)
                    for index, delete_files in raw_position_deletes.items()
                }
                deletion_vectors = dict(raw_deletion_vectors)
            elif deletion_kind == "delta-deletion-vector":
                frame = deletion_payload(pl.DataFrame({"path": list(paths)}))
                if (
                    frame is None
                ):  # pragma: no cover; the polars callback returns a frame
                    delta_selections = {}
                else:
                    delta_selections = {
                        index: selection
                        for index, selection in enumerate(
                            frame.get_column("selection_vector")
                        )
                        if selection is not None
                    }
            else:
                raise NotImplementedError(  # pragma: no cover; only kinds polars emits
                    f"Unhandled deletion files {deletion_kind!r}"
                )

        partition_values: dict[int, pl.Series] = {}
        initial_defaults: dict[int, pl.Series] = {}
        if file_options.default_values is not None:
            defaults_kind, defaults_payload = file_options.default_values
            if defaults_kind != "iceberg":  # pragma: no cover; only kind polars emits
                raise NotImplementedError(f"Unhandled default values {defaults_kind!r}")
            partition_fields, raw_initial_defaults = defaults_payload
            for physical_id, value in partition_fields.items():
                if isinstance(value, str):
                    raise NotImplementedError(
                        f"Iceberg partition field {physical_id}: {value}"
                    )
                partition_values[physical_id] = pl.Series._from_pyseries(value)
            initial_defaults = {
                physical_id: pl.Series._from_pyseries(value)
                for physical_id, value in raw_initial_defaults.items()
            }
        if (
            columns is None
            and not position_deletes
            and not deletion_vectors
            and not delta_selections
            and not partition_values
            and not initial_defaults
            and file_options.missing_columns_policy == "raise"
        ):
            return None
        return cls(
            columns=columns,
            position_deletes=position_deletes,
            deletion_vectors=deletion_vectors,
            delta_selections=delta_selections,
            partition_values=partition_values,
            initial_defaults=initial_defaults,
            missing_columns_policy=file_options.missing_columns_policy,
            extra_columns_policy=file_options.extra_columns_policy,
            cast_columns_policy=file_options.cast_columns_policy,
            row_count=file_options.row_count,
        )

    @property
    def has_deletions(self) -> bool:
        """Whether any rows have to be dropped after reading."""
        return bool(
            self.position_deletes or self.deletion_vectors or self.delta_selections
        )

    def slice(self, start: int, stop: int) -> LakeScanOptions:
        """
        Restrict to the paths in ``range(start, stop)``.

        Deletions and partition values are keyed by the position of a path
        within the scan, so both have to be re-keyed when a task takes only
        part of the paths.

        Parameters
        ----------
        start
            Index of the first path to keep.
        stop
            Index one past the last path to keep.

        Returns
        -------
        Options covering the selected paths.
        """
        return dataclasses.replace(
            self,
            position_deletes={
                i - start: v
                for i, v in self.position_deletes.items()
                if start <= i < stop
            },
            deletion_vectors={
                i - start: v
                for i, v in self.deletion_vectors.items()
                if start <= i < stop
            },
            delta_selections={
                i - start: v
                for i, v in self.delta_selections.items()
                if start <= i < stop
            },
            partition_values={
                physical_id: series.slice(start, stop - start)
                for physical_id, series in self.partition_values.items()
            },
            row_count=None,
        )

    def __hash__(self) -> int:
        """Hash of every option."""
        return hash(
            (
                self.columns,
                tuple(sorted(self.position_deletes.items())),
                tuple(sorted(self.deletion_vectors.items())),
                tuple(sorted(self.delta_selections)),
                tuple(sorted(self.partition_values)),
                tuple(sorted(self.initial_defaults)),
                self.missing_columns_policy,
                self.extra_columns_policy,
                self.row_count,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """Whether two sets of options agree."""
        return (
            isinstance(other, LakeScanOptions)
            and self.columns == other.columns
            and self.position_deletes == other.position_deletes
            and self.deletion_vectors == other.deletion_vectors
            and _series_mapping_eq(self.delta_selections, other.delta_selections)
            and _series_mapping_eq(self.partition_values, other.partition_values)
            and _series_mapping_eq(self.initial_defaults, other.initial_defaults)
            and self.missing_columns_policy == other.missing_columns_policy
            and self.extra_columns_policy == other.extra_columns_policy
            and self.cast_columns_policy == other.cast_columns_policy
            and self.row_count == other.row_count
        )


def _series_mapping_eq(
    left: Mapping[int, pl.Series], right: Mapping[int, pl.Series]
) -> bool:
    return left.keys() == right.keys() and all(
        left_series.equals(right_series)
        for left_series, right_series in zip(left.values(), right.values(), strict=True)
    )


def read_lake_files(
    paths: Sequence[str],
    lake_options: LakeScanOptions,
    schema: Schema,
    with_columns: Sequence[str] | None,
    cached_parquet_info: Sequence[CachedParquetInfo] | None = None,
    *,
    stream: Stream,
) -> tuple[DataFrame, list[int]]:
    """
    Read the data files of an Iceberg or Delta scan onto the table schema.

    The files of one table need not agree with each other or with the table
    schema: a column can have been renamed, added or dropped since a file
    was written. Iceberg tracks a column through all of that by the field ID
    in the parquet footer, so the columns of an Iceberg file are matched on
    that rather than on their name; Delta has no such ID and is matched on
    name.

    Files are read in runs that hold the same columns, since the parquet
    reader requires every source of one read to agree. Each run is then
    renamed and cast to the table schema, and columns the files do not hold
    are filled in.

    Parameters
    ----------
    paths
        Data files of the scan, in scan order.
    lake_options
        Options of the scan.
    schema
        Schema the scan must produce.
    with_columns
        Columns to project, or ``None`` for all of them.
    cached_parquet_info
        Footers already read for ``paths``, in the same order, or ``None`` to
        read them here.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    The concatenated frame and the number of rows each path contributed.
    """
    by_field_id = lake_options.columns is not None
    projected = [
        name for name in schema if with_columns is None or name in with_columns
    ]
    wanted: dict[Any, str]
    if lake_options.columns is not None:
        wanted = {
            column.physical_id: column.name
            for column in lake_options.columns
            if column.name in projected
            and column.physical_id not in lake_options.partition_values
        }
    else:
        wanted = {name: name for name in projected}

    metadatas: Sequence[plc.io.parquet_metadata.FileMetaData | None] = (
        [None] * len(paths)
        if cached_parquet_info is None
        else [info.file_metadata for info in cached_parquet_info]
    )
    footers = [
        _footer(path, metadata, by_field_id=by_field_id)
        for path, metadata in zip(paths, metadatas, strict=True)
    ]
    rows_per_path = [footer.num_rows for footer in footers]

    frames: list[DataFrame] = []
    start = 0
    for keys, group in itertools.groupby(footers, key=lambda footer: footer.keys):
        stop = start + len(list(group))
        frames.append(
            _read_group(
                paths[start:stop],
                [key for key in wanted if key in frozenset(keys)],
                footers[start],
                sum(rows_per_path[start:stop]),
                wanted,
                schema,
                lake_options,
                by_field_id=by_field_id,
                stream=stream,
            )
        )
        start = stop

    if len(frames) == 1:
        return frames[0], rows_per_path
    names = list(frames[0].column_map)
    if not names:
        return DataFrame([], num_rows=sum(rows_per_path), stream=stream), rows_per_path
    return (
        DataFrame.from_table(
            plc.concatenate.concatenate(
                [
                    plc.Table([frame.column_map[name].obj for name in names])
                    for frame in frames
                ],
                stream=stream,
            ),
            names,
            [schema[name] for name in names],
            stream=stream,
        ),
        rows_per_path,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Footer:
    """What the parquet footer of one data file says about its columns."""

    names: tuple[str, ...]
    keys: tuple[Any, ...]
    num_rows: int

    def key_of(self, name: str) -> Any:
        """The key the read of a column named ``name`` maps onto."""
        return dict(zip(self.names, self.keys, strict=True))[name]


def _footer(
    path: str,
    metadata: plc.io.parquet_metadata.FileMetaData | None,
    *,
    by_field_id: bool,
) -> _Footer:
    """Top-level columns and row count of a parquet file."""
    if metadata is None:
        metadata = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.SourceInfo([path])
        )[0]
    elements = iter(metadata.schema)
    root = next(elements)
    names = []
    field_ids = []
    for _ in range(root.num_children):
        element = next(elements)
        if by_field_id and element.field_id is None:
            raise NotImplementedError(
                f"Iceberg data file without parquet field IDs: {path}"
            )
        names.append(element.name)
        field_ids.append(element.field_id)
        remaining = element.num_children
        while remaining:
            remaining += next(elements).num_children - 1
    return _Footer(
        names=tuple(names),
        keys=tuple(field_ids) if by_field_id else tuple(names),
        num_rows=metadata.num_rows,
    )


def _read_group(
    paths: Sequence[str],
    present: Sequence[Any],
    footer: _Footer,
    num_rows: int,
    wanted: Mapping[Any, str],
    schema: Schema,
    lake_options: LakeScanOptions,
    *,
    by_field_id: bool,
    stream: Stream,
) -> DataFrame:
    """Read one run of same-shaped files and map it onto the table schema."""
    read: dict[Any, plc.Column] = {}
    if present:
        builder = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(list(paths))
        )
        if by_field_id:
            builder = builder.column_field_ids(list(present))
        options = builder.build()
        if not by_field_id:
            options.set_column_names(list(present))
        table = plc.io.parquet.read_parquet(options, stream=stream)
        read = {
            footer.key_of(name): column
            for name, column in zip(
                table.column_names(include_children=False),
                table.tbl.columns(),
                strict=True,
            )
        }

    columns = []
    for key, name in wanted.items():
        dtype = schema[name]
        if key in read:
            columns.append(
                Column(read[key], name=name, dtype=dtype).astype(dtype, stream)
            )
        elif lake_options.missing_columns_policy == "raise":
            raise RuntimeError(f"column {name!r} is missing from a data file")
        else:
            default = lake_options.initial_defaults.get(key) if by_field_id else None
            if default is None:
                obj = plc.Column.from_scalar(
                    plc.Scalar.from_py(None, dtype.plc_type, stream=stream),
                    num_rows,
                    stream=stream,
                )
                columns.append(Column(obj, name=name, dtype=dtype))
            else:
                (obj,) = plc.filling.repeat(
                    plc.Table([plc.Column.from_arrow(default, stream=stream)]),
                    num_rows,
                    stream=stream,
                ).columns()
                columns.append(
                    Column(obj, name=name, dtype=dtype).astype(dtype, stream)
                )
    if not columns:
        return DataFrame([], num_rows=num_rows, stream=stream)
    return DataFrame(columns, stream=stream)
