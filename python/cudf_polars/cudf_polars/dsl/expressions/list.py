# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import Enum, IntEnum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc
from pylibcudf.traits import is_chrono, is_fixed_point, is_numeric_not_bool

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from polars import _plr as polars

    from pylibcudf.span import Span
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]


class SetOperation(Enum):
    """Set operation of a polars ``ListFunction.SetOperation``."""

    UNION = "union"
    INTERSECTION = "intersection"
    DIFFERENCE = "difference"
    SYMMETRIC_DIFFERENCE = "symmetric_difference"


def _sequence(size: int, stream: Stream) -> plc.Column:
    """
    Return the ``SIZE_TYPE`` column ``0, 1, ..., size - 1``.

    Runs on ``stream`` and does not synchronize it.
    """
    return plc.filling.sequence(
        size,
        plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
        plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream),
        stream=stream,
    )


def _row_ids(col: plc.Column, stream: Stream) -> plc.Column:
    """
    Return the row index of every element of the sliced child of list ``col``.

    ``col`` must be valid on ``stream``; synchronizes ``stream``.
    """
    sizes = plc.lists.count_elements(col.with_mask(None, 0), stream=stream)
    (row_ids,) = plc.filling.repeat(
        plc.Table([_sequence(col.size(), stream)]), sizes, stream=stream
    ).columns()
    return row_ids


def _exploded(col: plc.Column, stream: Stream) -> plc.Table:
    """
    Return the ``(row_id, value)`` table of the elements of list ``col``.

    ``col`` must be valid on ``stream``; synchronizes ``stream``.
    """
    return plc.Table(
        [_row_ids(col, stream), col.list_view().get_sliced_child(stream=stream)]
    )


def _distinct_elements(exploded: plc.Table, stream: Stream) -> plc.Table:
    """
    Return the first appearance of every distinct row of ``exploded``, in order.

    ``exploded`` must be valid on ``stream``; synchronizes ``stream``.
    """
    return plc.stream_compaction.stable_distinct(
        exploded,
        [0, 1],
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.types.NullEquality.EQUAL,
        plc.types.NanEquality.ALL_EQUAL,
        stream=stream,
    )


def _offsets_from_sorted_row_ids(
    row_ids: plc.Column, n: int, stream: Stream
) -> plc.Column:
    """
    Return the offsets of ``n`` lists whose elements have the sorted ``row_ids``.

    ``row_ids`` must be valid on ``stream``; does not synchronize ``stream``.
    """
    return plc.search.lower_bound(
        plc.Table([row_ids]),
        plc.Table([_sequence(n + 1, stream)]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.AFTER],
        stream=stream,
    )


def _rebuild_list(
    offsets: plc.Column,
    child: plc.Column,
    parent_mask: Span | None,
    null_count: int,
) -> plc.Column:
    """
    Build a list column from ``offsets``, ``child`` and an unsliced null mask.

    Launches no device work, so it takes no stream.
    """
    return plc.Column(
        plc.DataType(plc.TypeId.LIST),
        offsets.size() - 1,
        None,
        parent_mask,
        null_count,
        0,
        [offsets, child],
    )


def _concat(columns: Sequence[plc.Column], stream: Stream) -> plc.Column:
    """Concatenate the rows of list ``columns``, with a null row if any input row is null."""
    mask, null_count = plc.null_mask.bitmask_and(columns, stream=stream)
    if null_count > 0:
        # concatenate_rows only nullifies a row if every input row is null: https://github.com/rapidsai/cudf/blob/d71b3b743ca6fc9d8d4b1718d845eec4bc63dee6/python/pylibcudf/pylibcudf/lists.pyx#L163-L165
        columns = [
            (column.copy(stream=stream) if column.offset() != 0 else column).with_mask(
                mask, null_count
            )
            for column in columns
        ]
    return plc.lists.concatenate_rows(plc.Table(list(columns)), stream=stream)


def _set_union(a: plc.Column, b: plc.Column, stream: Stream) -> plc.Column:
    """
    Return the distinct elements of each row of ``a`` followed by ``b``, in order.

    ``a`` and ``b`` must be valid on ``stream``; synchronizes ``stream``.
    """
    concatenated = _concat([a, b], stream)
    row_ids, values = _distinct_elements(
        _exploded(concatenated, stream), stream
    ).columns()
    return _rebuild_list(
        _offsets_from_sorted_row_ids(row_ids, concatenated.size(), stream),
        values,
        concatenated.null_mask(),
        concatenated.null_count(),
    )


_SET_OPERATIONS = MappingProxyType(
    {
        SetOperation.UNION: _set_union,
    }
)


def _is_set_operation_type(dtype: DataType) -> bool:
    """Whether the set operations support list elements of ``dtype``."""
    typ = dtype.plc_type
    return (
        typ.id() == plc.TypeId.STRING
        or is_numeric_not_bool(typ)
        or is_chrono(typ)
        or is_fixed_point(typ)
    )


class ListFunction(Expr):
    """Expression node for functions in the polars ``list`` namespace."""

    class Name(IntEnum):
        """Internal and picklable representation of polars' `ListFunction`."""

        ArgMax = auto()
        ArgMin = auto()
        Concat = auto()
        Contains = auto()
        CountMatches = auto()
        Diff = auto()
        DropNulls = auto()
        Gather = auto()
        GatherEvery = auto()
        Get = auto()
        Join = auto()
        Length = auto()
        Max = auto()
        Mean = auto()
        Median = auto()
        Min = auto()
        Sample = auto()
        SetOperation = auto()
        Shift = auto()
        Slice = auto()
        Sort = auto()
        Std = auto()
        Sum = auto()
        ToArray = auto()
        ToStruct = auto()
        Var = auto()

        @classmethod
        def from_polars(cls, obj: polars._expr_nodes.ListFunction) -> Self:
            """Convert from polars' `ListFunction`."""
            return getattr(cls, str(obj).removeprefix("ListFunction."))

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    _valid_ops: ClassVar[frozenset[Name]] = frozenset(
        {
            Name.SetOperation,
        }
    )

    def __init__(
        self,
        dtype: DataType,
        name: ListFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"ListFunction.{self.name.name} is not supported")
        if self.name is ListFunction.Name.SetOperation:
            (operation,) = self.options
            operation = SetOperation(operation)
            self.options = (operation,)
            if operation not in _SET_OPERATIONS:
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for {operation.value}"
                )
            (inner,) = self.dtype.children
            if not _is_set_operation_type(inner):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for {inner.polars_type} elements"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.SetOperation:
            (operation,) = self.options
            a, b = broadcast(
                *(child.evaluate(df, context=context) for child in self.children),
                stream=df.stream,
            )
            return Column(
                _SET_OPERATIONS[operation](a.obj, b.obj, df.stream), dtype=self.dtype
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
