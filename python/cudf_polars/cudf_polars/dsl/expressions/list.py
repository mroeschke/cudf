# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

import pylibcudf as plc
from pylibcudf.traits import is_chrono, is_numeric_not_bool

from cudf_polars.containers import Column
from cudf_polars.containers.column import with_mask_from
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]


class NullBehavior(Enum):
    """Null behavior of a polars ``ListFunction.Diff``."""

    IGNORE = "ignore"
    DROP = "drop"


def _bool_list_like(col: plc.Column, bool_child: plc.Column) -> plc.Column:
    """
    Build a list-of-bool column with the offsets of ``col`` and no null mask.

    ``bool_child`` has one value per element of the full child of ``col``.
    """
    return plc.Column(
        col.type(),
        col.size(),
        None,
        None,
        0,
        col.offset(),
        [col.list_view().offsets(), bool_child],
    )


def _diff_operand(child: plc.Column, dtype: plc.DataType, stream: Stream) -> plc.Column:
    """
    Cast the list elements ``child`` to the type Polars subtracts them in.

    Like Polars, a Date that overflows as microseconds and a UInt64 that
    overflows as Int64 become null.

    ``child`` must be valid on ``stream``; synchronizes ``stream`` for Date
    and UInt64 elements.
    """
    typ = child.type()
    if typ.id() == plc.TypeId.TIMESTAMP_DAYS:
        cast = plc.unary.cast(
            child, plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS), stream=stream
        )
        in_range = plc.binaryop.binary_operation(
            plc.unary.cast(cast, typ, stream=stream),
            child,
            plc.binaryop.BinaryOperator.EQUAL,
            plc.DataType(plc.TypeId.BOOL8),
            stream=stream,
        )
    elif typ.id() == plc.TypeId.UINT64:
        cast = plc.unary.cast(child, dtype, stream=stream)
        in_range = plc.binaryop.binary_operation(
            child,
            plc.Scalar.from_py(2**63 - 1, typ, stream=stream),
            plc.binaryop.BinaryOperator.LESS_EQUAL,
            plc.DataType(plc.TypeId.BOOL8),
            stream=stream,
        )
    elif is_numeric_not_bool(typ) and typ != dtype:
        return plc.unary.cast(child, dtype, stream=stream)
    else:
        return child
    return plc.copying.copy_if_else(
        cast,
        plc.Scalar.from_py(None, cast.type(), stream=stream),
        in_range,
        stream=stream,
    )


def _shift(
    column: Column, child: plc.Column, n: int, stream: Stream
) -> tuple[plc.Column, plc.Column, plc.Column | None]:
    """
    Shift the elements ``child`` of every list of ``column`` by ``n`` positions.

    ``child`` has one value per element of the sliced child of ``column``.
    Return the offsets of the lists starting at 0, ``child`` shifted across
    list boundaries, and a mask that is false where the shifted value comes
    from another list (``None`` if ``n`` is 0).

    ``column`` and ``child`` must be valid on ``stream``; synchronizes
    ``stream``.
    """
    n = max(-child.size(), min(n, child.size()))
    sizes = plc.lists.count_elements(column.obj.with_mask(None, 0), stream=stream)
    if n >= 0:
        starts = plc.Column.from_scalar(
            plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
            column.size,
            stream=stream,
        )
        steps = None
    else:
        starts = plc.binaryop.binary_operation(
            sizes,
            plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream),
            plc.binaryop.BinaryOperator.SUB,
            plc.types.SIZE_TYPE,
            stream=stream,
        )
        steps = plc.Column.from_scalar(
            plc.Scalar.from_py(-1, plc.types.SIZE_TYPE, stream=stream),
            column.size,
            stream=stream,
        )
    positions = plc.lists.sequences(starts, sizes, steps, stream=stream).list_view()
    shifted = plc.copying.shift(
        child, n, plc.Scalar.from_py(None, child.type(), stream=stream), stream=stream
    )
    if n == 0:
        return positions.offsets(), shifted, None
    keep = plc.binaryop.binary_operation(
        positions.child(),
        plc.Scalar.from_py(abs(n), plc.types.SIZE_TYPE, stream=stream),
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
        plc.DataType(plc.TypeId.BOOL8),
        stream=stream,
    )
    return positions.offsets(), shifted, keep


def _diff(
    column: Column,
    n: int,
    null_behavior: NullBehavior,
    dtype: plc.DataType,
    stream: Stream,
) -> plc.Column:
    """
    Subtract from every list element the element ``n`` positions before it.

    ``column`` must be valid on ``stream``; synchronizes ``stream``.
    """
    child = _diff_operand(
        column.obj.list_view().get_sliced_child(stream=stream), dtype, stream
    )
    offsets, shifted, keep = _shift(column, child, n, stream)
    diff = plc.binaryop.binary_operation(
        child, shifted, plc.binaryop.BinaryOperator.SUB, dtype, stream=stream
    )
    if keep is not None and null_behavior is NullBehavior.IGNORE:
        diff = plc.copying.copy_if_else(
            diff, plc.Scalar.from_py(None, dtype, stream=stream), keep, stream=stream
        )
    result = with_mask_from(
        plc.Column(
            plc.DataType(plc.TypeId.LIST),
            column.size,
            None,
            None,
            0,
            0,
            [offsets, diff],
        ),
        column,
        stream,
    )
    if keep is not None and null_behavior is NullBehavior.DROP:
        return plc.lists.apply_retention_mask(
            result, _bool_list_like(result, keep), stream=stream
        )
    return result


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
            Name.Diff,
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
        if self.name is ListFunction.Name.Diff:
            n, null_behavior = self.options
            null_behavior = NullBehavior(null_behavior)
            self.options = (n, null_behavior)
            (inner,) = self.children[0].dtype.children
            if isinstance(inner.polars_type, pl.Null) or not (
                is_numeric_not_bool(inner.plc_type) or is_chrono(inner.plc_type)
            ):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for {inner.polars_type} elements"
                )
            if null_behavior is NullBehavior.DROP and abs(n) > 2:
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for null_behavior='drop' with abs(n) > 2"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Diff:
            (column,) = (child.evaluate(df, context=context) for child in self.children)
            n, null_behavior = self.options
            (inner,) = self.dtype.children
            return Column(
                _diff(column, n, null_behavior, inner.plc_type, df.stream),
                dtype=self.dtype,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
