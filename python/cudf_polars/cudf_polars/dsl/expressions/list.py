# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
from polars.exceptions import ComputeError

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils.dtypes import make_all_null_column
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame

__all__ = ["ListFunction"]

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_OFFSET_MAX = 2**31


def _slice_arg_to_int64(arg: Column, stream: Stream) -> plc.Column:
    """
    Cast an integer ``list.slice`` argument to Int64.

    Polars casts the arguments to Int64 non-strictly, so UInt64 values
    above the Int64 range are null.
    """
    obj = arg.obj
    if obj.type().id() == plc.TypeId.UINT64:
        obj = plc.copying.copy_if_else(
            plc.Scalar.from_py(None, obj.type(), stream=stream),
            obj,
            plc.binaryop.binary_operation(
                obj,
                plc.Scalar.from_py(_INT64_MAX, obj.type(), stream=stream),
                plc.binaryop.BinaryOperator.GREATER,
                plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            ),
            stream=stream,
        )
    return Column(obj, dtype=arg.dtype).astype(DataType(pl.Int64()), stream).obj


def _slice_bounds(
    counts: plc.Column,
    offset: plc.Column | int,
    length: plc.Column | int,
    stream: Stream,
) -> tuple[plc.Column, plc.Column]:
    """
    Compute the start and the size of every sliced list.

    ``counts`` holds the Int64 list lengths. ``offset`` and ``length``
    are Int64 without nulls, ``offset`` is at most ``_OFFSET_MAX`` and
    ``length`` is non-negative, so no step overflows. Like Polars, a
    negative offset counts from the end of the list, and the part of the
    slice before the start of the list is cut from the length.
    """
    int64 = plc.DataType(plc.TypeId.INT64)
    BinOp = plc.binaryop.BinaryOperator

    def scalar(value: int) -> plc.Scalar:
        return plc.Scalar.from_py(value, int64, stream=stream)

    def binop(
        lhs: plc.Column, rhs: plc.Column | plc.Scalar, op: plc.binaryop.BinaryOperator
    ) -> plc.Column:
        return plc.binaryop.binary_operation(lhs, rhs, op, int64, stream=stream)

    def clamp(column: plc.Column, lo: int, hi: int) -> plc.Column:
        return plc.replace.clamp(column, scalar(lo), scalar(hi), stream=stream)

    if isinstance(offset, int):
        resolved = (
            binop(counts, scalar(offset), BinOp.ADD)
            if offset < 0
            else plc.Column.from_scalar(scalar(offset), counts.size(), stream=stream)
        )
    else:
        resolved = plc.copying.copy_if_else(
            binop(offset, counts, BinOp.ADD),
            offset,
            plc.binaryop.binary_operation(
                offset,
                scalar(0),
                BinOp.LESS,
                plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            ),
            stream=stream,
        )
    start = binop(clamp(resolved, 0, _INT64_MAX), counts, BinOp.NULL_MIN)
    wanted = binop(
        clamp(resolved, _INT64_MIN, 0),
        scalar(length) if isinstance(length, int) else length,
        BinOp.ADD,
    )
    size = binop(
        clamp(wanted, 0, _INT64_MAX),
        binop(counts, start, BinOp.SUB),
        BinOp.NULL_MIN,
    )
    return start, size


def _slice(
    column: Column,
    offset: Column | int | None,
    length: Column | int | None,
    dtype: DataType,
    *,
    stream: Stream,
) -> Column:
    """Slice every list in ``column``."""
    column, *args = broadcast(
        column,
        *(arg for arg in (offset, length) if isinstance(arg, Column)),
        stream=stream,
    )
    if isinstance(length, Column):
        length = args.pop()
    if isinstance(offset, Column):
        offset = args.pop()
    if offset is None or (isinstance(offset, int) and offset > _INT64_MAX):
        return Column(make_all_null_column(dtype, column.size, stream), dtype=dtype)
    if (
        POLARS_VERSION_LT_21 and isinstance(offset, Column) and offset.null_count > 0
    ):  # pragma: no cover; CI tests the newest polars
        raise ComputeError("could not extract number from any-value of dtype: 'Null'")
    int64 = DataType(pl.Int64())
    int32 = DataType(pl.Int32())
    offset_mask = None
    offset_bound: plc.Column | int
    if isinstance(offset, Column):
        offset_obj = _slice_arg_to_int64(offset, stream)
        if offset_obj.null_count() > 0:
            offset_mask = offset_obj
        offset_bound = plc.replace.replace_nulls(
            plc.replace.clamp(
                offset_obj,
                plc.Scalar.from_py(_INT64_MIN, int64.plc_type, stream=stream),
                plc.Scalar.from_py(_OFFSET_MAX, int64.plc_type, stream=stream),
                stream=stream,
            ),
            plc.Scalar.from_py(_OFFSET_MAX, int64.plc_type, stream=stream),
            stream=stream,
        )
    else:
        offset_bound = offset
    length_bound: plc.Column | int
    if isinstance(length, Column):
        length_obj = _slice_arg_to_int64(length, stream)
        length_bound = plc.copying.copy_if_else(
            length_obj,
            plc.Scalar.from_py(_INT64_MAX, int64.plc_type, stream=stream),
            plc.binaryop.binary_operation(
                length_obj,
                plc.Scalar.from_py(0, int64.plc_type, stream=stream),
                plc.binaryop.BinaryOperator.GREATER_EQUAL,
                plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            ),
            stream=stream,
        )
    elif length is None or not 0 <= length <= _INT64_MAX:
        length_bound = _INT64_MAX
    else:
        length_bound = length
    counts = Column(
        plc.replace.replace_nulls(
            plc.lists.count_elements(column.obj, stream=stream),
            plc.Scalar.from_py(0, int32.plc_type, stream=stream),
            stream=stream,
        ),
        dtype=int32,
    ).astype(int64, stream)
    starts, sizes = (
        Column(bound, dtype=int64).astype(int32, stream).obj
        for bound in _slice_bounds(counts.obj, offset_bound, length_bound, stream)
    )
    result = plc.lists.segmented_gather(
        column.obj,
        plc.lists.sequences(starts, sizes, stream=stream),
        stream=stream,
    )
    if offset_mask is not None:
        result = result.with_mask(
            *plc.null_mask.bitmask_and([result, offset_mask], stream=stream)
        )
    return Column(result, dtype=dtype)


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
            Name.Slice,
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
        if self.name is ListFunction.Name.Slice:
            _, offset, length = children
            for arg, role in ((offset, "offset"), (length, "length")):
                if not (
                    plc.traits.is_integral_not_bool(arg.dtype.plc_type)
                    or (isinstance(arg, Literal) and arg.value is None)
                ):
                    raise NotImplementedError(
                        f"ListFunction.{self.name.name} is not supported for "
                        f"{arg.dtype.polars_type} {role}"
                    )
            if POLARS_VERSION_LT_21 and (
                offset.dtype.id() == plc.TypeId.UINT64
                or (isinstance(offset, Literal) and offset.value is None)
            ):  # pragma: no cover; CI tests the newest polars
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for "
                    "null or UInt64 offset on Polars < 2.1"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Slice:
            list_expr, offset_expr, length_expr = self.children
            offset, length = (
                child.value
                if isinstance(child, Literal)
                else child.evaluate(df, context=context)
                for child in (offset_expr, length_expr)
            )
            return _slice(
                list_expr.evaluate(df, context=context),
                offset,
                length,
                self.dtype,
                stream=df.stream,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
