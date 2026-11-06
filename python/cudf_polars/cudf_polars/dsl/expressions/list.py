# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl
from polars.exceptions import ComputeError

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils.dtypes import make_all_null_column

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame

__all__ = ["ListFunction"]

_SIZE_TYPE_MIN = -(2**31)
_SIZE_TYPE_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1
_SIZE_TYPE = plc.DataType(plc.TypeId.INT32)
_WIDE_INDEX_TYPE_IDS = frozenset(
    {plc.TypeId.INT64, plc.TypeId.UINT32, plc.TypeId.UINT64}
)


def _compact(column: plc.Column, stream: Stream) -> plc.Column:
    """Copy ``column`` to offset 0 if it is a sliced column with nulls."""
    if column.offset() != 0 and column.null_count() > 0:
        return column.copy(stream=stream)
    return column


def _to_size_type(indices: Column, stream: Stream) -> plc.Column:
    """
    Cast integer indices to ``size_type``.

    Polars casts indices to Int64 non-strictly, so UInt64 indices above
    the Int64 range are null. Other indices outside the ``size_type``
    range are clamped to it. They stay out of bounds, because a list has
    at most ``2**31 - 1`` elements.
    """
    obj = indices.obj
    plc_type = obj.type()
    if plc_type.id() == plc.TypeId.UINT64:
        obj = plc.copying.copy_if_else(
            plc.Scalar.from_py(None, plc_type, stream=stream),
            obj,
            plc.binaryop.binary_operation(
                obj,
                plc.Scalar.from_py(_INT64_MAX, plc_type, stream=stream),
                plc.binaryop.BinaryOperator.GREATER,
                plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            ),
            stream=stream,
        )
    if plc_type.id() in _WIDE_INDEX_TYPE_IDS:
        lo = 0 if plc.traits.is_unsigned(plc_type) else _SIZE_TYPE_MIN
        indices = Column(
            plc.replace.clamp(
                obj,
                plc.Scalar.from_py(lo, plc_type, stream=stream),
                plc.Scalar.from_py(_SIZE_TYPE_MAX, plc_type, stream=stream),
                stream=stream,
            ),
            dtype=indices.dtype,
        )
    return indices.astype(DataType(pl.Int32()), stream).obj


def _check_index_in_bounds(lists: plc.Column, index: int, stream: Stream) -> None:
    """Raise if a scalar index is out of bounds for a non-null list."""
    lengths = plc.lists.count_elements(lists, stream=stream)
    shortest = cast(
        "int | None",
        plc.reduce.reduce(
            lengths, plc.aggregation.min(), _SIZE_TYPE, stream=stream
        ).to_py(stream=stream),
    )
    required = index + 1 if index >= 0 else -index
    if shortest is not None and shortest < required:
        raise ComputeError("get index is out of bounds")


def _check_indices_in_bounds(
    lists: plc.Column, indices: plc.Column, stream: Stream
) -> None:
    """Raise if a non-null index is out of bounds for its non-null list."""
    Op = plc.expressions.Operation
    Ref = plc.expressions.ColumnReference
    ast_op = plc.expressions.ASTOperator
    lengths = plc.lists.count_elements(lists, stream=stream)
    zero = plc.expressions.Literal(plc.Scalar.from_py(0, _SIZE_TYPE, stream=stream))
    in_bounds = Op(
        ast_op.LOGICAL_AND,
        Op(ast_op.LESS, Ref(0), Ref(1)),
        Op(ast_op.GREATER_EQUAL, Ref(0), Op(ast_op.SUB, zero, Ref(1))),
    )
    mask = plc.transform.compute_column(
        plc.Table([indices, lengths]), in_bounds, stream=stream
    )
    all_in_bounds = plc.reduce.reduce(
        mask, plc.aggregation.all(), plc.DataType(plc.TypeId.BOOL8), stream=stream
    ).to_py(stream=stream)
    if all_in_bounds is False:
        raise ComputeError("get index is out of bounds")


def _get(
    column: Column,
    index: Column | int | None,
    dtype: DataType,
    *,
    null_on_oob: bool,
    stream: Stream,
) -> Column:
    """Get the element at ``index`` of every list in ``column``."""
    if index is None or (isinstance(index, int) and index > _INT64_MAX):
        return Column(make_all_null_column(dtype, column.size, stream), dtype=dtype)
    if isinstance(index, Column):
        column, index = broadcast(column, index, stream=stream)
    # extract_list_element ignores the column offset when it applies the parent null mask: https://github.com/rapidsai/cudf/blob/d71b3b743ca6fc9d8d4b1718d845eec4bc63dee6/cpp/src/lists/extract.cu#L171-L172
    lists = _compact(column.obj, stream)
    if isinstance(index, int):
        size_type_index = min(max(index, _SIZE_TYPE_MIN), _SIZE_TYPE_MAX)
        if not null_on_oob:
            _check_index_in_bounds(lists, size_type_index, stream)
        result = plc.lists.extract_list_element(lists, size_type_index, stream=stream)
    else:
        indices = _to_size_type(index, stream)
        if not null_on_oob:
            _check_indices_in_bounds(lists, indices, stream)
        result = plc.lists.extract_list_element(lists, indices, stream=stream)
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
            Name.Get,
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
        if self.name is ListFunction.Name.Get:
            _, index = children
            if not plc.traits.is_integral_not_bool(index.dtype.plc_type):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for "
                    f"{index.dtype.polars_type} index"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Get:
            (null_on_oob,) = self.options
            list_expr, index_expr = self.children
            return _get(
                list_expr.evaluate(df, context=context),
                index_expr.value
                if isinstance(index_expr, Literal)
                else index_expr.evaluate(df, context=context),
                self.dtype,
                null_on_oob=null_on_oob,
                stream=df.stream,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
