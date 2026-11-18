# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl
from polars.exceptions import OutOfBoundsError

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame

__all__ = ["ListFunction"]

_SIZE_TYPE_MAX = 2**31 - 1
_SIZE_TYPE = plc.DataType(plc.TypeId.INT32)
_WIDE_INDEX_TYPE_IDS = frozenset(
    {plc.TypeId.INT64, plc.TypeId.UINT32, plc.TypeId.UINT64}
)


def _with_list_child(lists: plc.Column, child: plc.Column) -> plc.Column:
    """Replace the child of ``lists`` and drop its null mask."""
    return plc.Column(
        lists.type(),
        lists.size(),
        None,
        None,
        0,
        lists.offset(),
        [lists.list_view().offsets(), child],
    )


def _gather_map(indices: Column, stream: Stream) -> plc.Column:
    """
    Convert the elements of a list of integer indices to ``size_type``.

    Indices of wider types outside ``[-(2**31 - 1), 2**31 - 1]`` are
    clamped to that range. They stay out of bounds, because a list has at most
    ``2**31 - 1`` elements.
    """
    child = indices.obj.list_view().child()
    plc_type = child.type()
    if plc_type.id() in _WIDE_INDEX_TYPE_IDS:
        lo = 0 if plc.traits.is_unsigned(plc_type) else -_SIZE_TYPE_MAX
        child = plc.replace.clamp(
            child,
            plc.Scalar.from_py(lo, plc_type, stream=stream),
            plc.Scalar.from_py(_SIZE_TYPE_MAX, plc_type, stream=stream),
            stream=stream,
        )
    inner = cast("pl.List", indices.dtype.polars_type).inner
    return _with_list_child(
        indices.obj,
        Column(child, dtype=DataType(inner)).astype(DataType(pl.Int32()), stream).obj,
    )


def _check_gather_in_bounds(
    lists: plc.Column, indices: plc.Column, gather_map: plc.Column, stream: Stream
) -> None:
    """Raise if a non-null index is out of bounds for its non-null list."""
    lengths = plc.lists.count_elements(lists, stream=stream)
    if POLARS_VERSION_LT_21:  # pragma: no cover; CI tests the newest polars
        # Polars 2.0 checks the bounds of an empty list as if it had one element.
        lengths = plc.replace.clamp(
            lengths,
            plc.Scalar.from_py(1, _SIZE_TYPE, stream=stream),
            plc.Scalar.from_py(_SIZE_TYPE_MAX, _SIZE_TYPE, stream=stream),
            stream=stream,
        )
    if indices.null_count() > 0:
        lengths = lengths.with_mask(
            *plc.null_mask.bitmask_and([lengths, indices], stream=stream)
        )
    (element_lengths,) = plc.filling.repeat(
        plc.Table([lengths]),
        plc.lists.count_elements(gather_map, stream=stream),
        stream=stream,
    ).columns()
    Op = plc.expressions.Operation
    Ref = plc.expressions.ColumnReference
    ast_op = plc.expressions.ASTOperator
    zero = plc.expressions.Literal(plc.Scalar.from_py(0, _SIZE_TYPE, stream=stream))
    in_bounds = Op(
        ast_op.LOGICAL_AND,
        Op(ast_op.LESS, Ref(0), Ref(1)),
        Op(ast_op.GREATER_EQUAL, Ref(0), Op(ast_op.SUB, zero, Ref(1))),
    )
    mask = plc.transform.compute_column(
        plc.Table([gather_map.list_view().get_sliced_child(stream), element_lengths]),
        in_bounds,
        stream=stream,
    )
    all_in_bounds = plc.reduce.reduce(
        mask, plc.aggregation.all(), plc.DataType(plc.TypeId.BOOL8), stream=stream
    ).to_py(stream=stream)
    if all_in_bounds is False:
        raise OutOfBoundsError("gather indices are out of bounds")


def _gather(
    column: Column,
    indices: Column,
    dtype: DataType,
    *,
    null_on_oob: bool,
    stream: Stream,
) -> Column:
    """Gather the elements at ``indices`` of every list in ``column``."""
    column, indices = broadcast(column, indices, stream=stream)
    gather_map = _gather_map(indices, stream)
    if not null_on_oob and column.size > 0:
        _check_gather_in_bounds(column.obj, indices.obj, gather_map, stream)
    child = gather_map.list_view().child()
    if child.null_count() > 0:
        gather_map = _with_list_child(
            gather_map,
            plc.replace.replace_nulls(
                child,
                plc.Scalar.from_py(_SIZE_TYPE_MAX, _SIZE_TYPE, stream=stream),
                stream=stream,
            ),
        )
    result = plc.lists.segmented_gather(
        column.obj,
        gather_map,
        plc.copying.OutOfBoundsPolicy.NULLIFY,
        stream=stream,
    )
    if indices.obj.null_count() > 0:
        result = result.with_mask(
            *plc.null_mask.bitmask_and([result, indices.obj], stream=stream)
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
            Name.Gather,
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

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Gather:
            (null_on_oob,) = self.options
            column, indices = (
                child.evaluate(df, context=context) for child in self.children
            )
            return _gather(
                column,
                indices,
                self.dtype,
                null_on_oob=null_on_oob,
                stream=df.stream,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
