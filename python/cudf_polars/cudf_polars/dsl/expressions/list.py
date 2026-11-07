# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc
from pylibcudf.traits import is_floating_point, is_integral_not_bool

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils.dtypes import is_order_preserving_cast, make_all_null_column

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]


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


def _is_null_literal(expr: Expr) -> bool:
    """Whether ``expr`` is a null literal that is not a list."""
    return (
        isinstance(expr, Literal)
        and expr.value is None
        and expr.dtype.id() != plc.TypeId.LIST
    )


def _is_exact_cast(source: DataType, target: DataType) -> bool:
    """Whether every value of ``source`` is unchanged by a cast to ``target``."""
    if source == target:
        return True
    from_, to = source.plc_type, target.plc_type
    return (
        (is_integral_not_bool(from_) and is_integral_not_bool(to))
        or (is_floating_point(from_) and is_floating_point(to))
    ) and is_order_preserving_cast(from_, to)


def _as_list(column: Column, dtype: DataType, stream: Stream) -> plc.Column:
    """Return ``column`` if it has ``dtype``, else ``column`` as one-element lists."""
    if column.dtype == dtype:
        return column.obj
    (inner,) = dtype.children
    child = column.astype(inner, stream).obj
    size_type = plc.DataType(plc.TypeId.INT32)
    offsets = plc.filling.sequence(
        child.size() + 1,
        plc.Scalar.from_py(0, size_type, stream=stream),
        plc.Scalar.from_py(1, size_type, stream=stream),
        stream=stream,
    )
    return plc.Column(dtype.plc_type, child.size(), None, None, 0, 0, [offsets, child])


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
            Name.Concat,
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
        if self.name is ListFunction.Name.Concat:
            (inner,) = self.dtype.children
            if any(
                child.dtype != self.dtype
                if child.dtype.id() == plc.TypeId.LIST
                else not (_is_null_literal(child) or _is_exact_cast(child.dtype, inner))
                for child in self.children
            ):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for inputs of different dtypes"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Concat:
            (inner,) = self.dtype.children
            columns = broadcast(
                *(
                    Column(make_all_null_column(inner, 1, df.stream), dtype=inner)
                    if _is_null_literal(child)
                    else child.evaluate(df, context=context)
                    for child in self.children
                ),
                stream=df.stream,
            )
            return Column(
                _concat(
                    [_as_list(column, self.dtype, df.stream) for column in columns],
                    df.stream,
                ),
                dtype=self.dtype,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
