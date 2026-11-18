# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

import math
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame

__all__ = ["ListFunction"]


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


def _count_matches_item(item: Literal, inner: DataType) -> Literal | None:
    """
    ``item`` as a literal of ``inner``, or None if the GPU can't match Polars.

    Polars compares the elements and ``item`` in their supertype. That is the same
    as comparing in ``inner`` only if ``item`` converts to ``inner`` exactly, and
    the supertype doesn't lose precision of the elements.
    """
    if item.dtype == inner:
        return item
    source, target = item.dtype.polars_type, inner.polars_type
    if not (
        isinstance(source, pl.Null)
        or (source.is_integer() and target.is_numeric())
        or (source.is_float() and target.is_float())
        or (source.is_decimal() and target.is_decimal())
    ):
        return None
    value = pl.Series([item.value], dtype=source).cast(target, strict=False).item()
    if value == item.value or (
        source.is_float() and math.isnan(item.value) and math.isnan(value)
    ):
        return Literal(inner, value)
    return None


def _count_matches(column: plc.Column, item: Literal, stream: Stream) -> plc.Column:
    """Count the elements of every list of ``column`` that are equal to ``item``."""
    child = column.list_view().child()
    if item.value is None:
        matches = plc.unary.is_null(child, stream=stream)
    elif isinstance(item.value, float) and math.isnan(item.value):
        matches = plc.unary.is_nan(child, stream=stream)
    else:
        matches = plc.binaryop.binary_operation(
            child,
            plc.Scalar.from_py(item.value, item.dtype.plc_type, stream=stream),
            plc.binaryop.BinaryOperator.NULL_EQUALS,
            plc.DataType(plc.TypeId.BOOL8),
            stream=stream,
        )
    return plc.lists.count_elements(
        plc.lists.apply_retention_mask(
            column, _bool_list_like(column, matches), stream=stream
        ),
        stream=stream,
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
            Name.CountMatches,
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
        if self.name is ListFunction.Name.CountMatches:
            column, item = self.children
            (inner,) = column.dtype.children
            if not isinstance(item, Literal):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for a "
                    "non-literal item"
                )
            if inner.id() in {plc.TypeId.LIST, plc.TypeId.STRUCT}:
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for "
                    f"{column.dtype.polars_type}"
                )
            match = _count_matches_item(item, inner)
            if match is None:
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for a "
                    f"{item.dtype.polars_type} item in {column.dtype.polars_type}"
                )
            self.children = (column, match)

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.CountMatches:
            column, item = self.children
            assert isinstance(item, Literal)
            return Column(
                _count_matches(
                    column.evaluate(df, context=context).obj, item, df.stream
                ),
                dtype=DataType(pl.Int32()),
            ).astype(self.dtype, stream=df.stream)
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
