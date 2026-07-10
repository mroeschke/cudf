# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D101
"""DSL nodes for list operations."""

from __future__ import annotations

import functools
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]


class ListFunction(Expr):
    class Name(IntEnum):
        """Internal representation of Polars list functions."""

        Concat = auto()
        Contains = auto()
        DropNulls = auto()

        @classmethod
        def from_polars(cls, obj: Any) -> Self:
            """Convert from Polars' ListFunction."""
            function, name = str(obj).split(".", maxsplit=1)
            if function != "ListFunction":
                raise ValueError("ListFunction required")
            return getattr(cls, name)

    _valid_ops: ClassVar[set[Name]] = {Name.Concat, Name.Contains, Name.DropNulls}
    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: DataType,
        name: ListFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.children = children
        self.is_pointwise = True
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"List function {self.name!r}")

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = broadcast(
            *(child.evaluate(df, context=context) for child in self.children),
            stream=df.stream,
        )
        if self.name is ListFunction.Name.Concat:
            result = plc.lists.concatenate_rows(
                plc.Table([column.obj for column in columns]), stream=df.stream
            )
            if any(column.null_count for column in columns):
                valid = functools.reduce(
                    lambda left, right: plc.binaryop.binary_operation(
                        left,
                        right,
                        plc.binaryop.BinaryOperator.LOGICAL_AND,
                        plc.DataType(plc.TypeId.BOOL8),
                        stream=df.stream,
                    ),
                    (
                        plc.unary.is_valid(column.obj, stream=df.stream)
                        for column in columns
                    ),
                )
                result = result.with_mask(
                    *plc.transform.bools_to_mask(valid, stream=df.stream)
                )
            return Column(result, dtype=self.dtype)
        if self.name is ListFunction.Name.DropNulls:
            (list_column,) = columns
            child = list_column.obj.child(1)
            mask = plc.Column(
                plc.DataType(plc.TypeId.LIST),
                list_column.obj.size(),
                None,
                list_column.obj.null_mask(),
                list_column.obj.null_count(),
                list_column.obj.offset(),
                [
                    list_column.obj.child(0),
                    plc.unary.is_valid(child, stream=df.stream),
                ],
            )
            return Column(
                plc.lists.apply_boolean_mask(list_column.obj, mask, stream=df.stream),
                dtype=self.dtype,
            )
        list_column, item = columns
        contains = plc.lists.contains(list_column.obj, item.obj, stream=df.stream)
        (nulls_equal,) = self.options
        if nulls_equal and item.null_count:
            contains = plc.copying.copy_if_else(
                plc.lists.contains_nulls(list_column.obj, stream=df.stream),
                contains,
                plc.unary.is_null(item.obj, stream=df.stream),
                stream=df.stream,
            )
        return Column(contains, dtype=self.dtype)
