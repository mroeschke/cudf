# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D101
"""DSL nodes for list operations."""

from __future__ import annotations

import functools
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

from polars.exceptions import ComputeError

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["Explode", "ListFunction"]


class Explode(Expr):
    __slots__ = ("options",)
    _non_child = ("dtype", "options")

    def __init__(
        self, dtype: DataType, options: tuple[bool, bool], child: Expr
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.children = (child,)
        self.is_pointwise = False
        if self.options != (True, True):
            raise NotImplementedError(f"Explode options {self.options!r}")

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        result = plc.lists.explode_outer(
            plc.Table([column.obj]), 0, stream=df.stream
        ).columns()[0]
        return Column(result, dtype=self.dtype)


class ListFunction(Expr):
    class Name(IntEnum):
        """Internal representation of Polars list functions."""

        Concat = auto()
        Contains = auto()
        DropNulls = auto()
        Get = auto()
        Length = auto()
        SetOperation = auto()
        Sort = auto()

        @classmethod
        def from_polars(cls, obj: Any) -> Self:
            """Convert from Polars' ListFunction."""
            function, name = str(obj).split(".", maxsplit=1)
            if function != "ListFunction":
                raise ValueError("ListFunction required")
            return getattr(cls, name)

    _valid_ops: ClassVar[set[Name]] = {
        Name.Concat,
        Name.Contains,
        Name.DropNulls,
        Name.Get,
        Name.Length,
        Name.SetOperation,
        Name.Sort,
    }
    _valid_set_operations: ClassVar[set[str]] = {
        "difference",
        "intersection",
        "symmetric_difference",
        "union",
    }
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
        if (
            self.name is self.Name.SetOperation
            and self.options[0] not in self._valid_set_operations
        ):
            raise NotImplementedError(f"List set operation {self.options[0]!r}")

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = broadcast(
            *(child.evaluate(df, context=context) for child in self.children),
            stream=df.stream,
        )
        if self.name is ListFunction.Name.Concat:
            list_columns = []
            for column in columns:
                if column.dtype.id() == plc.TypeId.LIST:
                    list_columns.append(column.obj)
                else:
                    int32 = plc.DataType(plc.TypeId.INT32)
                    offsets = plc.filling.sequence(
                        column.size + 1,
                        plc.Scalar.from_py(0, int32, stream=df.stream),
                        plc.Scalar.from_py(1, int32, stream=df.stream),
                        stream=df.stream,
                    )
                    list_columns.append(
                        plc.Column(
                            plc.DataType(plc.TypeId.LIST),
                            column.size,
                            None,
                            None,
                            0,
                            0,
                            [offsets, column.obj],
                        )
                    )
            result = plc.lists.concatenate_rows(
                plc.Table(list_columns), stream=df.stream
            )
            if any(column.null_count() for column in list_columns):
                valid = functools.reduce(
                    lambda left, right: plc.binaryop.binary_operation(
                        left,
                        right,
                        plc.binaryop.BinaryOperator.LOGICAL_AND,
                        plc.DataType(plc.TypeId.BOOL8),
                        stream=df.stream,
                    ),
                    (
                        plc.unary.is_valid(column, stream=df.stream)
                        for column in list_columns
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
        if self.name is ListFunction.Name.Contains:
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
        if self.name is ListFunction.Name.Length:
            (list_column,) = columns
            return Column(
                plc.unary.cast(
                    plc.lists.count_elements(list_column.obj, stream=df.stream),
                    self.dtype.plc_type,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        if self.name is ListFunction.Name.Sort:
            (list_column,) = columns
            descending, nulls_last = self.options
            order, null_order = sorting.sort_order(
                [descending], nulls_last=[nulls_last], num_keys=1
            )
            return Column(
                plc.lists.sort_lists(
                    list_column.obj,
                    order[0],
                    null_order[0],
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        if self.name is ListFunction.Name.SetOperation:
            lhs, rhs = columns
            (operation,) = self.options
            if operation == "symmetric_difference":
                lhs_only = plc.lists.difference_distinct(
                    lhs.obj,
                    rhs.obj,
                    nulls_equal=plc.types.NullEquality.EQUAL,
                    nans_equal=plc.types.NanEquality.ALL_EQUAL,
                    stream=df.stream,
                )
                rhs_only = plc.lists.difference_distinct(
                    rhs.obj,
                    lhs.obj,
                    nulls_equal=plc.types.NullEquality.EQUAL,
                    nans_equal=plc.types.NanEquality.ALL_EQUAL,
                    stream=df.stream,
                )
                result = plc.lists.union_distinct(
                    lhs_only,
                    rhs_only,
                    nulls_equal=plc.types.NullEquality.EQUAL,
                    nans_equal=plc.types.NanEquality.ALL_EQUAL,
                    stream=df.stream,
                )
                return Column(result, dtype=self.dtype)
            function = {
                "difference": plc.lists.difference_distinct,
                "union": plc.lists.union_distinct,
            }.get(operation)
            if function is None:
                return Column(
                    plc.lists.intersect_distinct(
                        rhs.obj,
                        lhs.obj,
                        nulls_equal=plc.types.NullEquality.EQUAL,
                        nans_equal=plc.types.NanEquality.ALL_EQUAL,
                        stream=df.stream,
                    ),
                    dtype=self.dtype,
                )
            return Column(
                function(
                    lhs.obj,
                    rhs.obj,
                    nulls_equal=plc.types.NullEquality.EQUAL,
                    nans_equal=plc.types.NanEquality.ALL_EQUAL,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        list_column, index = columns
        (null_on_oob,) = self.options
        if not null_on_oob:
            lengths = plc.unary.cast(
                plc.lists.count_elements(list_column.obj, stream=df.stream),
                index.obj.type(),
                stream=df.stream,
            )
            upper_oob = plc.binaryop.binary_operation(
                index.obj,
                lengths,
                plc.binaryop.BinaryOperator.GREATER_EQUAL,
                plc.DataType(plc.TypeId.BOOL8),
                stream=df.stream,
            )
            lower_oob = plc.binaryop.binary_operation(
                index.obj,
                plc.unary.unary_operation(
                    lengths, plc.unary.UnaryOperator.NEGATE, stream=df.stream
                ),
                plc.binaryop.BinaryOperator.LESS,
                plc.DataType(plc.TypeId.BOOL8),
                stream=df.stream,
            )
            oob = plc.binaryop.binary_operation(
                upper_oob,
                lower_oob,
                plc.binaryop.BinaryOperator.LOGICAL_OR,
                plc.DataType(plc.TypeId.BOOL8),
                stream=df.stream,
            )
            if plc.reduce.reduce(
                oob,
                plc.aggregation.any(),
                plc.DataType(plc.TypeId.BOOL8),
                stream=df.stream,
            ).to_py(stream=df.stream):
                raise ComputeError("get index is out of bounds")
        return Column(
            plc.lists.extract_list_element(
                list_column.obj, index.obj, stream=df.stream
            ),
            dtype=self.dtype,
        )
