# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.utils.dtypes import make_all_null_column

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]


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
            Name.Contains,
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
        if self.name is ListFunction.Name.Contains:
            column, item = self.children
            if item.dtype != column.dtype.children[0]:
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for a "
                    f"{item.dtype.polars_type} item in {column.dtype.polars_type}"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Contains:
            (nulls_equal,) = self.options
            column, item = self.children
            needle = (
                item
                if isinstance(item, Literal)
                else item.evaluate(df, context=context)
            )
            return Column(
                _contains(
                    column.evaluate(df, context=context),
                    needle,
                    self.dtype,
                    nulls_equal=nulls_equal,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover


def _contains(
    column: Column,
    item: Literal | Column,
    dtype: DataType,
    *,
    nulls_equal: bool,
    stream: Stream,
) -> plc.Column:
    """
    Whether each list of ``column`` contains ``item``, as a ``dtype`` column.

    ``column`` and an ``item`` column must be valid on ``stream``.
    """
    if isinstance(item, Literal):
        if item.value is not None:
            return plc.lists.contains(
                column.obj,
                plc.Scalar.from_py(item.value, item.dtype.plc_type, stream=stream),
                stream=stream,
            )
        if nulls_equal:
            return plc.lists.contains_nulls(column.obj, stream=stream)
        return make_all_null_column(dtype, column.size, stream)
    column, item = broadcast(column, item, stream=stream)
    result = plc.lists.contains(column.obj, item.obj, stream=stream)
    if nulls_equal and item.null_count > 0:
        return plc.replace.replace_nulls(
            result, plc.lists.contains_nulls(column.obj, stream=stream), stream=stream
        )
    return result
