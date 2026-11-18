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

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]


def _join(
    lists: plc.Column,
    separator: plc.Column | str | None,
    *,
    ignore_nulls: bool,
    stream: Stream,
) -> plc.Column:
    """Join the strings of each list in ``lists`` with ``separator``."""
    string = plc.DataType(plc.TypeId.STRING)
    if separator is None:
        return plc.Column.from_scalar(
            plc.Scalar.from_py(None, string, stream=stream), lists.size(), stream=stream
        )
    narep = plc.Scalar.from_py("" if ignore_nulls else None, string, stream=stream)
    if isinstance(separator, plc.Column):
        separator_narep = plc.Scalar.from_py(None, string, stream=stream)
        separator_value: plc.Column | plc.Scalar = separator
    else:
        # For a scalar separator, pylibcudf passes separator_narep to libcudf as the narep of null strings: https://github.com/NVIDIA/cudf/blob/d3654d6f27e4fb30c0424a57cdbebfb1e73d3027/python/pylibcudf/pylibcudf/strings/combine.pyx#L265-L279
        separator_narep = narep
        separator_value = plc.Scalar.from_py(separator, string, stream=stream)
    return plc.strings.combine.join_list_elements(
        lists,
        separator_value,
        separator_narep,
        narep,
        plc.strings.combine.SeparatorOnNulls.NO,
        plc.strings.combine.OutputIfEmptyList.EMPTY_STRING,
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
            Name.Join,
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
        if self.name is ListFunction.Name.Join:
            _, separator = self.children
            if separator.dtype.id() != plc.TypeId.STRING:
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for a non-String separator"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Join:
            (ignore_nulls,) = self.options
            lists, separator = self.children
            column = lists.evaluate(df, context=context)
            if isinstance(separator, Literal):
                return Column(
                    _join(
                        column.obj,
                        separator.value,
                        ignore_nulls=ignore_nulls,
                        stream=df.stream,
                    ),
                    dtype=self.dtype,
                )
            column, separators = broadcast(
                column, separator.evaluate(df, context=context), stream=df.stream
            )
            return Column(
                _join(
                    column.obj,
                    separators.obj,
                    ignore_nulls=ignore_nulls,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
