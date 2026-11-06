# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast

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
            if any(child.dtype.id() != plc.TypeId.LIST for child in self.children):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for non-list inputs"
                )
            if any(child.dtype != self.dtype for child in self.children):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for inputs of different dtypes"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Concat:
            columns = broadcast(
                *(child.evaluate(df, context=context) for child in self.children),
                stream=df.stream,
            )
            return Column(
                _concat([column.obj for column in columns], df.stream),
                dtype=self.dtype,
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
