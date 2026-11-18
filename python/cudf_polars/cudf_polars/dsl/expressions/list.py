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
from cudf_polars.utils.dtypes import make_all_null_column

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]

_UINT32_MAX = 2**32 - 1


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


def _gather_every(
    column: plc.Column, n: int, offset: int, stream: Stream
) -> plc.Column:
    """
    Keep every ``n``-th element of every list, starting at position ``offset``.

    ``n`` is positive. ``column`` must be valid on ``stream``.
    """
    if n == 1 and offset == 0:
        return column
    size_max = 2**31 - 1
    sizes = plc.lists.count_elements(column.with_mask(None, 0), stream=stream)
    positions = plc.lists.sequences(
        plc.Column.from_scalar(
            plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
            column.size(),
            stream=stream,
        ),
        sizes,
        stream=stream,
    )
    Op = plc.expressions.Operation
    AST = plc.expressions.ASTOperator
    position = plc.expressions.ColumnReference(0)

    def lit(value: int) -> plc.expressions.Literal:
        return plc.expressions.Literal(
            plc.Scalar.from_py(min(value, size_max), plc.types.SIZE_TYPE, stream=stream)
        )

    keep = plc.transform.compute_column(
        plc.Table([positions.list_view().child()]),
        Op(
            AST.LOGICAL_AND,
            Op(AST.GREATER_EQUAL, position, lit(offset)),
            Op(
                AST.EQUAL,
                Op(AST.MOD, Op(AST.SUB, position, lit(offset)), lit(n)),
                lit(0),
            ),
        ),
        stream=stream,
    )
    return plc.lists.apply_retention_mask(
        column, _bool_list_like(positions, keep), stream=stream
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
            Name.GatherEvery,
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
        if self.name is ListFunction.Name.GatherEvery:
            _, n, offset = self.children
            if not (isinstance(n, Literal) and isinstance(offset, Literal)):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for "
                    "non-literal n or offset"
                )
            for arg in (n, offset):
                if arg.value is not None and not (
                    plc.traits.is_integral(arg.dtype.plc_type)
                    and 0 <= arg.value <= _UINT32_MAX
                ):
                    raise NotImplementedError(
                        f"ListFunction.{self.name.name} is not supported for "
                        f"n or offset {arg.value!r} of type {arg.dtype.polars_type}"
                    )
            self.options = tuple(
                None if arg.value is None else int(arg.value) for arg in (n, offset)
            )
            if self.options[0] == 0:
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for n=0"
                )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.GatherEvery:
            column = self.children[0].evaluate(df, context=context)
            n, offset = self.options
            if n is None or offset is None:
                return Column(
                    make_all_null_column(self.dtype, column.size, df.stream),
                    dtype=self.dtype,
                )
            return Column(
                _gather_every(column.obj, n, offset, df.stream), dtype=self.dtype
            )
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
