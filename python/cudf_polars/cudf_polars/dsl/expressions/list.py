# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""List DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.containers.column import with_mask_from
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.utils.dtypes import make_all_null_column

if TYPE_CHECKING:
    from typing import Self

    from polars import _plr as polars

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["ListFunction"]


def _shift(
    column: Column, child: plc.Column, n: int, stream: Stream
) -> tuple[plc.Column, plc.Column, plc.Column | None]:
    """
    Shift the elements ``child`` of every list of ``column`` by ``n`` positions.

    ``child`` has one value per element of the sliced child of ``column``.
    Return the offsets of the lists starting at 0, ``child`` shifted across
    list boundaries, and a mask that is false where the shifted value comes
    from another list (``None`` if ``n`` is 0).

    ``column`` and ``child`` must be valid on ``stream``; synchronizes
    ``stream``.
    """
    n = max(-child.size(), min(n, child.size()))
    sizes = plc.lists.count_elements(column.obj.with_mask(None, 0), stream=stream)
    if n >= 0:
        starts = plc.Column.from_scalar(
            plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
            column.size,
            stream=stream,
        )
        steps = None
    else:
        starts = plc.binaryop.binary_operation(
            sizes,
            plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream),
            plc.binaryop.BinaryOperator.SUB,
            plc.types.SIZE_TYPE,
            stream=stream,
        )
        steps = plc.Column.from_scalar(
            plc.Scalar.from_py(-1, plc.types.SIZE_TYPE, stream=stream),
            column.size,
            stream=stream,
        )
    positions = plc.lists.sequences(starts, sizes, steps, stream=stream).list_view()
    shifted = plc.copying.shift(
        child, n, plc.Scalar.from_py(None, child.type(), stream=stream), stream=stream
    )
    if n == 0:
        return positions.offsets(), shifted, None
    keep = plc.binaryop.binary_operation(
        positions.child(),
        plc.Scalar.from_py(abs(n), plc.types.SIZE_TYPE, stream=stream),
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
        plc.DataType(plc.TypeId.BOOL8),
        stream=stream,
    )
    return positions.offsets(), shifted, keep


def _list_shift(column: Column, n: int, stream: Stream) -> plc.Column:
    """
    Shift the elements of every list of ``column`` by ``n`` positions.

    ``column`` must be valid on ``stream``; synchronizes ``stream``.
    """
    child = column.obj.list_view().get_sliced_child(stream=stream)
    indices = plc.filling.sequence(
        child.size(),
        plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
        plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream),
        stream=stream,
    )
    offsets, indices, keep = _shift(column, indices, n, stream)
    if keep is not None:
        indices = plc.copying.copy_if_else(
            indices,
            plc.Scalar.from_py(child.size(), plc.types.SIZE_TYPE, stream=stream),
            keep,
            stream=stream,
        )
    (shifted,) = plc.copying.gather(
        plc.Table([child]),
        indices,
        plc.copying.OutOfBoundsPolicy.NULLIFY,
        stream=stream,
    ).columns()
    return with_mask_from(
        plc.Column(
            plc.DataType(plc.TypeId.LIST),
            column.size,
            None,
            None,
            0,
            0,
            [offsets, shifted],
        ),
        column,
        stream,
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
            Name.Shift,
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
        if self.name is ListFunction.Name.Shift:
            _, n = self.children
            if not isinstance(n, Literal):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for non-literal n"
                )
            if n.value is not None and not plc.traits.is_integral(n.dtype.plc_type):
                raise NotImplementedError(
                    f"ListFunction.{self.name.name} is not supported for "
                    f"n {n.value!r} of type {n.dtype.polars_type}"
                )
            self.options = (
                None if n.value is None or n.value >= 2**63 else int(n.value),
            )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is ListFunction.Name.Shift:
            column = self.children[0].evaluate(df, context=context)
            (n,) = self.options
            if n is None:
                return Column(
                    make_all_null_column(self.dtype, column.size, df.stream),
                    dtype=self.dtype,
                )
            if n == 0:
                return Column(column.obj, dtype=self.dtype)
            return Column(_list_shift(column, n, df.stream), dtype=self.dtype)
        raise NotImplementedError(
            f"ListFunction.{self.name.name} is not supported"
        )  # pragma: no cover
