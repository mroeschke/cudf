# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame

__all__ = ["BitwiseFunction"]


def _leading_zeros(
    values: plc.Column, width: int, dtype: plc.DataType, stream: Stream
) -> plc.Column:
    """
    Count the leading zeros of each value.

    ``values`` is an unsigned integer column that is ``width`` bits wide and
    must be valid on ``stream``. The work is enqueued on ``stream`` with no
    host synchronization. The result has type ``dtype``.
    """
    unsigned = values.type()
    shift = 1
    while shift < width:
        shifted = plc.binaryop.binary_operation(
            values,
            plc.Scalar.from_py(shift, unsigned, stream=stream),
            plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
            unsigned,
            stream=stream,
        )
        values = plc.binaryop.binary_operation(
            values,
            shifted,
            plc.binaryop.BinaryOperator.BITWISE_OR,
            unsigned,
            stream=stream,
        )
        shift *= 2
    bit_length = plc.unary.unary_operation(
        values, plc.unary.UnaryOperator.BIT_COUNT, stream=stream
    )
    return plc.binaryop.binary_operation(
        plc.Scalar.from_py(width, dtype, stream=stream),
        bit_length,
        plc.binaryop.BinaryOperator.SUB,
        dtype,
        stream=stream,
    )


class BitwiseFunction(Expr):
    """Bit counting functions of integer and boolean columns."""

    class Name(IntEnum):
        """Internal and picklable representation of polars' `BitwiseFunction`."""

        CountOnes = auto()
        CountZeros = auto()
        LeadingOnes = auto()
        LeadingZeros = auto()
        TrailingOnes = auto()
        TrailingZeros = auto()
        And = auto()
        Or = auto()
        Xor = auto()

        @classmethod
        def from_polars(cls, obj: Any) -> Self:
            """Convert from polars' `BitwiseFunction`."""
            return cls[str(obj).removeprefix("BitwiseFunction.")]

    __slots__ = ("name",)
    _non_child = ("dtype", "name")

    _supported_ops: ClassVar[frozenset[Name]] = frozenset({Name.LeadingOnes})
    _bit_widths: ClassVar[Mapping[plc.TypeId, int]] = MappingProxyType(
        {
            plc.TypeId.BOOL8: 1,
            plc.TypeId.INT8: 8,
            plc.TypeId.INT16: 16,
            plc.TypeId.INT32: 32,
            plc.TypeId.INT64: 64,
            plc.TypeId.UINT8: 8,
            plc.TypeId.UINT16: 16,
            plc.TypeId.UINT32: 32,
            plc.TypeId.UINT64: 64,
        }
    )
    _unsigned_types: ClassVar[Mapping[plc.TypeId, DataType]] = MappingProxyType(
        {
            plc.TypeId.INT8: DataType(pl.UInt8()),
            plc.TypeId.INT16: DataType(pl.UInt16()),
            plc.TypeId.INT32: DataType(pl.UInt32()),
            plc.TypeId.INT64: DataType(pl.UInt64()),
        }
    )

    def __init__(
        self,
        dtype: DataType,
        name: BitwiseFunction.Name,
        child: Expr,
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.children = (child,)
        self.is_pointwise = True
        if self.name not in self._supported_ops:
            raise NotImplementedError(
                f"BitwiseFunction.{self.name.name} is not supported"
            )
        if child.dtype.id() not in self._bit_widths:
            raise NotImplementedError(
                f"BitwiseFunction.{self.name.name} is not supported "
                f"for {child.dtype.polars_type} input"
            )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        if child.dtype.id() == plc.TypeId.BOOL8:
            # A Boolean has one bit, so its leading ones count is its value.
            return column.astype(self.dtype, df.stream)
        unsigned_dtype = self._unsigned_types.get(child.dtype.id(), child.dtype)
        inverted = plc.unary.unary_operation(
            column.astype(unsigned_dtype, df.stream).obj,
            plc.unary.UnaryOperator.BIT_INVERT,
            stream=df.stream,
        )
        return Column(
            _leading_zeros(
                inverted,
                self._bit_widths[child.dtype.id()],
                self.dtype.plc_type,
                df.stream,
            ),
            dtype=self.dtype,
        )
