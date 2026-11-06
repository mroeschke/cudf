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

    from cudf_polars.containers import DataFrame

__all__ = ["BitwiseFunction"]


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

    _supported_ops: ClassVar[frozenset[Name]] = frozenset({Name.LeadingZeros})
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
        width = self._bit_widths[child.dtype.id()]
        unsigned_dtype = self._unsigned_types.get(child.dtype.id(), child.dtype)
        smeared = column.astype(unsigned_dtype, df.stream).obj
        for shift in (1 << i for i in range(width.bit_length() - 1)):
            shifted = plc.binaryop.binary_operation(
                smeared,
                plc.Scalar.from_py(shift, unsigned_dtype.plc_type, stream=df.stream),
                plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
                unsigned_dtype.plc_type,
                stream=df.stream,
            )
            smeared = plc.binaryop.binary_operation(
                smeared,
                shifted,
                plc.binaryop.BinaryOperator.BITWISE_OR,
                unsigned_dtype.plc_type,
                stream=df.stream,
            )
        significant_bits = plc.unary.unary_operation(
            smeared, plc.unary.UnaryOperator.BIT_COUNT, stream=df.stream
        )
        leading_zeros = plc.binaryop.binary_operation(
            plc.Scalar.from_py(width, self.dtype.plc_type, stream=df.stream),
            significant_bits,
            plc.binaryop.BinaryOperator.SUB,
            self.dtype.plc_type,
            stream=df.stream,
        )
        return Column(leading_zeros, dtype=self.dtype)
