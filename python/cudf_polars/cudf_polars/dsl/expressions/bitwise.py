# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from cudf_polars.containers import DataFrame, DataType

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

    _supported_ops: ClassVar[frozenset[Name]] = frozenset({Name.CountZeros})
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
        ones = plc.unary.unary_operation(
            column.obj, plc.unary.UnaryOperator.BIT_COUNT, stream=df.stream
        )
        width = plc.Scalar.from_py(
            self._bit_widths[child.dtype.id()], self.dtype.plc_type, stream=df.stream
        )
        zeros = plc.binaryop.binary_operation(
            width,
            ones,
            plc.binaryop.BinaryOperator.SUB,
            self.dtype.plc_type,
            stream=df.stream,
        )
        return Column(zeros, dtype=self.dtype)
