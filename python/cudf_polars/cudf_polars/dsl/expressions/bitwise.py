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

    _supported_ops: ClassVar[frozenset[Name]] = frozenset({Name.TrailingOnes})
    _unsigned_types: ClassVar[Mapping[plc.TypeId, plc.TypeId]] = MappingProxyType(
        {
            plc.TypeId.BOOL8: plc.TypeId.UINT8,
            plc.TypeId.INT8: plc.TypeId.UINT8,
            plc.TypeId.INT16: plc.TypeId.UINT16,
            plc.TypeId.INT32: plc.TypeId.UINT32,
            plc.TypeId.INT64: plc.TypeId.UINT64,
            plc.TypeId.UINT8: plc.TypeId.UINT8,
            plc.TypeId.UINT16: plc.TypeId.UINT16,
            plc.TypeId.UINT32: plc.TypeId.UINT32,
            plc.TypeId.UINT64: plc.TypeId.UINT64,
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
        if child.dtype.id() not in self._unsigned_types:
            raise NotImplementedError(
                f"BitwiseFunction.{self.name.name} is not supported "
                f"for {child.dtype.polars_type} input"
            )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context).obj
        unsigned_type = plc.DataType(self._unsigned_types[child.dtype.id()])
        if column.type() != unsigned_type:
            column = plc.unary.cast(column, unsigned_type, stream=df.stream)
        Op = plc.expressions.Operation
        ast_op = plc.expressions.ASTOperator
        col = plc.expressions.ColumnReference(0)
        one = plc.expressions.Literal(
            plc.Scalar.from_py(1, unsigned_type, stream=df.stream)
        )
        lowest_ones = plc.transform.compute_column(
            plc.Table([column]),
            Op(
                ast_op.BIT_INVERT,
                Op(
                    ast_op.BITWISE_OR,
                    Op(ast_op.BIT_INVERT, col),
                    Op(ast_op.ADD, col, one),
                ),
            ),
            stream=df.stream,
        )
        ones = plc.unary.unary_operation(
            lowest_ones, plc.unary.UnaryOperator.BIT_COUNT, stream=df.stream
        )
        return Column(ones, dtype=DataType(pl.Int32())).astype(self.dtype, df.stream)
