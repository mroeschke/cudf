# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.libcudf cimport contiguous_split as cpp_contiguous_split

from .table cimport Table


cpdef list contiguous_split(Table input_table, list splits):
    """
    Splits a table into a list of of contiguous values according to
    a sequence of indices.

    For details, see :cpp:func:`contiguous_split`.

    Parameters
    ----------
    input_table : Table
        Table of columns to split.
    splits : list
        A list of integer indices where the table will be split.

    Returns
    -------
    list[dict[]]
        The necessary number of bytes
    """
    with nogil:
        return cpp_contiguous_split.contiguous_split(input_table.view(), splits)
