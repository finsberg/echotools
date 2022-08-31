"""FIXME:"""
# Copyright (C) 2016 Simone Pezzuto
#
# This file is part of FENICSHOTOOLS.
#
# FENICSHOTOOLS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FENICSHOTOOLS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FENICSHOTOOLS. If not, see <http://www.gnu.org/licenses/>.
import numpy as np

# -------- from FIAT/reference_element.py --------
def make_affine_mapping(xs, ys):
    """Constructs (A,b) such that x --> A * x + b is the affine
    mapping from the simplex defined by xs to the simplex defined by
    ys."""

    import numpy

    dim_x = len(xs[0])
    dim_y = len(ys[0])

    if len(xs) != len(ys):
        raise Exception("")

    # find A in R^{dim_y,dim_x}, b in R^{dim_y} such that
    # A xs[i] + b = ys[i] for all i

    mat = numpy.zeros((dim_x * dim_y + dim_y, dim_x * dim_y + dim_y), "d")
    rhs = numpy.zeros((dim_x * dim_y + dim_y,), "d")

    # loop over points
    for i in range(len(xs)):
        # loop over components of each A * point + b
        for j in range(dim_y):
            row_cur = i * dim_y + j
            col_start = dim_x * j
            col_finish = col_start + dim_x
            mat[row_cur, col_start:col_finish] = numpy.array(xs[i])
            rhs[row_cur] = ys[i][j]
            # need to get terms related to b
            mat[row_cur, dim_y * dim_x + j] = 1.0

    sol = numpy.linalg.solve(mat, rhs)

    A = numpy.reshape(sol[: dim_x * dim_y], (dim_y, dim_x))
    b = sol[dim_x * dim_y :]

    return A, b
