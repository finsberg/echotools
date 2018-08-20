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
from reference_topology import gmsh_shape

__all__ = ["generate_points"]

# ------ from Numerics/pointsGenerators.cpp ------

def generate_points(shape, order) :
    "FIXME"

    mono = { 'line'          : _generate_monomial_line,
             'triangle'      : _generate_monomial_triangle,
             'quadrilateral' : None,
             'tetrahedron'   : _generate_monomial_tetrahedron,
             'hexahedron'    : None }[shape](order)

    if shape == 'line' :
        return map(lambda v : float(v)/order, mono)
    else :
        return zip(*map(lambda v : map(lambda vv : float(vv)/order, v), mono))

def _generate_monomial_line(order) :
    mono = [0]
    if order > 0 : 
        mono += [order] + range(1, order)
    return mono

def _generate_monomial_triangle(order) :
    mono = [ [0], [0] ]
    if order > 0 :
        mono[0] += [order, 0]
        mono[1] += [0, order]
    if order > 1 :
        for v0, v1 in gmsh_shape['triangle'].topology()[1] :
            u00, u01 = mono[0][v0], mono[0][v1]
            u10, u11 = mono[1][v0], mono[1][v1]
            mono[0] += map(lambda i: u00 + (u01-u00)/order*i, range(1, order))
            mono[1] += map(lambda i: u10 + (u11-u10)/order*i, range(1, order))
    if order > 2 :
        inner = _generate_monomial_triangle(order-3)
        mono[0] += map(lambda v: v+1, inner[0])
        mono[1] += map(lambda v: v+1, inner[1])
    return mono

def _generate_monomial_tetrahedron(order) :
    # P0+
    mono = [ [0], [0], [0] ]
    if order > 0 :
        # P1+ - (+vertices)
        mono[0] += [order, 0, 0]
        mono[1] += [0, order, 0]
        mono[2] += [0, 0, order]
    if order > 1 :
        # P2+ - (+edges)
        for v0, v1 in gmsh_shape['tetrahedron'].topology()[1] :
            u00, u01 = mono[0][v0], mono[0][v1]
            u10, u11 = mono[1][v0], mono[1][v1]
            u20, u21 = mono[2][v0], mono[2][v1]
            mono[0] += map(lambda i: u00 + (u01-u00)/order*i, range(1, order))
            mono[1] += map(lambda i: u10 + (u11-u10)/order*i, range(1, order))
            mono[2] += map(lambda i: u20 + (u21-u20)/order*i, range(1, order))
    if order > 2 :
        # P3+ - (+faces)
        dudv = _generate_monomial_triangle(order-3)
        dudv[0] = map(lambda v: v+1, dudv[0])
        dudv[1] = map(lambda v: v+1, dudv[1])
        for v0, v1, v2 in gmsh_shape['tetrahedron'].topology()[2] :
            u00, u01, u02 = mono[0][v0], mono[0][v1], mono[0][v2]
            u10, u11, u12 = mono[1][v0], mono[1][v1], mono[1][v2]
            u20, u21, u22 = mono[2][v0], mono[2][v1], mono[2][v2]
            for i in xrange(0, len(dudv[0])) :
                mono[0] += [ u00 + (u01-u00)/order*dudv[0][i] \
                             + (u02-u00)/order*dudv[1][i] ]
                mono[1] += [ u10 + (u11-u10)/order*dudv[0][i] \
                             + (u12-u10)/order*dudv[1][i] ]
                mono[2] += [ u20 + (u21-u20)/order*dudv[0][i] \
                             + (u22-u20)/order*dudv[1][i] ]
    if order > 3 :
        # P4+ - (+cell)
        inner = _generate_monomial_tetrahedron(order-4)
        mono[0] += map(lambda v: v+1, inner[0])
        mono[1] += map(lambda v: v+1, inner[1])
        mono[2] += map(lambda v: v+1, inner[2])
    return mono

