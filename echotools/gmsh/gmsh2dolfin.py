"FIXME"
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
from dolfin import *
import numpy as np
from itertools import izip
from generate_points import *
from make_affine_mapping import *
from reference_topology import *
from scipy.spatial import cKDTree

__all__ = ["gmsh2dolfin"]

def _tabulate_all_gmsh_dofs(gmsh):

    # Gmsh provides the position of the dofs in the deformed state
    # and not in the reference.
    # Here we construct the list of all reference dofs in the same
    # order of the nodes.

    # the type of the cell
    cell_shape = gmsh.cell_shape()
    vert_per_cell = gmsh_shape[cell_shape].num_vertices()

    # cell dofs in the reference element
    cell_ref_dofs = np.array(generate_points(cell_shape, gmsh.mesh_order))
    cell_ref_vert = cell_ref_dofs[:vert_per_cell]

    # cell with dofs
    cells, cells_dofs = gmsh.cells(), gmsh.cells(True)
    nodes = gmsh.dof_coords()
    global_dofs = np.zeros(nodes.shape)

    for cell_vert, cell_dofs in izip(nodes[cells], cells_dofs):
        # create the affine mapping
        A, b = make_affine_mapping(cell_ref_vert, cell_vert)
        # maps the dofs
        dofs = cell_ref_dofs.dot(A.T) + np.array([b,]*cell_ref_dofs.shape[0])
        global_dofs[cell_dofs,:] = dofs

    return global_dofs


def gmsh2dolfin(gmsh, comm=None, use_coords=False):
    """
    FIXME
    """

    comm = comm if comm is not None else mpi_comm_world()

    # ---------------------------
    # Construction of DOLFIN mesh
    # ---------------------------
    mesh = Mesh(comm)
    editor = MeshEditor()

    # The embedding space dimension in always 3
    # in gmsh, but not in DOLFIN.
    mdim  = gmsh.topo_dim
    gdim  = gmsh.geom_dim
    shape = gmsh.cell_shape()
    editor.open(mesh, shape, mdim, gdim)

    # vertices are not indexed in a specific order,
    # so a map is necessary to build the mesh
    msh2dolfin = {}
    vert = gmsh.vertices()
    dofs = gmsh.dof_coords()
    editor.init_vertices(len(vert))
    for dolfin_id, msh_id in enumerate(vert):
        editor.add_vertex(dolfin_id, dofs[msh_id,:])
        msh2dolfin[msh_id] = dolfin_id

    # cells
    # translate gmsh ids to dolfin ids for each cell
    cells = np.vectorize(msh2dolfin.get, otypes=[np.uintp])(gmsh.cells())
    # cell vertices needs to be sorted (ufc assumption)
    if shape not in [ "quadrilateral", "hexahedron" ]:
        cells.sort(axis=1)
    editor.init_cells(len(cells))
    for idx, vert in enumerate(cells):
        editor.add_cell(idx, vert)

    # Finalise the mesh
    editor.close(False)

    # -----------------
    # Physical entities
    # -----------------
    # cells handled directly
    mesh.domains().init(3)
    for idx, m in enumerate(gmsh.entity_markers(0)):
        if m == 0: continue
        if not mesh.domains().set_marker((idx, m), gmsh.topo_dim):
            raise RuntimeError("Something wrong during cell marking")

    # other entities
    for codim in range(1, gmsh.topo_dim + 1):
        entities = gmsh.entities(codim)
        if entities is None: continue
        # map the numbering of the vertices
        ent = []
        for e in entities:
            try:
                ent.append([ msh2dolfin[c] for c in e ])
            except KeyError:
                # we skip marked entities which are not belonging to
                # the mesh (e.g. external points)
                continue
        ent = np.array(ent, dtype = np.uintp)
        ent_dim = mdim - codim
        # to look for the entity we need the vertex to entity connectivity
        # map and the entity to cell map
        mesh.init(0, ent_dim)
        mesh.init(ent_dim, mdim)
        vert_to_enti = mesh.topology()(0, ent_dim)
        enti_to_cell = mesh.topology()(ent_dim, mdim)

        # the entity id is obtained by intersecating all the entities
        # connected to the given vertices.
        # The result should always be one entity.
        eids = np.concatenate([ reduce(np.intersect1d, map(vert_to_enti, e)) \
                for e in ent ])
        
        for e, marker in izip(eids, gmsh.entity_markers(codim)):
            if marker == 0: continue
            # each cell containing the entity is marked
            for c in enti_to_cell(e):
                mesh.domains().set_marker((e, marker), ent_dim)

    # gmsh human readable names for the markers
    markers = dict(gmsh.marker_name)
    if len(markers) == 1 and 0 in markers:
        markers = {}

    # -----------------------
    # Construction of the map
    # -----------------------

    return mesh, markers

    # if gmsh.mesh_order == 1:
    #     return mesh.ufl_domain(), markers

    # # we need to compare the list of dofs generated by gmsh and
    # # the list from ufc.
    # V = VectorFunctionSpace(mesh, "P", gmsh.mesh_order, gdim)

    # # first I need to reorganize the ufc dofs
    # idx = np.column_stack([ V.sub(i).dofmap().dofs()
    #             for i in range(0, gdim) ])
    # ufc_dofs = V.dofmap().tabulate_all_coordinates(mesh).reshape(-1, gdim)
    # ufc_dofs = ufc_dofs[idx[:,0],:]

    # # gmsh dofs are generated from topology
    # msh_dofs = _tabulate_all_gmsh_dofs(gmsh)

    # # now we compute the permutation
    # tree = cKDTree(msh_dofs)
    # msh2ufc = tree.query(ufc_dofs)[1]

    # phi = Function(V)
    # if use_coords:
    #     displ = gmsh.dof_coords()
    # else:
    #     displ = gmsh.dof_coords() - msh_dofs

    # for i in range(0, gdim):
    #     phi.vector()[idx[:,i].copy()] = (displ[msh2ufc, i]).astype(np.float)

    # return Domain(phi), markers

