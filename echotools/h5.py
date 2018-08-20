# UNLOADING is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UNLOADING is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UNLOADING. If not, see <http://www.gnu.org/licenses/>.

__author__ = "Henrik Finsberg (henriknf@simula.no)"
import os
import h5py
import numpy as np
try:
    from .utils import make_logger
except ImportError:
    from utils import make_logger

logger = make_logger(__name__)

try:
    import dolfin as df
except ImportError:
    has_dolfin = False
    logger.warning('Dolfin is not installed')
else:
    has_dolfin = True


def dolfin_to_hd5(obj, h5name, time=None, comm=None, name=None):
    """
    Save object to and HDF file.
    """
    if comm is None:
        comm = df.mpi_comm_world()
        
    name = obj.name() if name is None else name
    logger.info("Save {0} to {1}:{0}/{2}".format(name,
                                                 h5name,
                                                 time))


    group = name if time is None else "/".join([name, str(time)])
    file_mode = "a" if os.path.isfile(h5name) else "w"

    if isinstance(obj, df.Function):

        if obj.value_size() == 1:
            return save_scalar_function(comm, obj, h5name, group, file_mode)
        else:
            return save_vector_function(comm, obj, h5name, group, file_mode)
        
    elif isinstance(obj, df.Mesh):
        with df.HDF5File(comm, h5name, file_mode) as h5file:
            h5file.write(obj, group)

        if obj.geometry().dim() == 3:
            coords = "XYZ"
        elif obj.geometry().dim() == 2:
            coords = "XY"
        else:
            coords = "X"

        return {"h5group": group,
                "cell_indices":"/".join([group, "cell_indices"]),
                "coordinates":"/".join([group, "coordinates"]),
                "topology":"/".join([group, "topology"]),
                "ncells":  obj.num_cells(),
                "nverts": obj.num_vertices(),
                "coords":coords,
                "type":"mesh",
                "cell": str(obj.ufl_cell()).capitalize(),
                "top_dim": obj.topology().dim()+1,
                "geo_dim": obj.geometry().dim()}
    else:
        
        raise ValueError("Unknow type {}".format(type(obj)))


def save_scalar_function(comm, obj, h5name, h5group="", file_mode="w"):

    element = obj.ufl_element()
    V = obj.function_space()
    dim = V.mesh().geometry().dim()
    coords_tmp = V.tabulate_dof_coordinates()
    coords = coords_tmp.reshape((-1, dim))
    dm = V.dofmap()
    top = [dm.cell_dofs(i) for i in range(V.mesh().num_cells())]

    obj_arr = obj.vector().get_local()
    vecs = np.array(obj_arr).T

    coord_group = "/".join([h5group, "coordinates"])
    vector_group = "/".join([h5group, "vector"])
    top_group = "/".join([h5group, "topology"])
    if comm.rank == 0:
        with h5py.File(h5name, file_mode) as h5file:
            if h5group in h5file:
                del h5file[h5group]

            h5file.create_dataset(coord_group, data=coords)
            h5file.create_dataset(vector_group, data=vecs)
            h5file.create_dataset(top_group, data=top)

    element = obj.ufl_element()

    if dim == 3:
        coords = "XYZ"
    elif dim == 2:
        coords = "XY"
    else:
        coords = "X"

    return {"h5group": h5group,
            "coordinates": coord_group,
            "vector": vector_group,
            "nverts": obj.vector().size(),
            "top_dim": V.mesh().topology().dim()+1,
            "topology": top_group,
            "cell": str(V.mesh().ufl_cell()).capitalize(),
            "ncells": V.mesh().num_cells(),
            "dim": 1,
            "family": element.family(),
            "geo_dim": dim,
            "coords": coords,
            "degree": element.degree(),
            "type": "scalar"}


def save_vector_function(comm, obj, h5name, h5group="", file_mode="w"):

    element = obj.ufl_element()
    V = obj.function_space()
    gs = obj.split(deepcopy=True)
    W = V.sub(0).collapse()
    dim = V.mesh().geometry().dim()
    coords_tmp = W.tabulate_dof_coordinates()
    coords = coords_tmp.reshape((-1, dim))
    dm = W.dofmap()

    us = [g.vector().get_local() for g in gs]
    vecs = np.array(us).T
    top = [dm.cell_dofs(i) for i in range(W.mesh().num_cells())]

    coord_group = "/".join([h5group, "coordinates"])
    top_group = "/".join([h5group, "topology"])
    vector_group = "/".join([h5group, "vector"])
    if comm.rank == 0:
        with h5py.File(h5name, file_mode) as h5file:

            if h5group in h5file:
                del h5file[h5group]
            h5file.create_dataset(coord_group, data=coords)
            h5file.create_dataset(vector_group, data=vecs)
            h5file.create_dataset(top_group, data=top)

    if dim == 3:
        coords = "XYZ"
    elif dim == 2:
        coords = "XY"
    else:
        coords = "X"

    return {"h5group": h5group,
            "coordinates": coord_group,
            "vector": vector_group,
            "top_dim": W.mesh().topology().dim()+1,
            "topology": top_group,
            "cell": str(W.mesh().ufl_cell()).capitalize(),
            "ncells": W.mesh().num_cells(),
            "nverts": int(obj.vector().size()/dim),
            "dim": dim,
            "family": element.family(),
            "geo_dim": dim,
            "coords": coords,
            "degree": element.degree(),
            "type": "vector"}


def load_dict_from_h5(fname, h5group=""):
    """
    Load the given h5file into
    a dictionary
    """
    assert os.path.isfile(fname), \
        "File {} does not exist".format(fname)

    with h5py.File(fname, "r") as h5file:

        def h52dict(hdf):
            if isinstance(hdf, h5py._hl.group.Group):
                t = {}
                for key in hdf.keys():
                    t[str(key)] = h52dict(hdf[key])
            elif isinstance(hdf, h5py._hl.dataset.Dataset):
                t = np.array(hdf)

            return t

        if h5group != "" and h5group in h5file:
            d = h52dict(h5file[h5group])
        else:
            d = h52dict(h5file)

    return d


if __name__ == "__main__":

    mesh = df.UnitSquareMesh(3, 3)

    spaces = ["DG_0", "DG_1", "CG_1", "CG_2", "R_0", "Quadrature_2"]

    finite_elements = [df.FiniteElement(s.split("_")[0],
                                        mesh.ufl_cell(),
                                        int(s.split("_")[1]),
                                        quad_scheme="default") for s in spaces]
    scalar_spaces = [df.FunctionSpace(mesh, el) for el in finite_elements]
    scalar_functions = [df.Function(V, name="Scalar_{}".format(s))
                        for (V, s) in zip(scalar_spaces, spaces)]

    vector_elements = [df.VectorElement(s.split("_")[0],
                                     mesh.ufl_cell(),
                                     int(s.split("_")[1]),
                                     quad_scheme="default") for s in spaces]
    vector_spaces = [df.FunctionSpace(mesh, el) for el in vector_elements]
    vector_functions = [df.Function(V, name="Vector_{}".format(s))
                        for (V, s) in zip(vector_spaces, spaces)]

    h5name = "test_unit_square.h5"

    for f in scalar_functions:
        dolfin_to_hd5(f, h5name, 0.0)

    for f in vector_functions:
        dolfin_to_hd5(f, h5name, 0.0)

    f = df.XDMFFile(df.mpi_comm_world(), "test.xdmf")
    scalar_functions[0].vector()[:] \
        = np.linspace(0, 1, len(scalar_functions[0].vector()))
    vector_functions[0].vector()[:] \
        = np.linspace(0, 1, len(vector_functions[0].vector()))
    # f.write(scalar_functions[2])
    f.write(vector_functions[0])
