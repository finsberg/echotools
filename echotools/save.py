# from xdmf import *
import numpy as np

import h5py
import os
import dolfin as df
from .h5 import dolfin_to_hd5
from . import xdmf


def fun_to_xdmf(fun, fname, name="function"):

    h5name = "{}.h5".format(fname)
    dolfin_to_hd5(fun, h5name, name=name)

    dim = fun.function_space().mesh().geometry().dim()

    if fun.value_size() == 1:
        nverts = len(fun.vector())
        fun_str = xdmf.scalar_attribute.format(name=name,
                                               nverts=nverts,
                                               center="Node",
                                               h5group="/".join([name,
                                                                 "vector"]),
                                               dim=1,
                                               h5name=os.path.basename(h5name))
    else:
        nverts = int(len(fun.vector())/dim)
        fun_str = xdmf.vector_attribute.format(name=name,
                                               nverts=nverts,
                                               dim=dim,
                                               h5group="/".join([name,
                                                                 "vector"]),
                                               center="Node",
                                               h5name=os.path.basename(h5name))
        
    fun_top = xdmf.topology_polyvert.format(nverts=nverts)
    fun_geo = xdmf.geometry.format(nverts=nverts,
                                   dim=dim,
                                   coords="XYZ",
                                   h5group="/".join([name, "coordinates"]),
                                   h5name=os.path.basename(h5name))

    fun_entry = xdmf.entry.format(frame=fun_geo + fun_top + fun_str, iter=0)
    T = xdmf.body.format(body=fun_entry,
                         name="Visualzation of {}".format(name))

    with open("{}.xdmf".format(fname), "w") as f:
        f.write(T)


def fiber_to_xdmf(fun, fname, comm=None):

    if comm is None:
        comm = df.mpi_comm_world()

    h5name = "{}.h5".format(fname)
    import os
    if os.path.isfile(h5name):
        if comm.rank == 0:
            os.unlink(h5name)

    dolfin_to_hd5(fun, h5name, name="fiber")

    fx = fun.split(deepcopy=True)[0]
    fx_arr = fx.vector().array()
    scalar = np.arcsin(fx_arr)*180/np.pi
    with h5py.File(h5name, "a") as h5file:
        if comm.rank == 0:
            h5file.create_dataset("fiber/scalar",
                                  data=scalar)

    dim = fun.function_space().mesh().geometry().dim()
    nverts = int(fun.vector().size()/dim)
    name = "fiber"
    
    fun_scal = xdmf.scalar_attribute.format(name="angle",
                                            nverts=nverts,
                                            center="Node",
                                            h5group="/".join([name, "scalar"]),
                                            dim=1,
                                            h5name=os.path.basename(h5name))

    fun_vec = xdmf.vector_attribute.format(name=name,
                                           nverts=nverts,
                                           dim=dim,
                                           center="Node",
                                           h5group="/".join([name, "vector"]),
                                           h5name=os.path.basename(h5name))

    fun_top = xdmf.topology_polyvert.format(nverts=nverts)
    fun_geo = xdmf.geometry.format(nverts=nverts,
                                   dim=dim,
                                   coords="XYZ",
                                   h5group="/".join([name, "coordinates"]),
                                   h5name=os.path.basename(h5name))

    fname = fun_geo + fun_top + fun_scal+fun_vec
    fun_entry = xdmf.entry_single.format(frame=fname,
                                         iter=0)
    T = xdmf.body.format(body=fun_entry,
                         name="Visualzation of {}".format(name))

    with open("{}.xdmf".format(fname), "w") as f:
        f.write(T)


class MyXDMFFile:
    """
    A class for saving dolfin functions to HDF5 and
    writing a corresponding XDMF file.
    This class implements more control for the user to for instance
    save multiple function to the same file, plotting of functions
    defined on higher order elements (e.g CG 2), and plotting of
    quadrature functions.

    Note that for plotting of functions defined on higer order elements
    the underlying mesh cannot be set as the underlying topology.


    Parameters
    ----------

    comm : petsc4py.PETSc.Comm or mpi4py.MPI.Comm
        MPI communicator
    name : str
        filename (with or without extesions)
    iterpolate : bool
        If true, interpolate functions so that they can be plotted
        used in the mesh as underlying topology. For function defined pointwise
        this means interpolating into linear Lagrange element, while for
        function defined at facets this means interpolating onto discontinous
        lagrange elements of order 0
    

    """
    
    def __init__(self, name, comm=None,
                 overwrite_file=True):

        if comm is None:
            comm = df.mpi_comm_world()
        self.name = os.path.splitext(name)[0]
        self.h5name = self.name + ".h5"
        self.xdmfname = self.name + ".xdmf"

        # Check file exists and act thereafter
        if os.path.isfile(self.h5name) and overwrite_file:
            if comm.rank == 0:
                os.remove(self.h5name)

        if os.path.isfile(self.xdmfname) and overwrite_file:
            if comm.rank == 0:
                os.remove(self.xdmfname)

        self.comm = comm
        self.cache = {}
        self.functions = {}
        self.mesh = {}

    def write(self, obj, name=None):

        name = obj.name() if name is None else name
        if name in self.cache:
            time = self.cache.pop(name)
            time += 1
        else:
            time = 0

        self.cache[name] = time

        if name not in self.functions:
            self.functions[name] = {}

        if obj.ufl_element().family() == "Lagrange":

            # We can visualize the function with on top of the mesh
            degree = obj.ufl_element().degree()
            mesh_str = "mesh_lagrange_1"  # {}".format(degree)

            m = obj.function_space().mesh()

            for i in range(degree-1):

                m = df.refine(m)
                V = df.FunctionSpace(m, "Lagrange", 1)
                obj = df.interpolate(obj, V)
        else:
            mesh_str = None

        self.functions[name][time] = dolfin_to_hd5(obj,
                                                   self.h5name,
                                                   time, self.comm,
                                                   name)

        # For moving mesh see own class XDMFFileMove
        if mesh_str and mesh_str not in self.mesh:
            self.mesh[mesh_str] = dolfin_to_hd5(m, self.h5name,
                                                "", self.comm, mesh_str)

    def finalize(self):
        """
        Write XDMFFile based on content in H5 file
        """
        families = np.array([v[0]["family"] for v in self.functions.values()])
        families[families == "Discontinuous Lagrange"] = "Lagrange"
        entries = {k: "" for k in np.unique(families)}

        # Check how many entries there are of each.
        # If the number do not correspond, repeat the final entry
        # until the numbers correspond
        times = {k: np.max(list(v.keys())) for k, v in self.functions.items()}
        N = np.max(list(times.values()))

        for i in range(N+1):

            entry = {k: {"att": "", "top": "", "geo": ""}
                     for k in np.unique(families)}

            for k, v in self.functions.items():

                if i in v:
                    i_ = i
                else:
                    i_ = times[k]

                d = v[i_]
                if d["family"] in ["Lagrange", 'Discontinuous Lagrange']:

                    degree = d["degree"] if d["family"] == "Lagrange"\
                             else d["degree"]+1
                    m = self.mesh["mesh_lagrange_{}".format(degree)]

                    top = xdmf.topology.format(ncells=d["ncells"],
                                               dim=d["top_dim"],
                                               h5group=d["topology"],
                                               cell=d["cell"],
                                               h5name=self.h5name)
                    
                    geo = xdmf.geometry.format(nverts=d["nverts"],
                                               dim=d["geo_dim"],
                                               coords=d["coords"],
                                               h5group=d["coordinates"],
                                               h5name=self.h5name)

                    family = "Lagrange"

                elif d["family"] == "Quadrature":

                    top = xdmf.topology_polyvert.format(nverts=d["nverts"])
                    geo = xdmf.geometry.format(nverts=d["nverts"],
                                               dim = d["geo_dim"],
                                               coords=d["coords"],
                                               h5group=d["coordinates"],
                                               h5name=self.h5name)

                    family = "Quadrature"

                entry[family]["top"] = top
                entry[family]["geo"] = geo

                if d["type"] == "scalar":

                    att_ = xdmf.scalar_attribute

                elif d["type"] == "vector":
                    att_ = xdmf.vector_attribute

                if d["family"] == 'Discontinuous Lagrange':
                    center = "Cell"
                else:
                    center = "Node"

                entry[family]["att"] += att_.format(name=k,
                                                    nverts=d["nverts"],
                                                    dim=d["dim"],
                                                    h5name=self.h5name,
                                                    center=center,
                                                    h5group=d["vector"])
       
            for k, v in entry.items():
                fname = v["top"]+v["geo"]+v["att"]
                entries[k] += xdmf.entry.format(frame=fname,
                                                iter=str(i))
        lst = " ".join(np.array(range(N+1), dtype=str))

        for k, entry in entries.items():
            body = xdmf.series.format(entry=entry, N=N+1,
                                      lst=lst, name=self.name)
            B = xdmf.body.format(body=body)
            with open("{}_{}.xdmf".format(self.name, k), "w") as f:
                f.write(B)


def _test_fun_to_xdmf():
    mesh = df.UnitSquareMesh(4, 4)
    elm = df.VectorElement(family="Quadrature",
                           cell=mesh.ufl_cell(),
                           degree=4,
                           quad_scheme="default")
    
    V = df.FunctionSpace(mesh, elm)
    fun = df.interpolate(df.Expression(("sin(x[0])",
                                        "cos(x[1])"),
                                       degree=2), V)

    fun_to_xdmf(fun, "test")


def _test_xdmf_2D():

    xd = MyXDMFFile("test.h5", df.mpi_comm_world(), interpolate=False)
    
    mesh = df.UnitSquareMesh(4, 4)
    elm = df.VectorElement(family="Quadrature",
                           cell=mesh.ufl_cell(),
                           degree=4,
                           quad_scheme="default")
    
    Q = df.FunctionSpace(mesh, elm)
    fun = df.interpolate(df.Expression(("sin(x[0])",
                                        "cos(x[1])"),
                                       degree=4), Q)

    elm = df.FiniteElement(family="CG",
                           cell=mesh.ufl_cell(), degree=1)
    V = df.FunctionSpace(mesh, elm)
    W = df.FunctionSpace(mesh, "DG", 0)

    f = df.Function(V, name="test_cg1")
    f2 = df.Function(V, name="test2_cg1")
    f_dg = df.interpolate(df.Expression("x[0]", degree=1), W)
    f.assign(df.Constant(1.0))
    f2.assign(df.Constant(0.1))

    xd.write(f)
    xd.write(f2)
    xd.write(f_dg)

    f.assign(df.Constant(2.0))
    f2.assign(df.Constant(0.2))
    f_dg.assign(df.Constant(0.4))

    xd.write(fun, "quad_vec")
    xd.write(f)
    xd.write(f2)
    xd.write(f_dg)

    # fun.assign()
    f = df.interpolate(df.Expression(("0.0", "1.0"), degree=2), Q)
    xd.write(f, "quad_vec2")
    xd.finalize()


if __name__ == "__main__":

    _test_xdmf_2D()
    # _test_fun_to_xdmf()
