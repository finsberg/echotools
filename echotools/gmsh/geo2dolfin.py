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
from mpi4py import MPI as mpi
from fenics import *
from io import save_geometry, load_geometry
from gmsh2dolfin import gmsh2dolfin
from gmshfile import GmshFile
from inline_backend import gmsh_cpp_geo2msh

import os
import tempfile
import shutil

__all__ = ["geo2dolfin"]

def geo2dolfin(code, topological_dimension=3, geometric_dimension=None,
               comm=None, marker_ids=None):
    """
    FIXME
    """

    comm = comm if comm is not None else mpi_comm_world()
    ilead = comm.tompi4py().rank == 0

    if ilead:
        # create a temporary directory
        # tmpdir = tempfile.mkdtemp()

        # generates a new geo file
        # geoname = os.path.join(tmpdir, 'mesh.geo')
        geoname =  'mesh.geo'
        
        with open(geoname, 'w') as f:
            f.write(code)
        
        # generates the mesh
        info("--- Generating .msh file from .geo (may take a while)")
        # mshname = os.path.join(tmpdir, 'mesh.msh')
        # logname = os.path.join(tmpdir, 'mesh.log')
        mshname = 'mesh.msh'
        logname = 'mesh.log'
        curdir = os.getcwd()
        # os.chdir(tmpdir)
        gmsh_cpp_geo2msh(geoname, topological_dimension, mshname, logname)
        # os.chdir(curdir)

        
        # communicate the filename
        comm.tompi4py().bcast(mshname, root=0)
    else:
        # receive the filename
        mshname = comm.tompi4py().bcast(None, root=0)

    # import the mesh
    info("--- Importing from .msh")
    gmsh = GmshFile(mshname, geometric_dimension, get_log_level() == PROGRESS, marker_ids)
    mesh, markers = gmsh2dolfin(gmsh, mpi_comm_self(), use_coords=True)
    
    # save it to hdf5
    if ilead:
        # h5name = os.path.join(tmpdir, 'mesh.h5')
        h5name = 'mesh.h5'
        save_geometry(mpi_comm_self(), mesh, h5name, markers=markers)
        # communicate the filename
        comm.tompi4py().bcast(h5name, root = 0)
    else:
        h5name = comm.tompi4py().bcast(None, root = 0)


    geo = load_geometry(comm, h5name)
    
    for p in [geoname, logname, mshname, h5name, "LloydInit.pos"]:
        os.unlink(p)
        
    # import the mesh in parallel
    return geo

