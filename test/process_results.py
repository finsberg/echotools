#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import dolfin as df
import h5py

import echotools


def get_args():

    descr = "Process results"
    usage = "Show uage here"
    parser = argparse.ArgumentParser(description=descr,
                                     usage=usage,
                                     add_help=True)

    parser.add_argument(action='store',
                        dest='fname',
                        type=str,
                        default='result.h5',
                        help='File with results')
    parser.add_argument(action='store',
                        dest='echo_file',
                        type=str,
                        default='test_data.h5',
                        help='File with echo data')
    return parser


def load(fname):

    if T is None:
        T = np.diag(np.ones(4))

    groups = []
    with h5py.File(fname, 'r') as h5file:
        for path in ['/passive_inflation/displacement/{}',
                     '/active_contraction/contract_point_{}/displacement/0']:
            i = 0
            while path.format(i) in h5file:
                groups.append(path.format(i))
                i += 1

    comm = df.mpi_comm_world()
    us = []
    Us = []

    xdmf = echotools.save.MyXDMFFile('displacement.h5')

    with df.HDF5File(comm, fname, "r") as h5file:

        mesh = df.Mesh(comm)

        h5file.read(mesh, '/unloaded/geometry/mesh', False)

        V = df.VectorFunctionSpace(mesh, 'CG', 2)
        u = df.Function(V)

        W = df.VectorFunctionSpace(mesh, 'CG', 1)
        U = df.Function(W, name='displacement')

        # Do not use the unloaded geometry as the first frame
        h5file.read(u, groups[1])
        U0 = df.interpolate(u, W)
        df.ALE.move(mesh, U0)

        print('Load data')
        for p in groups[1:]:
            
            print(p)
            h5file.read(u, p)
            us.append(u.vector().get_local())

            U_ = df.interpolate(u, W)
            U.vector()[:] = U_.vector() - U0.vector()
            # if i > 0:
            xdmf.write(U)

            Us.append(U.vector().get_local())
            p = path.format(i)
            i += 1

        xdmf.finalize()
    return mesh, Us


def check_args(args):

    for arg in ['fname', 'echo_file']:
        msg = "The file {} does not exist".format(args[arg])
        if not os.path.isfile(args['fname']):
            raise IOError(msg)


def main(args):

    echo = echotools.EchoFile(args['echo_file'])
    mesh, us = load(args['fname'])


if __name__ == "__main__":

    parser = get_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = vars(parser.parse_args())

    check_args(args)
    main(args)

