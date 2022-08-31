import numpy as np
import vtk
from textwrap import dedent
from tvtk.array_handler import array2vtk, array2vtkPoints, array2vtkCellArray


vtk_shapes = {
    1: {1: vtk.VTK_LINE, 2: vtk.VTK_TRIANGLE, 3: vtk.VTK_TETRA},
    2: {
        1: vtk.VTK_QUADRATIC_EDGE,
        2: vtk.VTK_QUADRATIC_TRIANGLE,
        3: vtk.VTK_QUADRATIC_TETRA,
    },
}


def to_grid(verts, faces):
    """
    Convert vertices and faces to vtkUnstructuredGrid

    Arguments
    ---------
    verts : array
        List of vertices
    faces : array
        List of faces / connectivity
    mdim : int 
        Topological dimension (Surface = 2, Volume = 3).
    order : int 
        Quadratic (2) or linear (1) elements
    """
    vtk_shape = vtk_shapes[1][np.shape(faces)[1]]

    # create the grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(array2vtkPoints(verts))
    grid.SetCells(vtk_shape, array2vtkCellArray(faces))
    return grid


def write_vtu(filename, grid):
    """
    Write vtk unstructured grid file (vtu)

    Arguments
    ---------
    fnane : str
        Filename
    grid : vtk.vtkUnstructuredGrid
        The data
    """

    writer = vtk.vtkXMLUnstructuredGridWriter()

    if vtk.VTK_VERSION >= "6.0":
        writer.SetInputData(grid)
    else:
        writer.SetInput(grid)

    writer.SetFileName(filename)
    writer.Write()


def to_polydata(verts, faces):

    grid = vtk.vtkPolyData()
    grid.SetPoints(array2vtkPoints(verts))
    grid.SetPolys(array2vtkCellArray(faces))
    return grid


def write_ply(fname, grid):
    """
    Write plyfile

    Arguments
    ---------
    fnane : str
        Filename
    grid : vtk.vtkPolyData
        The data
    """

    writer = vtk.vtkPLYWriter()
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.SetFileTypeToASCII()
    writer.Write()


def write_vtk(fname, grid):
    """
    Write vtk polydata file (vtp)

    Arguments
    ---------
    fnane : str
        Filename
    grid : vtk.vtkPolyData
        The data
    """

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.SetFileTypeToASCII()
    writer.Write()
