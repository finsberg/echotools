import os
import numpy as np
import h5py

from . import surface
from . import strain_regions


class EchoFile(object):
    def __init__(self, path):

        self.path = path
        assert os.path.isfile(path), "File {} does not exist".format(path)

        self._load()

    def __repr__(self):
        return ("{self.__class__.__name__}" "({self.path})").format(self=self)

    def _load(self):
        # def load_echo_geometry(echo_path, time, scale = 100):
        """Load data from echo file
        """
        with h5py.File(self.path, "r") as echo_file:

            epi = echo_file["/LV_Mass_Epi"]
            endo = echo_file["/LV_Mass_Endo"]

            self.epi_faces = epi["indices"][:]
            # Take out epi vertices for the timeslot given
            self._epi_verts = epi["vertices"][:]

            self.endo_faces = endo["indices"][:]
            # Take out endo vertices for the timeslot given
            self._endo_verts = endo["vertices"][:]

            # Strain mesh containing information about AHA segment
            self._strain_mesh = echo_file["/LV_Strain/mesh"][:]

            self.time_stamps = echo_file["/time_stamps"][:]
            self.volume_trace = echo_file["/LV_Volume_Trace"][:]

            self._strain_trace = {}
            for k in echo_file["/LV_Strain_Trace"]:
                self._strain_trace[k] = echo_file["/LV_Strain_Trace"][k][:]

    @property
    def num_points(self):
        return len(self.time_stamps)

    def endo_verts(self, time, scale=100):
        """
        Get endocardial vertices

        Arguments
        ---------
        time : int
            Time point
        scale : scalar
            Scalar to multiply with coordinates to scale the
            geomtry. Default is 100 meaning that the scale
            is converted from m3 to cm3, since m3 is the
            default scale in echopac.
        """
        return scale * self._endo_verts[time, :, :]

    def epi_verts(self, time, scale=100):
        """
        Get epicardial vertices

        Arguments
        ---------
        time : int
            Time point
        scale : scalar
            Scalar to multiply with coordinates to scale the
            geomtry. Default is 100 meaning that the scale
            is converted from m3 to cm3, since m3 is the
            default scale in echopac.
        """
        return scale * self._epi_verts[time, :, :]

    def strain_mesh(self, time, scale=100):
        """
        Get strain mesh coordinates

        Arguments
        ---------
        time : int
            Time point
        scale : scalar
            Scalar to multiply with coordinates to scale the
            geomtry. Default is 100 meaning that the scale
            is converted from m3 to cm3, since m3 is the
            default scale in echopac.
        """
        return scale * self._strain_mesh[time, :, :]

    @property
    def strain_faces(self):

        if not hasattr(self, "_strain_faces"):

            self._strain_faces = np.array(
                [strain_regions.strain_faces(i) for i in range(1, 18)]
            )

        return self._strain_faces

    def strain_verts(self, time, scale=100):
        """
        Get the coordinates in the strain mesh

        Arguments
        ---------
        time : int
            Time point
        scale : scalar
            Scalar to multiply with coordinates to scale the
            geomtry. Default is 100 meaning that the scale
            is converted from m3 to cm3, since m3 is the
            default scale in echopac.
        """
        return surface.strain_mesh_coordinates(self.strain_mesh(time, scale))

    def trasformation_matrix(self, time):
        return surface.get_geometric_matrix(
            self.strain_mesh(time), np.copy(self.endo_verts(time))
        )

    def list_strain_types(self):
        return set(
            np.transpose([k.split("_") for k in list(self._strain_trace.keys())])[0]
        )

    def list_strain_regions(self, strain_type="RadialStrain"):
        return set(
            np.transpose(
                [
                    k.split("_")
                    for k in list(self._strain_trace.keys())
                    if k.split("_")[0] == strain_type
                ]
            )[1]
        )

    def strain_trace(self, strain_type, strain_region):
        """
        Get strain trace
        """
        key = "_".join([strain_type, strain_region])
        if key not in self._strain_trace:
            msg = (
                "Wrong strain type {0} or strain region {1}. "
                "\nPossible strain types {2}"
                "\nCheck list_strain_regions for possible "
                "strain regions"
            ).format(strain_type, strain_region, self.list_strain_types())
            raise KeyError(msg)
        return self._strain_trace[key]
