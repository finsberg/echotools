# Copyright (C) 2016 Henrik Finsberg
#
# This file is part of MESH_GENERATION.
#
# MESH_GENERATION is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MESH_GENERATION is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MESH_GENERATION. If not, see <http://www.gnu.org/licenses/>.
from .mesh_utils import *
from .surface import WrongGeometryException
from scipy.optimize import minimize_scalar as minimize
import pickle


def estimate_round_off_buffer(
    patient,
    time,
    echo_path,
    ply_dir,
    mesh_type="lv",
    use_gamer=False,
    mesh_char_len=1.0,
):
    # Estimate round-off buffer, i.e cut size so the cavivty volume mathces the measured one

    tol = 0.05
    # Check if we allready have found the round off buffer

    if get_round_off_buffer(patient, time):

        round_off_buffer = get_round_off_buffer(patient, time)

        err = create_mesh_w_round_off_buffer(
            patient,
            time,
            echo_path,
            ply_dir,
            mesh_type,
            mesh_char_len,
            round_off_buffer,
            use_gamer,
            return_domain=False,
        )

        if err > tol:
            estimate_round_off_buffer = True
        else:
            estimate_round_off_buffer = False
            round_off_buffer = round_off_buffer
    else:

        estimate_round_off_buffer = True

    if estimate_round_off_buffer:
        logger.debug("{:<10}\t{:<10}".format("Round off", "Error"))

        def measured_vs_computed_volume(round_off_buffer):

            err = create_mesh_w_round_off_buffer(
                patient,
                time,
                echo_path,
                ply_dir,
                mesh_type,
                mesh_char_len,
                round_off_buffer,
                use_gamer=use_gamer,
                return_domain=False,
            )

            logger.debug("{:<10.2f}\t{:<10.2e}".format(round_off_buffer, err))
            # exit()
            return err

        res = minimize(
            measured_vs_computed_volume,
            method="bounded",
            bounds=(-1.0, 1.0),
            options={"xatol": 0.01},
        )

        round_off_buffer = res.x
        logger.info(
            "Optimization finished. Round of buffer = {}".format(round_off_buffer)
        )
        save_round_off_buffer(round_off_buffer, patient, time)

    return round_off_buffer


def create_mesh_w_round_off_buffer(
    patient,
    time,
    echo_path,
    ply_dir,
    mesh_type,
    mesh_char_len,
    round_off_buffer,
    use_gamer,
    return_domain=True,
):
    # Get markers for the facets
    fiber_markers = get_fiber_markers(mesh_type)

    # Get measured volume for echo
    v_meas = get_measured_volume(echo_path, time)

    # Create ply file from echo surfaces
    try:
        data = create_lv_ply_files(
            echo_path, time, round_off_buffer, True, ply_dir, use_gamer=use_gamer
        )

        # Mesh ply files using Gmsh
        mesh, markers = create_lv_geometry(
            time, data["plydir"], mesh_char_len=mesh_char_len, marker_ids=fiber_markers
        )

        # plot(mesh, interactive=True)
        # Compute cavity volume
        v_mesh = compute_cavity_volume(mesh, 30)
        err = (v_mesh - v_meas) ** 2

        if return_domain:
            return mesh, markers, err, data
        else:
            return err
    except WrongGeometryException as ex:
        logger.debug(ex)
        return 10000


def create_mesh(
    patient,
    time,
    echo_path,
    ply_dir=".",
    mesh_type="lv",
    mesh_char_len=1.0,
    use_gamer=False,
    **kwargs
):

    # Estimate were we should cut the
    round_off_buffer = estimate_round_off_buffer(
        patient, time, echo_path, ply_dir, mesh_type, use_gamer, mesh_char_len
    )

    # round_off_buffer = -0.2
    mesh, markers, err, data = create_mesh_w_round_off_buffer(
        patient,
        time,
        echo_path,
        ply_dir,
        mesh_type,
        mesh_char_len,
        round_off_buffer,
        use_gamer,
        True,
    )

    logger.info("Volume difference is {}".format(err))
    return mesh, markers, data


class MeshConstructor(object):
    def __init__(self, parameters):
        """
        Create mesh object
        
        """
        self.parameters = parameters

        # for key, val in parameters.iteritems():
        #     setattr(self, key, val)

    def mesh(self):

        mesh = self.generate_mesh()

        if not self.parameters["strain_type"] == "0":
            self.generate_strain_markers(mesh)

        if self.parameters["refine_mesh"]:
            # Refine mesh in some regions where the wall is thin
            self._mesh = refine_at_strain_regions(mesh, [2])
        else:
            self._mesh = mesh

        self.generate_fibers()
        self.generate_local_basis_functions()

        self.save()

    def get_mesh(self):
        return self._mesh

    def move_mesh(self, dst):
        import shutil

        if not os.path.exists(dst):
            os.makedirs(dst)

        shutil.copy(self.parameters["h5name"], dst)

    def generate_fibers(self):

        if self.parameters["generate_fibers"]:
            # Generate fibers using fiberrules
            self.fields = generate_fibers(self._mesh, **self.parameters["Fibers"])
        else:
            self.fields = None

    def generate_strain_markers(self, mesh):

        # Generate markers for the 17-segement AHA-zones
        generate_strain_markers(
            mesh,
            self.surface_data["focal_point"],
            self.surface_data["strain_coords"],
            self.parameters["strain_type"],
        )

        # mark_segment_centers(mesh)

    def generate_local_basis_functions(self):

        if self.parameters["generate_local_basis"]:
            # Generate local basis function in the circumferential
            # radial and longitidunal direction
            self.local_basis = generate_local_basis_functions(
                self._mesh, self.surface_data["focal_point"]
            )
        else:
            self.local_basis = None

    def generate_mesh(self):

        mesh, markers, data = create_mesh(**self.parameters)

        self.surface_data = data

        return mesh

    def save(self, overwrite_file=True, overwrite_group=True):

        markers = get_markers(self.parameters["mesh_type"])

        save_geometry_to_h5(
            self._mesh,
            self.parameters["h5name"],
            "",
            markers,
            self.fields,
            self.local_basis,
            overwrite_file=overwrite_file,
            overwrite_group=overwrite_group,
        )

        ply_folder = os.path.join(
            self.parameters["ply_dir"],
            "_".join([self.parameters["patient"], self.parameters["mesh_type"]]),
        )
        # save_transformation_matrix(ply_folder, str(self.time), self.h5name)


def setup_mesh_parameters(echopath=".", h5name="test.h5", mesh_char_len=0.65, time=0):

    charlen_to_res = {0.65: "low_res", 0.45: "med_res", 0.3: "high_res"}
    if mesh_char_len in list(charlen_to_res.keys()):
        resolution = charlen_to_res[mesh_char_len]
    else:
        resolution = str(mesh_char_len)

    mesh_params = dolfin.Parameters("Mesh")

    # Paths
    name = os.path.basename(echopath).split(".")[0]
    mesh_params.add("patient", name)
    mesh_params.add("mesh_type", "lv", ["lv", "biv"])
    mesh_params.add("time", time)

    mesh_params.add("echo_path", echopath)
    # mesh_params.add("outdir", outdir)
    outdir = os.path.dirname(h5name)
    mesh_params.add("ply_dir", os.path.join(outdir, "ply_files", name))
    mesh_params.add("h5name", h5name)
    mesh_params.add("resolution", resolution)
    mesh_params.add("mesh_char_len", mesh_char_len)

    mesh_params.add("refine_mesh", False)
    mesh_params.add("use_gamer", False)

    mesh_params.add("strain_type", "17", ["0", "16", "17", "18"])
    mesh_params.add("generate_local_basis", True)

    mesh_params.add("generate_fibers", True)
    fiber_params = setup_fiber_parameters()
    mesh_params.add(fiber_params)

    return mesh_params


def setup_fiber_parameters(epi=-60, endo=60, include_sheets=False):

    fiber_params = dolfin.Parameters("Fibers")
    fiber_params.add("fiber_space", "Quadrature_4")
    # fiber_params.add("fiber_space", "CG_1")
    fiber_params.add("include_sheets", include_sheets)

    # Parameter set from Bayer et al.
    # fiber_params.add("fiber_angle_epi", 50)
    # fiber_params.add("fiber_angle_endo", 40)
    # fiber_params.add("sheet_angle_epi", 25)
    # fiber_params.add("sheet_angle_endo", 65)
    fiber_params.add("fiber_angle_epi", epi)
    fiber_params.add("fiber_angle_endo", endo)
    fiber_params.add("sheet_angle_epi", 0)
    fiber_params.add("sheet_angle_endo", 0)

    return fiber_params
