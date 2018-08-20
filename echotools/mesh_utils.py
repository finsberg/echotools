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
# from dolfin import *
import dolfin

import numpy as np
import strain_regions as sr

import os, yaml

try:
    import h5py
    has_h5py = True
except:
    print "Warning: h5py is not installed"
    has_h5py = False

parallel_h5py = h5py.h5.get_config().mpi

try:
    import mpi4py, petsc4py
    has_mpi4py = True
except:
    has_mpi4py = False
    if parallel_h5py: raise ImportError
else:
    from mpi4py import MPI as mpi4py_MPI

ROUND_OFF_FILE =  "round_off.yml"


# Mesh should be in cm
SCALE = 100


ESTIMATE_FOCAL_POINT = False
DEFAULT_FOCAL_POINT = 6.0

# Logger
import logging
log_level = logging.INFO
def make_logger(name, level = logging.INFO):

    mpi_filt = lambda: None
    def log_if_proc0(record):
        if dolfin.MPI.rank(dolfin.mpi_comm_world()) == 0:
            return 1
        else:
            return 0
        
    mpi_filt.filter = log_if_proc0

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(0)


    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    

    logger.addHandler(ch)
    logger.addFilter(mpi_filt)

    
    dolfin.set_log_active(False)
    dolfin.set_log_level(dolfin.WARNING)
    
    return logger

logger = make_logger("Mesh generation", log_level)

### Get/Load stuff ####
def load_echo_geometry(echo_path, time, scale = 100):
    """Get geometric data from echo data.
    Get verices and faces of the endo- and epicardium,
    as well as a strain mesh defining the location of the
    AHA segements

    :param str echo_path: Path to the echo file
    :param int time: Which time stamp
    :param float scale: Scale of the geometry.
    The original surfaces is in m3, so cm = 100, mm = 1000             
    :returns: A dictionary with the data
    :rtype: dict

    """
    
    import surface as surf
    assert os.path.isfile(echo_path), "File {} does not exist".format(echo_path)

    data = {}
    with h5py.File(echo_path, "r") as echo_file:
        
        epi = echo_file['/LV_Mass_Epi']
        endo = echo_file['/LV_Mass_Endo']

        data["epi_faces"] = np.array(epi['indices'])
        # Take out epi vertices for the timeslot given
        # and convert from m3 to cm3
        data["epi_verts"] = scale*np.array(epi['vertices'])[time,:,:]
            
        data["endo_faces"] = np.array(endo['indices'])
        # Take out endo vertices for the timeslot given
        # and convert from m3 to cm3
        data["endo_verts"] = scale*np.array(endo['vertices'])[time,:,:]

        # Strain mesh containing information about AHA segment
        # Convert from m3 to cm3
        data["strain_mesh"] = scale*np.array(echo_file['/LV_Strain/mesh'])[time,:,:]


    data["strain_coords"] = surf.get_strain_region_coordinates(data["strain_mesh"])

    return data   

def create_lv_ply_files(echo_path, time, round_off = 0.0,
                        esitmate_focal_point = True, plydir = None,
                        use_gamer = False, cut_base = True):
    """Create ply files to be used for lv mesh generation

    :param patient: Name of patient
    :param time: the timestamp (index) from echo
    :param round_off: Value determining the location of the basal plane
    :param esitmate_focal_point: (bool) Estimate focal point or use default (6.0)
    :param plydir: directory for the plyfiles. If None a temporary directory is created
    :param bool use_gamer: Use gamer to smooth surfaces before meshing
    :returns: the surfaces ready for meshing
    :rtype: dict

    """
    import surface as surf

    original_data = load_echo_geometry(echo_path, time)
    if plydir is not None:
        surf.save_surfaces_to_ply(plydir, time,  original_data, "_raw")
        
        
    # Original long axis diameter
    from scipy.spatial import distance
    long_axis_endo = np.max(distance.cdist(original_data["endo_verts"],
                                           original_data["endo_verts"],
                                           "euclidean"))

    # Transform the data
    transformed_data = surf.transform_surfaces(round_off, **original_data)


    # Cut of the base
    if cut_base:
        data = surf.remove_base(**transformed_data)
    else:
        data = transformed_data

    if use_gamer:
        data = surf.smooth_surface_gamer(data)
    
    if esitmate_focal_point:
        foc = surf.compute_focal_point(long_axis_endo, **data)
    else:
        foc = DEFAULT_FOCAL_POINT

    data["focal_point"] = foc

    if plydir is None:
        import tempfile
        plydir = tempfile.mkdtemp()
        
    
    surf.save_surfaces_to_ply(plydir, time,  data)
    data["plydir"] = plydir
    return data
    
def get_time_stamps(h5path):
    """
    Get the time stamps for the
    echo data.

    :param str h5path: Path to the file containg echo data
    :returns: list of time stamps
    :rtype: list

    """
    assert os.path.isfile(h5path), "Invalid name {}".format(h5path)
    echo_file = h5py.File(h5path, "r")
    time_stamps = np.array(echo_file['time_stamps'])
    echo_file.close()
    return time_stamps

def get_markers(mesh_type = "lv"):
    
    assert mesh_type in ["lv", "biv"]

    fiber_markers = get_fiber_markers(mesh_type)

    markers = {}
    markers["NONE"] = (0, 3) 

    markers["BASE"] = (fiber_markers["BASE"], 2) 
    markers["EPI"] =  (fiber_markers["EPI"], 2)
    markers["EPIRING"] =  (fiber_markers["EPIRING"], 1)

    if mesh_type == "lv":
    
        markers["ENDO"] = (fiber_markers["ENDO"], 2)
        
        markers["ENDORING"] =  (fiber_markers["ENDORING"], 1)

    else:

        markers["ENDO_RV"] = (fiber_markers["ENDO_RV"], 2)
        markers["ENDO_LV"] = (fiber_markers["ENDO_LV"], 2)
        
        markers["ENDORING_RV"] =  (fiber_markers["ENDORING_RV"], 1)
        markers["ENDORING_LV"] =  (fiber_markers["ENDORING_LV"], 1)

    return markers

    
def get_round_off_buffer(patient, time):
    if os.path.isfile(ROUND_OFF_FILE):
        with open(ROUND_OFF_FILE, 'r') as outfile:
            dic = yaml.load(outfile)

        try:
            return float(dic[patient][str(time)])
        except:
            return False
    else:
        return False


def get_measured_volume(echo_path, time):

    h5file = h5py.File(echo_path, "r")
    volumes = np.array(h5file["LV_Volume_Trace"])*1000*1000
    vol  = volumes[time]
    h5file.close()

    return vol

### Create/generate stuff ###
def generate_local_basis_functions(mesh, focal_point):
    """Generate vector field for the circumferential, 
    radial and longitudinal direction.

    :param mesh: The mesh
    :type mesh: (:py:class:`dolfin.Mesh`)
    :param float focal_point: Focel point
    :returns: List of functions 
    [circumeferential, Radial, Longitudinal]
    :rtype: list

    """

    # Make basis functions
    c, r, l = sr.make_crl_basis(mesh, focal_point) 

    return [c,r,l]
    

def generate_strain_markers(mesh, focal_point,
                            strain_regions, strain_type):
    """Generate markers for the AHA segements, 
    and mark the mesh accordingly

    :param mesh: The mesh
    :type mesh: (:py:class:`dolfin.Mesh`)
    :param float focal_point: Focal point
    :param strain_regions: 
    :param str strain_type: 16 or 17 segements


    """
    

    
    # Strain Markers
    sfun = dolfin.MeshFunction("size_t", mesh, 3)
    sfun = sr.mark_lv_strain_regions(sfun,
                                     mesh,
                                     focal_point,
                                     strain_regions,
                                     strain_type)


    
    # Mark the cells accordingly
    for cell in dolfin.cells(mesh):
        mesh.domains().set_marker((cell.index(), sfun[cell]), 3)



def create_geometry_with_strain(meshdir, case, stage, mesh_scaling=1.0, markings = None) :
    """
    Create geometry by meshing the three surfaces (endo lv, endo rv and epi)
    together using a wrapped version for gmsh.
    This also mesh the strain patches inside the mesh.

    NOTE: The strain mesh will in some cases intersect the endo lv surface.
    In this case we are not able to mesh the patches inside the mesh.
    """
    from textwrap import dedent
    geocode = dedent(\
    """\
    // meshing options
    Mesh.CharacteristicLengthFromCurvature = 1;
    Mesh.Lloyd = 1;
    Mesh.CharacteristicLengthMin = 0.8;
    Mesh.CharacteristicLengthMax = 0.8;
    Mesh.Optimize = 1;
    Mesh.OptimizeNetgen = 1;
    Mesh.RemeshParametrization = 7;
    Mesh.SurfaceFaces = 1;
    Mesh.CharacteristicLengthFactor = {mesh_scaling};

    // load the surfaces
    Merge "{meshdir}/endo_lv_us_{stage}.ply";
    Merge "{meshdir}/endo_rv_us_{stage}.ply";
    Merge "{meshdir}/epi_us_{stage}.ply";

    // load strain pathes
    Merge "{meshdir}/strain_region1_{stage}.ply";
    Merge "{meshdir}/strain_region2_{stage}.ply";
    Merge "{meshdir}/strain_region3_{stage}.ply";
    Merge "{meshdir}/strain_region4_{stage}.ply";
    Merge "{meshdir}/strain_region5_{stage}.ply";
    Merge "{meshdir}/strain_region6_{stage}.ply";
    Merge "{meshdir}/strain_region7_{stage}.ply";
    Merge "{meshdir}/strain_region8_{stage}.ply";
    Merge "{meshdir}/strain_region9_{stage}.ply";
    Merge "{meshdir}/strain_region10_{stage}.ply";
    Merge "{meshdir}/strain_region11_{stage}.ply";
    Merge "{meshdir}/strain_region12_{stage}.ply";
    Merge "{meshdir}/strain_region13_{stage}.ply";
    Merge "{meshdir}/strain_region14_{stage}.ply";
    Merge "{meshdir}/strain_region15_{stage}.ply";
    Merge "{meshdir}/strain_region16_{stage}.ply";
    Merge "{meshdir}/strain_region17_{stage}.ply";

    CreateTopology;

    ll[] = Line "*";
    L_LV_base = newl; Compound Line(L_LV_base) = ll[2];
    L_RV_base = newl; Compound Line(L_RV_base) = ll[0];
    L_epi_base = newl; Compound Line(L_epi_base) = ll[1];
    Physical Line("ENDORING_LV") = {{ L_LV_base }};
    Physical Line("ENDORING_RV") = {{ L_RV_base }};
    Physical Line("EPIRING") = {{ L_epi_base }};

    L_R1_base = newl; Compound Line(L_R1_base) = ll[16];
    L_R2_base = newl; Compound Line(L_R2_base) = ll[12];
    L_R3_base = newl; Compound Line(L_R3_base) = ll[20];
    L_R4_base = newl; Compound Line(L_R4_base) = ll[24];
    L_R5_base = newl; Compound Line(L_R5_base) = ll[8];
    L_R6_base = newl; Compound Line(L_R6_base) = ll[7];

    L_strain = newl; 
    Compound Line(L_strain) = {{ L_R1_base, L_R2_base, L_R3_base, L_R4_base, L_R5_base, L_R6_base }};


    ss[] = Surface "*";
    S_LV = news; Compound Surface(S_LV) = ss[0];
    S_RV = news; Compound Surface(S_RV) = ss[1];
    S_epi = news; Compound Surface(S_epi) = ss[2];

    S_1 = news; Compound Surface(S_1) = ss[3];		
    S_2 = news; Compound Surface(S_2) = ss[4];
    S_3 = news; Compound Surface(S_3) = ss[5];
    S_4 = news; Compound Surface(S_4) = ss[6];
    S_5 = news; Compound Surface(S_5) = ss[7];
    S_6 = news; Compound Surface(S_6) = ss[8];
    S_7 = news; Compound Surface(S_7) = ss[9];
    S_8 = news; Compound Surface(S_8) = ss[10];
    S_9 = news; Compound Surface(S_9) = ss[11];
    S_10 = news; Compound Surface(S_10) = ss[12];
    S_11 = news; Compound Surface(S_11) = ss[13];
    S_12 = news; Compound Surface(S_12) = ss[14];
    S_13 = news; Compound Surface(S_13) = ss[15];
    S_14 = news; Compound Surface(S_14) = ss[16];
    S_15 = news; Compound Surface(S_15) = ss[17];
    S_16 = news; Compound Surface(S_16) = ss[18];
    S_17 = news; Compound Surface(S_17) = ss[19];

    S_strain = news; Compound Surface(S_strain) = {{ S_1, S_2, S_3, S_4, S_5, S_6, S_7, S_8, S_9, S_10, S_11, S_12, S_13, S_14, S_15, S_16, S_17 }};

    Physical Surface("ENDO_LV") = {{ S_LV }};
    Physical Surface("ENDO_RV") = {{ S_RV }};
    Physical Surface("EPI") = {{ S_epi }};
    Physical Surface("REGION_1") = {{ S_1 }};
    Physical Surface("REGION_2") = {{ S_2 }};
    Physical Surface("REGION_3") = {{ S_3 }};
    Physical Surface("REGION_4") = {{ S_4 }};
    Physical Surface("REGION_5") = {{ S_5 }};
    Physical Surface("REGION_6") = {{ S_6 }};
    Physical Surface("REGION_7") = {{ S_7 }};
    Physical Surface("REGION_8") = {{ S_8 }};
    Physical Surface("REGION_9") = {{ S_9 }};
    Physical Surface("REGION_10") = {{ S_10 }};
    Physical Surface("REGION_11") = {{ S_11 }};
    Physical Surface("REGION_12") = {{ S_12 }};
    Physical Surface("REGION_13") = {{ S_13 }};
    Physical Surface("REGION_14") = {{ S_14 }};
    Physical Surface("REGION_15") = {{ S_15 }};
    Physical Surface("REGION_16") = {{ S_16 }};
    Physical Surface("REGION_17") = {{ S_17 }};


    LL_base_1 = newll; 
    Line Loop(LL_base_1) = {{ L_LV_base,L_R1_base, -L_R2_base, -L_R3_base, -L_R4_base, -L_R5_base, L_RV_base, L_epi_base, L_R6_base }};
    S_base_1 = news; Plane Surface(S_base_1) = {{ LL_base_1 }};

    LL_base_2 = newll;
    Line Loop(LL_base_2) = {{ L_R1_base, -L_R2_base, -L_R3_base, -L_R4_base, -L_R5_base,L_RV_base, -L_R6_base }};
    S_base_2 = news; Plane Surface(S_base_2) = {{ LL_base_2 }};

    Physical Surface("BASE") = {{ S_base_1, S_base_2 }};

    SL_wall1 = newsl; 
    Surface Loop(SL_wall1) = {{ S_LV, S_strain, S_base_2 }};

    SL_wall2 = newsl; 
    Surface Loop(SL_wall2) = {{ S_strain, S_RV, S_epi, S_base_1 }};


    V_wall1 = newv; Volume(V_wall1) = {{ SL_wall1 }};
    V_wall2 = newv; Volume(V_wall2) = {{ SL_wall2 }};
    Physical Volume("WALL") = {{ V_wall1, V_wall2 }};

    """.format(stage=stage, case=case, meshdir=meshdir,
                mesh_scaling=mesh_scaling))

    from gmsh import geo2dolfin
    mesh, markers = geo2dolfin(geocode, marker_ids = markings)
    


    return mesh, markers    

def create_lv_geometry(time, ply_dir, mesh_char_len=1.0, marker_ids=None) :
    """
    Create geometry by meshing the three surfaces (endo lv, endo rv and epi)
    together using a wrapped version for gmsh
    """
    
    # Make sure the ply files excists
    for fname in ["endo_lv_{}.ply".format(time), "epi_lv_{}.ply".format(time)]:
        if not os.path.isfile("{}/{}".format(ply_dir, fname)):
            raise IOError("'{}' is not a valid .ply file.".format(\
                "{}/{}".format(ply_dir, fname)))

    from textwrap import dedent
    geocode = dedent(\
    """\
    // meshing options
    Mesh.CharacteristicLengthFromCurvature = 1;
    Mesh.Lloyd = 1;
    Geometry.HideCompounds = 0;
    Mesh.CharacteristicLengthMin = {mesh_char_len};
    Mesh.CharacteristicLengthMax = {mesh_char_len};
    Mesh.ScalingFactor = {mesh_scaling};
    Mesh.Optimize = 1;
    Mesh.OptimizeNetgen = 1;
    Mesh.RemeshParametrization = 7;  // (0=harmonic_circle, 1=conformal_spectral, 2=rbf, 3=harmonic_plane, 4=convex_circle, 5=convex_plane, 6=harmonic square, 7=conformal_fe) (Default=4)
    Mesh.SurfaceFaces = 1;
    Mesh.Algorithm    = 6; // (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal, 7=bamg, 8=delquad) (Default=2)
    Mesh.Algorithm3D    = 4; // (1=Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree) (Default=1)
    Mesh.Recombine3DAll = 0;

    // load the surfaces
    Merge "{ply_dir}/endo_lv_{time}.ply";
    Merge "{ply_dir}/epi_lv_{time}.ply";

    CreateTopology;

    ll[] = Line "*";
    L_LV_base = newl; Compound Line(L_LV_base) = ll[1];
    L_epi_base = newl; Compound Line(L_epi_base) = ll[0];
    Physical Line("ENDORING") = {{ L_LV_base }};
    Physical Line("EPIRING") = {{ L_epi_base }};

    ss[] = Surface "*";
    S_LV = news; Compound Surface(S_LV) = ss[0];
    S_epi = news; Compound Surface(S_epi) = ss[1];
    Physical Surface("ENDO") = {{ S_LV }};
    Physical Surface("EPI") = {{ S_epi }};

    LL_base = newll; 
    Line Loop(LL_base) = {{ L_LV_base, L_epi_base }};
    S_base = news; Plane Surface(S_base) = {{ LL_base }};
    Physical Surface("BASE") = {{ S_base }};

    SL_wall = newsl; 
    Surface Loop(SL_wall) = {{ S_LV, S_epi, S_base }};
    V_wall = newv; Volume(V_wall) = {{ SL_wall }};
    Physical Volume("WALL") = {{ V_wall }};
    Coherence;
    """).format(time=time, ply_dir=ply_dir,
                mesh_scaling=10.0, mesh_char_len=mesh_char_len)
    # mesh_scaling = 1.0 -> cm
    # mesh_scaling = 10 -> mm
    from gmsh import geo2dolfin
    mesh, markers = geo2dolfin(geocode, marker_ids=marker_ids)
    

    return mesh, markers

def create_biv_geometry(time, ply_dir, mesh_char_len=1.0, marker_ids=None) :
    """
    Create geometry by meshing the three surfaces (endo lv, endo rv and epi)
    together using a wrapped version for gmsh
    """

    # Make sure the ply files excists
    for fname in ["endo_lv_us_{}.ply".format(time), "endo_rv_us_{}.ply".format(time), "epi_us_{}.ply".format(time)]:
        if not os.path.isfile("{}/{}".format(ply_dir, fname)):
            raise IOError("'{}' is not a valid .ply file.".format(\
                "{}/{}".format(ply_dir, fname)))

    from textwrap import dedent
    geocode = dedent(\
    """\
    // meshing options
    Mesh.CharacteristicLengthFromCurvature = 1;
    Mesh.Lloyd = 1;
    Mesh.CharacteristicLengthMin = {mesh_char_len};
    Mesh.CharacteristicLengthMax = {mesh_char_len};
    Mesh.ScalingFactor = {mesh_scaling};
    Mesh.Optimize = 1;
    Mesh.OptimizeNetgen = 1;
    Mesh.RemeshParametrization = 7;
    Mesh.SurfaceFaces = 1;


    // load the surfaces
    Merge "{ply_dir}/endo_lv_us_{time}.ply";
    Merge "{ply_dir}/endo_rv_us_{time}.ply";
    Merge "{ply_dir}/epi_us_{time}.ply";

    CreateTopology;

    ll[] = Line "*";
    L_LV_base = newl; Compound Line(L_LV_base) = ll[2];
    L_RV_base = newl; Compound Line(L_RV_base) = ll[0];
    L_epi_base = newl; Compound Line(L_epi_base) = ll[1];
    Physical Line("ENDORING_LV") = {{ L_LV_base }};
    Physical Line("ENDORING_RV") = {{ L_RV_base }};
    Physical Line("EPIRING") = {{ L_epi_base }};

    ss[] = Surface "*";
    S_LV = news; Compound Surface(S_LV) = ss[0];
    S_RV = news; Compound Surface(S_RV) = ss[1];
    S_epi = news; Compound Surface(S_epi) = ss[2];
    Physical Surface("ENDO_LV") = {{ S_LV }};
    Physical Surface("ENDO_RV") = {{ S_RV }};
    Physical Surface("EPI") = {{ S_epi }};

    LL_base = newll; 
    Line Loop(LL_base) = {{ L_LV_base, L_RV_base, L_epi_base }};
    S_base = news; Plane Surface(S_base) = {{ LL_base }};
    Physical Surface("BASE") = {{ S_base }};

    SL_wall = newsl; 
    Surface Loop(SL_wall) = {{ S_LV, S_RV, S_epi, S_base }};
    V_wall = newv; Volume(V_wall) = {{ SL_wall }};
    Physical Volume("WALL") = {{ V_wall }};

    """).format(time=time, ply_dir=ply_dir,
                mesh_scaling=10.0, mesh_char_len=mesh_char_len)
    # mesh_scaling = 1.0 -> cm
    # mesh_scaling = 10 -> mm
   
    
    from gmsh import geo2dolfin
    mesh, markers = geo2dolfin(geocode, marker_ids=marker_ids)
    

    return mesh, markers



        

### Compute/Do stuff #####
                
def refine_mesh(mesh, cell_markers=None):

    
    logger.info("\nRefine mesh")
    dolfin.parameters["refinement_algorithm"] = "plaza_with_parent_facets"

    # Refine mesh
    if cell_markers is None:
        new_mesh = dolfin.adapt(mesh)
    else:
        new_mesh = dolfin.adapt(mesh, cell_markers)

    # Refine ridges function
    # Dont work!!
    # rfun = MeshFunction("size_t", mesh, 1, mesh.domains())
    # new_rfun = adapt(rfun, new_mesh)
    
    # Refine facetfunction
    ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())
    new_ffun = dolfin.adapt(ffun, new_mesh)

    # Refine cell function
    cfun = dolfin.MeshFunction("size_t", mesh, 3, mesh.domains())
    new_cfun = dolfin.adapt(cfun, new_mesh)

    # Mark the cells and facets for the new mesh
    for cell in dolfin.cells(new_mesh):
        new_mesh.domains().set_marker((cell.index(), new_cfun[cell]), 3)

        for f in dolfin.facets(cell):
            new_mesh.domains().set_marker((f.index(), new_ffun[f]), 2)

            # for e in edges(cell):
                # new_mesh.domains().set_marker((e.index(), new_rfun[e]), 1)

    return new_mesh


def refine_at_strain_regions(mesh, regions):
    strain_markers = dolfin.MeshFunction("size_t", mesh, 3, mesh.domains())

    cell_markers = dolfin.MeshFunction("bool", mesh, 3)

    if regions == "all":
        cell_markers.set_all(True)
    else:
    
        cell_markers.set_all(False)

        for c in dolfin.cells(mesh):
            if strain_markers[c] in regions:
                cell_markers[c] = True

    
    new_mesh = refine_mesh(mesh, cell_markers)
    return new_mesh


def compute_cavity_volume(mesh, endo):

    X = dolfin.SpatialCoordinate(mesh)
    N = dolfin.FacetNormal(mesh)
    ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())
    ds = dolfin.Measure("exterior_facet", subdomain_data = ffun, domain = mesh)(endo)
    vol_form = (-1.0/3.0)*dolfin.dot(X, N)*ds
    return dolfin.assemble(vol_form)

    
def save_transformation_matrix(ply_folder, reftime, h5name):

    output_params_name = os.path.join(ply_folder, "params_{}.p".format(reftime))
    import pickle
    params = pickle.load(open(output_params_name, 'rb'))
    T = params["T_mat"]
    
        
    with h5py.File(h5name, "a") as h5file:
        h5file["{}/transformation_matrix".format(reftime)] = T

            
def save_round_off_buffer(round_off_buffer, patient, time):

    from collections import defaultdict
    # If the file allready exist, load it
    if os.path.isfile(ROUND_OFF_FILE):
        with open(ROUND_OFF_FILE, 'r') as outfile:
            dic = yaml.load(outfile)
            d = defaultdict(dict,dic)
    else:
        # Otherwise create a new one
        d = defaultdict(dict)

    d[patient][str(time)] = str(round_off_buffer)

    # Save the file
    with open(ROUND_OFF_FILE, 'w') as outfile:
        yaml.dump(dict(d), outfile, default_flow_style=False)

