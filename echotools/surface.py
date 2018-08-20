import os
import numpy as np


def map_plane_to_xy_plane(c):
    """
    Given a normal vector we can rotate the
    plane so that it aligns with the xy plane

    A plane can be written in a general form
     c[0]X + c[1]y + c[2]z +c[3] = 0

    This function gives the rotation matrix for rotating
    this plane to the xy-plane, i.e z = 0

    """
    T_rot_xy = np.eye(4)

    abc = c[0]**2+c[1]**2+c[2]**2
    ab = c[0]**2+c[1]**2
    A = c[0]/np.sqrt(abc)
    B = c[1]/np.sqrt(abc)
    C = c[2]/np.sqrt(abc)

    # This formula can be found here:
    # http://math.stackexchange.com/questions/1435018/change-a-3d-plane-to-xy-plane?rq=1
    # Shold perhaps derrive this myself
    T_rot_xy[:3, 0] = [c[1]**2/ab + (1-c[1]**2/ab)*C,
                       -c[0]*c[1]*(1-C)/ab,
                       -A]
    T_rot_xy[:3, 1] = [-c[0]*c[1]*(1-C)/ab,
                       c[0]**2/ab + (1-c[0]**2/ab)*C,
                       -B]
    T_rot_xy[:3, 2] = [A,
                       B,
                       C]
    return T_rot_xy.T


def planefit(pts):
    """
    Assume plane is given by
    ax + by - z + c = 0

    
    """
    
    npts = pts.shape[0]

    x = np.array([pts[i][0] for i in range(npts)])
    y = np.array([pts[i][1] for i in range(npts)])
    z = np.array([pts[i][2] for i in range(npts)])

    one = np.ones(npts)

    A0 = np.ones((3, 3))
    A0[:, 0] = [x.dot(x), x.dot(y), x.dot(one)]
    A0[:, 1] = [y.dot(x), y.dot(y), y.dot(one)]
    A0[:, 2] = [one.dot(x), one.dot(y), one.dot(one)]

    A = np.matrix(A0)
    b = np.array([z.dot(x), z.dot(y), z.dot(one)])

    q = np.linalg.solve(A, b.T)
    c = [q[0], q[1], -1, q[2]]

    return c


def strain_mesh_coordinates(strain_mesh):
    """Given the strain mesh, return the coordinates
    within each semgent in a list

    :param strain_mesh: The strain mesh (24, 15, 3) array
    :returns: coordinates of the strain mesh
    :rtype: list

    """

    strain_region_coords = []
    # Loop over the regions
    for region in range(1, 18):
        
        # Basal segements
        if region in range(1, 7):

            y_start = 10
            y_end = 15
        
            x_start = 4*(region-1)
            x_end = 4*region +1

        # Mid segement
        elif region in range(7, 13):

            y_start = 6
            y_end = 11

            x_start = 4*(region-7)
            x_end = 4*(region-6) + 1

        # Apical segments
        elif region in range(13, 17):

            y_start = 2
            y_end = 7

            x_start = 6*(region-13)
            x_end = 6*(region-12) + 1

        # Apex
        else:
            x_start = 0
            x_end = 24
            
            y_start = 0
            y_end = 3

        dx = x_end-x_start
        dy = y_end-y_start

        strain_points = np.ones((dx*dy, 3))
        t = 0
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                for i in range(3):
                    if x == 24:
                        strain_points[t, i] = strain_mesh[0, y, i].T.ravel()
                    else:
                        strain_points[t, i] = strain_mesh[x, y, i].T.ravel()

                t += 1

        if region == 17:
            # Apex (remove duplicates)
            apex = strain_points[0]
            u = strain_points[1::3]
            v = strain_points[2::3]
            strain_points = np.vstack((v, u, apex))

        strain_region_coords.append(strain_points)

    return strain_region_coords


def get_rotation_matrix(strain_points_orig,
                        return_plane=False, second_layer=False):
    """
    Given the strain mesh, find the rotation matrix
    which aligns the base with the yz plane and with
    apex pointing in the positive x direction

    """

    # Take out the points that defines the base
    # USE SECOND LAYER TO AVOID SEGMENTATION ERRORS AT THE BASE
    if second_layer:
        plane_nodes = strain_points_orig[-48: -24, :3]
    else:
        plane_nodes = strain_points_orig[-24:, :3]

    # Approximate these base points by a plane
    c = planefit(plane_nodes)

    # Create a rotation matrix for rotating the basal plane to the xy-plane
    T_rot_xy = map_plane_to_xy_plane(c)

    # Now flip the axis so that the base lies in the yz plane
    # with apex pointing in the positive x direction
    T_flip = np.zeros((4, 4))
    T_flip[0, 2] = -1
    T_flip[1, 1] = 1
    T_flip[2, 0] = 1
    T_flip[3, 3] = 1

    # Total rotation
    T_rot = T_flip.dot(T_rot_xy)
    # T_rot = T_rot_xy

    if return_plane:
        return T_rot, c
    else:
        return T_rot


def get_translation_matrix(strain_points_rot, endo_verts_rot,
                           round_off_buffer=0.0):
    """
    Create a translation matrix which makes the basal plane of the
    strain mesh align with x = 0. The offset in the y and z direction
    is adjusted so the the endo vertices are centered at the origin

    You can also provide a round off buffer which adjust the offset
    in the x-direction
    """

    # The center offsets for the endo surface
    center_offsets = np.mean(endo_verts_rot, 0)[:3]

    # Make the basal plane the yz-plane
    center_offsets[0] \
        = np.mean(strain_points_rot[-24:, :3], axis=0)[0] - round_off_buffer
      
    # Create translation matrix
    T_trans = np.eye(4)
    T_trans[:3, -1] = -center_offsets
    return T_trans


def get_geometric_matrix(strain_mesh, endo_verts_orig, round_off_buffer=0.0,
                         second_layer=False):
    """Get matrix transformtaion of surfaces that
    transform the surfaces to lie with the base at
    x = 0, and the apex pointing in the postive
    x direction

    :param strain_mesh: The strain mesh
    :param endo_verts_orig: Endocardial vertices
    :param round_off_buffer: Value determining the location of the basal plane
    :param second_layer: If false use the points located at the base
    of the strain mesh to approximate the basal plane (using a least square
    fitting). If true, use the points one the second layer below the base.
    :returns: The transformations matrix
    :rtype: numpy.array

    """

    endo_verts = np.ones((endo_verts_orig.shape[0], 4))
    endo_verts[:, :3] = endo_verts_orig

    # Put strain mesh in an easier format to handle
    strain_points_orig = np.ones((24*15, 4))
    for i in range(3):
        strain_points_orig[:, i] = strain_mesh[:, :, i].T.ravel()

    # Get rotation matrix for aligning the base with yz-plane
    # and apex pointing in the postive x-direction
    return_plane = True
    T_rot, c = get_rotation_matrix(strain_points_orig, return_plane,
                                   second_layer)

    # Rotate strain mesh
    strain_points_rot = T_rot.dot(strain_points_orig.T).T

    # Rotate endo vertices
    try:
        endo_verts_rot = T_rot.dot(endo_verts.T).T
    except:
        from IPython import embed; embed()
    # Get traslation matrix
    T_trans = get_translation_matrix(strain_points_rot,
                                     endo_verts_rot,
                                     round_off_buffer)

    # Total geometric operation
    T_tot = T_trans.dot(T_rot)

    # Test that apex is pointing in posivitve x-direction
    strain_points = T_tot.dot(strain_points_orig.T).T

    if np.mean(strain_points, axis=0)[0] < 0:
        # flip everyting, and make sure the matrix has determinant 1
        T_flip = np.eye(4)
        T_flip[0, 0] = -1
        T_flip[1, 1] = -1
        T_tot = T_flip.dot(T_tot)

    return T_tot


def transform_surfaces(round_off, endo_verts, epi_verts,
                       strain_mesh, **kwargs):
    """Transform the surfaces to that the long axis points
    along the x-axis, the base located at x = 0, and apex at x>0.

    :param round_off: Value determining the location of the basal plane
    :param endo_verts: Vertices for endocardium
    :param epi_verts: Vertices for epicardium
    :param strain_mesh: The strain mesh
    :returns: A dictionart containing the transformed surfaces together
    with the transformation matrix
    :rtype: dict

    """
    
    # Put the data into 4-dimensional matrices for easier manipulation
    endo_verts_orig = np.ones((endo_verts.shape[0], 4))
    endo_verts_orig[:, :3] = np.copy(endo_verts)

    epi_verts_orig = np.ones((epi_verts.shape[0], 4))
    epi_verts_orig[:, :3] = np.copy(epi_verts)

    strain_points_orig = np.ones((24*15, 4))
    strain_points_orig[:, :3] = np.reshape(strain_mesh, (24*15, 3))

    strain_mesh_orig = np.copy(strain_mesh)

    # Get transformation matrix
    T = get_geometric_matrix(strain_mesh_orig, endo_verts_orig,
                             round_off, second_layer=False)

    # Transform the data
    # The surfaces
    endo_verts = T.dot(endo_verts_orig.T).T[:, :3]
    epi_verts = T.dot(epi_verts_orig.T).T[:, :3]

    # The strain mesh
    strain_points = T.dot(strain_points_orig.T).T[:,:3]
    strain_mesh = np.reshape(strain_points, (24, 15, 3))

    data = {"T": T,
            "endo_verts": endo_verts,
            "epi_verts": epi_verts,
            "strain_mesh": strain_mesh}

    kwargs.update(**data)

    kwargs["strain_coords"] = get_strain_region_coordinates(data["strain_mesh"])

    return kwargs


class WrongGeometryException(Exception):
    pass


def compute_focal_point(original_long_axis_endo,
                        endo_verts, epi_verts, **kwargs):
    """Copmute the focal point based on approximating the
    endocardial surfaces as a ellipsoidal cap.
    
    .. math::

           focal = \sqrt{ l^2 - s^2}


    :param original_long_axis_endo: Long axis of the surfaces without cut
    :param endo_verts: endocardial vertices
    :param epi_verts: epicardial vertices
    :returns: focal point
    :rtype: float

    """

    long_axis_endo = np.max(endo_verts, 0)[0]
    short_axis_endo = np.max(np.max(endo_verts, 0)[1:] -
                             np.min(endo_verts, 0)[1:])

    # Cut is probably to big
    if (long_axis_endo < short_axis_endo):
        # But it might be possible
        # A prolate ellipsoid is probably not
        # the best model though

        # Just return something
        return long_axis_endo/2.0
        # raise WrongGeometryException("Cut is too large")

    # No cut has been made
    if original_long_axis_endo - long_axis_endo < 0.5:
        raise WrongGeometryException("No cut has been made")

    foc_endo = np.sqrt(long_axis_endo**2 - short_axis_endo**2)

    return foc_endo


def save_surfaces_to_ply(plydir, time,  data, postfix=""):
    """Save the surfaces to *.ply format which can be used as
    input to Gmsh later

    :param plydir: Directore for the ply files
    :param time: the timestamp (index) from echo
    :param data: the surface data

    """
    if not os.path.exists(plydir):
        os.makedirs(plydir)

    epi_out = os.path.join(plydir, "epi_lv_{}{}.ply".format(time, postfix))
    endo_out = os.path.join(plydir, "endo_lv_{}{}.ply".format(time, postfix))

    writeplyfile(endo_out, data["endo_verts"]*1.,
                 data["endo_faces"]+1)
    writeplyfile(epi_out, data["epi_verts"]*1.,
                 data["epi_faces"]+1)


def cut_off_base(faces, points_in, cut_off_point=.0):
    """Cut off x-values less than cut_off_point
    Written by Sjur Gjerald
    """
    faces_red = []
    points = points_in.copy()
    added_points = []
    duplicate_index1 = []
    duplicate_index2 = []

    for iii in range(len(faces)):
        # Cut based on x-coor
        local_x = points[faces[iii, :].astype(int), 0]
        is_positive = local_x > cut_off_point
        
        if all(is_positive):
            faces_red.append(faces[iii, :])
        elif any(is_positive):
            # Create new points at instersection x=0
            point1 = np.zeros(3)
            point2 = np.zeros(3)
            
            # Count number of negative vertices
            if np.sum(is_positive) == 1:
                jjj = np.argmax(local_x)
            else:
                jjj = np.argmin(local_x)
            
            # Move along edges and cut at x=0
            d1 = points[faces[iii, (jjj+1) % 3].astype(int), 0] \
                - points[faces[iii, jjj].astype(int), 0]
            d2 = points[faces[iii, (jjj+2) % 3].astype(int), 0] \
                - points[faces[iii, jjj].astype(int), 0]

            
            if d1 != 0.0: 
                t1 = - points[faces[iii, jjj].astype(int), 0]/d1
                in_edge1 = (t1 >= 0.0) * (t1 <= 1.0)
            else:
                in_edge1 = False
            if d2 != 0.0:
                t2 = - points[faces[iii,jjj].astype(int),0]/d2
                in_edge2 = (t2>=0.0)*(t2<=1.0)
            else:
                in_edge2 = False

            
            if not in_edge1: t1 = 1.0
            if not in_edge2: t2 = 1.0
            # Fill in y and z coordinates of new point
            for kk in range(2):
                p1 = points[faces[iii,jjj].astype(int),kk+1]*(1.-t1)+points[faces[iii,(jjj+1)%3].astype(int),kk+1]*t1
                p2 = points[faces[iii,jjj].astype(int),kk+1]*(1.-t2)+points[faces[iii,(jjj+2)%3].astype(int),kk+1]*t2
                
                point1[kk+1]=p1
                point2[kk+1]=p2
            
            if len(added_points):
                duplicate_index1=np.where(np.sqrt(np.sum((point1-added_points)**2,axis=1))<1e-15)[0]
                duplicate_index2=np.where(np.sqrt(np.sum((point2-added_points)**2,axis=1))<1e-15)[0]
            
            if len(duplicate_index1):
                new_index1 = len(points)+int(duplicate_index1[0])
            else: 
                new_index1 = len(points)+len(added_points)
                added_points.append(point1)
                
            if len(duplicate_index2):
                new_index2 = len(points)+int(duplicate_index2[0])
            else: 
                new_index2 = len(points)+len(added_points)
                added_points.append(point2)

            if np.sum(is_positive)==1:
                # If one positive vertices -> create one triangle
                new_face = faces[iii,:].astype(int)
                new_face[(jjj+1)%3] = new_index1
                new_face[(jjj+2)%3] = new_index2
                faces_red.append(new_face)
            else:
                # If two positive vertices -> create two triangles
                new_face1 = faces[iii,:].astype(int)
                new_face1[jjj] = new_index1
                new_face2 = faces[iii,:].astype(int)
                new_face2[jjj] = new_index2
                new_face2[(jjj+1)%3] = new_index1
                
                faces_red.append(new_face1)
                faces_red.append(new_face2)
            
            
    # Set all new points
    # Duplicates have NOT been removed
    
    if len(added_points):
        points = np.vstack((points,np.asarray(added_points)))
    
    faces_red = np.asarray(faces_red).astype(int)
    
    return faces_red, points


def remove_base(endo_faces, endo_verts, epi_faces,
                epi_verts, **kwargs):
    """Remove base (points below the x = 0 plane)
    from endocardial and epicardial surfaces

    :param endo_faces: endocardial faces
    :param endo_verts: endocardial vertices
    :param epi_faces:  epicardial faces
    :param epi_verts:  epicardial vertices
    :returns: surfaces with based removed
    :rtype: dict

    """

   
    endo_faces, endo_verts = cut_off_base(endo_faces, endo_verts)
    epi_faces, epi_verts = cut_off_base(epi_faces, epi_verts)
    

    data = {"endo_verts": endo_verts,
            "epi_verts": epi_verts,
            "endo_faces": endo_faces,
            "epi_faces": epi_faces}


   
    kwargs.update(**data)
    
    
    return kwargs


def smooth_surface_gamer(data):
    """Smooth surfaces using gamer

    :param data: Dictionary containing data with keys 
    'endo_verts', 'endo_faces', 'epi_verts' and 'epi_faces'.

    :returns: The same data, with updated faces and vertices smoothed by gamer. 
    :rtype: dict

    """

    try:
        import gamer
    except ImportError:
        logger.warning("GAMer is not available")
        return data
    else:
        logger.info("Smooth surfaces using GAMer")


    for layer in ["endo", "epi"]:
        verts = data["{}_verts".format(layer)]
        faces = data["{}_faces".format(layer)]

        gmesh = gamer.SurfaceMesh(len(verts), len(faces))

        for i, bverts in enumerate(verts):
            
            gvert = gmesh.vertex(i)
            gvert.x, gvert.y, gvert.z = np.array(bverts, dtype=float)
            gvert.sel=True
    
    
        for i, bface in enumerate(faces):

            gface = gmesh.face(i)
            gface.a, gface.b, gface.c = tuple(map(int,bface))
            gface.sel=True

        # Use default options for now
        gmesh.smooth()
        gmesh.coarse_dense()
        gmesh.smooth()
        gmesh.normal_smooth()
        gmesh.smooth()

        new_verts = np.asarray([(gvert.x, gvert.y, gvert.z) for gvert in gmesh.vertices()])
        new_faces = np.asarray([(gface.a, gface.b, gface.c) for gface in gmesh.faces()])

        data["{}_verts".format(layer)] = new_verts
        data["{}_faces".format(layer)] = new_faces
        
    return data
