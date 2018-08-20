import dolfin



def mark_cell_function(fun, mesh, foc, regions):
    """
    Iterates over the mesh and stores the
    region number in a meshfunction
    """

    for cell in dolfin.cells(mesh):

        # Get coordinates to cell midpoint
        x = cell.midpoint().x()
        y = cell.midpoint().y()
        z = cell.midpoint().z()

        T = cartesian_to_prolate_ellipsoidal(x, y, z, foc)
        fun[cell] = strain_region_number(T, regions)

    return fun


def mark_lv_strain_regions(fun, mesh, focal_point,
                           strain_regions, strain_type):
    """
    Takes in a dolfin meshfunction (fun) and stores the
    region number (1-17) for each cell in that mesh function
    """

    regions = set_region_borders(strain_regions, focal_point, strain_type)

    # Fix the bounds on the regions so that each point belongs
    # to a unique segment
    base_rng = range(6)
    BASE_LOW = np.mean([regions[i][3] for i in base_rng])
    BASE_MAX = np.inf
    regions[base_rng, 3] = BASE_LOW
    regions[base_rng, 0] = BASE_MAX

    mid_rng = range(6, 12)
    MID_LOW = np.mean([regions[i][3] for i in mid_rng])
    regions[mid_rng, 3] = MID_LOW
    regions[mid_rng, 0] = BASE_LOW
    
    ap_rng = range(12, 16) if np.shape(regions)[0] < 18 else range(12, 18)
    APICAL_LOW = np.mean([regions[i][3] for i in ap_rng])
    regions[ap_rng, 3] = APICAL_LOW
    regions[ap_rng, 0] = MID_LOW

    fun = mark_cell_function(fun, mesh, focal_point, regions)

    return fun


def full_arctangent(x, y):
    t = np.arctan2(x, y)
    if t < 0:
        return t + 2*np.pi
    else:
        return t


def fill_coordinates_ec(i, e_c_x, e_c_y, e_c_z, coord, foci):
    norm = dolfin.sqrt(coord[1]**2 + coord[2]**2)
    if not dolfin.near(norm, 0):
        e_c_y.vector()[i] = -coord[2]/norm
        e_c_z.vector()[i] = coord[1]/norm
    else:
        # We are at the apex where clr system doesn't make sense
        # So just pick something.
        e_c_y.vector()[i] = 1
        e_c_z.vector()[i] = 0


def fill_coordinates_el(i, e_c_x, e_c_y, e_c_z, coord, foci):

    norm = dolfin.sqrt(coord[1]**2 + coord[2]**2)
    if not dolfin.near(norm, 0):
        mu, nu, phi = cartesian_to_prolate_ellipsoidal(*(coord.tolist() + [foci]))
        x, y, z = prolate_ellipsoidal_to_cartesian(mu, nu + 0.01, phi, foci)
        r = np.array([coord[0] - x,
                      coord[1] - y,
                      coord[2] - z])
        e_r = r/np.linalg.norm(r)
        e_c_x.vector()[i] = e_r[0]
        e_c_y.vector()[i] = e_r[1]
        e_c_z.vector()[i] = e_r[2]
    else:
        e_c_y.vector()[i] = 0
        e_c_z.vector()[i] = 1


def calc_cross_products(e1, e2, VV):
    e_crossed = dolfin.Function(VV)

    e1_arr = e1.vector().array().reshape((-1, 3))
    e2_arr = e2.vector().array().reshape((-1, 3))

    crosses = []
    for c1, c2 in zip(e1_arr, e2_arr):
        crosses.extend(np.cross(c1, c2.tolist()))

    e_crossed.vector()[:] = np.array(crosses)[:]
    return e_crossed


def check_norms(e):

    e_arr = e.vector().array().reshape((-1, 3))
    for e_i in e_arr:
        assert(dolfin.near(np.linalg.norm(e_i), 1.0))


def make_unit_vector(V, VV, dofs_x, fill_coordinates, foc=None):
    e_c_x = dolfin.Function(V)
    e_c_y = dolfin.Function(V)
    e_c_z = dolfin.Function(V)

    for i, coord in enumerate(dofs_x):
        fill_coordinates(i, e_c_x, e_c_y, e_c_z, coord, foc)

    e = dolfin.Function(VV)

    fa = [dolfin.FunctionAssigner(VV.sub(i), V) for i in range(3)]
    for i, e_c_comp in enumerate([e_c_x, e_c_y, e_c_z]):
        fa[i].assign(e.split()[i], e_c_comp)
    return e


def make_crl_basis(mesh, foc, space="Quadrature_4"):
    """
    Makes the crl  basis for the idealized mesh (prolate ellipsoidal)
    with prespecified focal length.
    """

    msg = ("Creating local basis function in the "
           "circumferential, radial and longitudinal "
           "direction...")
    logger.info(msg)
    family, degree = space.split("_")
    fem = dolfin.FiniteElement(family=family,
                               cell=mesh.ufl_cell(),
                               degree=int(degree),
                               quad_scheme="default")
    vem = dolfin.VectorElement(family=family,
                               cell=mesh.ufl_cell(),
                               degree=int(degree),
                               quad_scheme="default")
    
    VV = dolfin.FunctionSpace(mesh, vem)
    V = dolfin.FunctionSpace(mesh, fem)

    if dolfin.DOLFIN_VERSION_MAJOR > 1.6:
        dofs_x = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    else:
        dm = V.dofmap()
        dofs_x = dm.tabulate_all_coordinates(mesh).reshape((-1, mesh.geometry().dim()))

    logger.info("Creating circumferential")
    e_c = make_unit_vector(V, VV, dofs_x, fill_coordinates_ec)
    logger.info("Done creating circumferential")
    logger.info("Creating longitudinal")
    e_l = make_unit_vector(V, VV, dofs_x, fill_coordinates_el, foc)
    logger.info("Done creating longitudinal")
    logger.info("Creating radial")
    e_r = calc_cross_products(e_c, e_l, VV)
    logger.info("Done creating radial")

    e_c.rename("circumferential", "local_basis_function")
    e_r.rename("radial", "local_basis_function")
    e_l.rename("longitudinal", "local_basis_function")

    return e_c, e_r, e_l
