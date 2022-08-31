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

"""
 Partitions left ventricle into regions based on strain
 mesh generated from EchoPAC. Genereates local basis
 vector pointing in the radial, cicumferential and
 longitudinal direction
"""
import numpy as np


def cartesian_to_prolate_ellipsoidal(x, y, z, a):

    b1 = np.sqrt((x + a) ** 2 + y ** 2 + z ** 2)
    b2 = np.sqrt((x - a) ** 2 + y ** 2 + z ** 2)

    sigma = 1 / (2.0 * a) * (b1 + b2)
    tau = 1 / (2.0 * a) * (b1 - b2)
    phi = full_arctangent(z, y)
    mu = np.arccosh(sigma)
    nu = np.arccos(tau)
    return mu, nu, phi


def prolate_ellipsoidal_to_cartesian(mu, nu, phi, a):
    x = a * np.cosh(mu) * np.cos(nu)
    y = a * np.sinh(mu) * np.sin(nu) * np.cos(phi)
    z = a * np.sinh(mu) * np.sin(nu) * np.sin(phi)
    return x, y, z


def set_region_borders(strain_regions, foc, strain_type):

    """
    Creates a 16 by 4 marix whith coorinates
    which defines the different regions.

    Example:
    region_sph_coord[0][0] = upper bound for region 1
    region_sph_coord[0][1] = left bound for region 1
    region_sph_coord[9][2] = right bound for region 10
    region_sph_coord[7][3] = lower bound for region 8

    There are 16 regions (neglecting apex)
    For each region there is a start value
    and end value of longitudinal and a start
    value and end value of circumferential.
    So (region,[upper, right, left, lower])
    i.e region_sph_coord(0,0) gives start mu
    for region 1
    """

    region_sph_coord = np.zeros((18, 4)) if strain_type == "18" else np.zeros((16, 4))
    rng = list(range(18)) if strain_type == "18" else list(range(16))

    # Mid and Basal
    upper = list(range(4, 25, 5))
    left = list(range(5))
    right = list(range(20, 25))
    lower = list(range(0, 21, 5))

    # Apical
    upper_ap = list(range(4, 35, 5))
    left_ap = list(range(5))
    right_ap = list(range(30, 35))
    lower_ap = list(range(0, 31, 5))

    for j in rng:

        """
        If 0<=j<=11 then the region are in the
        basal or mid. Then there are in total
        25 nodes for this region. The corner
        points are point number 0 (upper right),
        4 (lower right), 20 (upper left) and
        24 (lower left).
        """
        if j in range(12):

            # upper
            for k in upper:
                p = strain_regions[j][k]
                # Convert to prolate coordinates
                T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))

                # Divide by 5 to get the average value
                region_sph_coord[j][0] += T[1] / 5

            # Left and right
            for side, coords in enumerate([left, right], start=1):

                arr = []
                for k in coords:
                    p = strain_regions[j][k]
                    T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                    arr.append(T[2])

                # Because of periodicity we might get values far appart,
                # e.g eps and 2pi - eps
                # Therefore we remove points, until the variace is low
                # so that the mean represents the acutal value.
                std = np.std(arr)
                while std > 1.0:
                    arr.pop()
                    std = np.std(arr)
                region_sph_coord[j][side] = np.mean(arr)

                # Just add pi so that the circumferential direction
                # goes from 0 to 2*pi
                region_sph_coord[j][side] = region_sph_coord[j][side] + np.pi

            if strain_type == "18" or j in range(0, 6):
                # The apical segements are similar to the mid segment
                for k in lower:
                    p = strain_regions[j][k]
                    T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                    region_sph_coord[j][3] += T[1] / 5

            else:  # "strain_type" in ["16", "17"]
                # The apical segments span more cells in the
                # circumferential direction than the mid segments.
                # To avoid gaps we need to average the lower coordinates in the
                # mid segments over a larger domain

                if j in range(6, 9):
                    for t in range(6, 9):
                        for k in range(0, 16, 5):
                            p = strain_regions[t][k]
                            T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                            region_sph_coord[j][3] += T[1] / 13

                    # Add the last point
                    p = strain_regions[8][20]
                    T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                    region_sph_coord[j][3] += T[1] / 13

                elif j in range(9, 12):
                    for t in range(9, 12):
                        for k in range(0, 16, 5):
                            p = strain_regions[t][k]
                            T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                            region_sph_coord[j][3] += T[1] / 13

                    # Add the last point
                    p = strain_regions[11][20]
                    T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                    region_sph_coord[j][3] += T[1] / 13
                else:
                    print("What!!!??")

        # Apical segments
        if j >= 12:

            # Upper
            if strain_type == "18":

                # Choose the lower bound for the adjacent mid segment
                region_sph_coord[j][0] = region_sph_coord[j - 6][3]

            else:
                if j in range(12, 14):
                    for t in range(12, 14):
                        for k in range(4, 30, 5):
                            p = strain_regions[t][k]
                            T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                            region_sph_coord[j][0] += T[1] / 13
                    # Add the last point
                    p = strain_regions[13][34]
                    T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                    region_sph_coord[j][0] += T[1] / 13

                else:  # j in range(14,16)
                    for t in range(14, 16):
                        for k in range(4, 30, 5):
                            p = strain_regions[t][k]
                            T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                            region_sph_coord[j][0] += T[1] / 13
                    # Add the last point
                    p = strain_regions[15][34]
                    T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                    region_sph_coord[j][0] += T[1] / 13

            # Left and right
            for side, coords in enumerate([left_ap, right_ap], start=1):

                if strain_type == "18":

                    # Choose the bound for the adjacent mid segment
                    region_sph_coord[j][side] = region_sph_coord[j - 6][side]

                else:
                    arr = []
                    for k in coords:
                        p = strain_regions[j][k]
                        T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                        arr.append(T[2])

                    # Because of periodicity we might get values far appart,
                    # e.g eps and 2pi - eps
                    # Therefore we remove points, until the variace is low
                    # so that the mean represents the acutal value.
                    std = np.std(arr)
                    while std > 1.0:
                        arr.pop()
                        std = np.std(arr)
                    region_sph_coord[j][side] = np.mean(arr)

                    # Just add pi so that the circumferential direction
                    # goes from 0 to 2*pi
                    region_sph_coord[j][side] = region_sph_coord[j][side] + np.pi

            for k in lower_ap:
                if strain_type in ["16", "18"]:
                    region_sph_coord[j][3] = 0
                else:
                    p = strain_regions[j][k]
                    T = cartesian_to_prolate_ellipsoidal(*(p.tolist() + [foc]))
                    region_sph_coord[j][3] += T[1] / 7

    return region_sph_coord


def get_sector(regions, theta):

    if not (
        np.count_nonzero(regions.T[1] <= regions.T[2]) >= 0.5 * np.shape(regions)[0]
    ):
        raise ValueError("Surfaces are flipped")

    sectors = []
    for i, r in enumerate(regions):

        if r[1] == r[2]:
            sectors.append(i)
        else:

            if r[1] > r[2]:
                if theta > r[1] or r[2] > theta:
                    sectors.append(i)

            else:
                if r[1] < theta < r[2]:
                    sectors.append(i)

    return sectors


def get_level(regions, mu):

    A = np.intersect1d(
        np.where((regions.T[3] <= mu))[0], np.where((mu <= regions.T[0]))[0]
    )
    if len(A) == 0:
        return [np.shape(regions)[0] + 1]
    else:
        return A


def strain_region_number(T, regions):
    """
    For a given point in prolate coordinates,
    return the region it belongs to.

    :param regions: Array of all coordinates for the strain
                    regions taken from the strain mesh.
    :type regions: :py:class:`numpy.array`

    :param T: Some value i prolate coordinates
    :type T: :py:class:`numpy.array`

    Resturn the region number that
    T belongs to
    """

    """
    The cricumferential direction is a bit
    tricky because it goes from -pi to pi.
    To overcome this we add pi so that the
    direction goes from 0 to 2*pi
    """

    lam, mu, theta = T

    theta = theta + np.pi

    levels = get_level(regions, mu)

    if np.shape(regions)[0] + 1 in levels:
        return np.shape(regions)[0] + 1

    sector = get_sector(regions, theta)

    assert len(np.intersect1d(levels, sector)) == 1

    return np.intersect1d(levels, sector)[0] + 1


def strain_faces(region_number):
    if region_number in range(1, 13):
        # Then we are at basal or mid meaning that
        # each region has 9 points
        """
        4----9---14---19---24
        |    |    |    |    |
        3----8---13---18---23
        |    |    |    |    |
        2----7---12---17---22
        |    |    |    |    |
        1----6---11---16---21
        |    |    |    |    |
        0----5---10---15---20
        """

        faces = [[i, i + 1, i + 5] for i in [i for i in range(20) if (i + 1) % 5]]
        for a in [[i, i - 4, i + 1] for i in [i for i in range(5, 25) if (i + 1) % 5]]:
            faces.append(a)

    elif region_number in range(13, 17):
        """
        4----9---14---19---24---29---34
        |    |    |    |    |   |    |
        3----8---13---18---23---28---33
        |    |    |    |    |   |    |
        2----7---12---17---22---27---32
        |    |    |    |    |   |    |
        1----6---11---16---21---26---31
        |    |    |    |    |   |    |
        0----5---10---15---20---25---30
        """

        faces = [[i, i + 1, i + 5] for i in [i for i in range(30) if (i + 1) % 5]]
        for a in [[i, i - 4, i + 1] for i in [i for i in range(5, 35) if (i + 1) % 5]]:
            faces.append(a)

    if region_number == 17:
        faces = []
        for i in range(23):
            faces.append([i, 24 + i, i + 1])
            faces.append([i + 1, 24 + i, 24 + i + 1])
            faces.append([24 + i, 48, 24 + i + 1])
        faces.append([23, 47, 0])
        faces.append([0, 47, 24])
        faces.append([47, 48, 24])

    return np.array(faces)
