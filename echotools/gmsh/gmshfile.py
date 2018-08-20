"""FIXME:"""
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
from reference_topology import *
from collections import defaultdict
import operator
import csv
import re
import numpy as np
import sys

__all__ = ["GmshFile"]

class GmshFile(object):
    """
    Interface to GMSH file data structure.
    """

    def __init__(self, ifilename=None, geometric_dimension=None,
                       verbose=False, marker_ids=None):
        """FIXME:"""
        self.verbose = verbose
        self.user_marker_ids = marker_ids or {}
        self.marker_mapping = {}
        
        # Parsing the file
        with open(ifilename, 'rb') as f: self._digest_file(f)

        # during the reading node indices are mapped, so this is
        # not necessary anymore
        del self._nodeids

        # convert entity node list to numpy array
        for d in [ "_entities", "_ent_tags" ]:
            setattr(self, d, { k: np.array(v, dtype='uintp') 
                for k, v in getattr(self, d).iteritems() })

        # restrict coordinates (dolfin assumes so)
        eps = sys.float_info.epsilon
        nnz_coords = np.where(np.ptp(self._coords, axis = 0) > eps)[0]
        geodim = geometric_dimension or len(nnz_coords)
        
        # check if geodim is valid
        if not isinstance(geodim, int):
            raise ValueError("Expecting an int for geometric dimension.")
        if not (len(nnz_coords) <= geodim <= 3):
            raise ValueError("Not a valid geometric dimension.")
        
        # FIXME this is not always true, e.g. 1d embedded in 3d
        if geodim > len(nnz_coords):
            nnz_coords = np.arange(0, geodim)
        
        # restrict coordinates
        self.geom_dim = geodim
        self._coords = self._coords[:, nnz_coords]

        # compute the cell type and topological dimension
        self.cell_type, self.topo_dim = max([ 
                (k, gmsh_shape[k.split("_")[0]].dimension()) 
                for k in self._entities.iterkeys() ], \
            key = operator.itemgetter(1))

        # order of the mesh
        self.mesh_order = int(self.cell_type.split("_")[1])

        # update marker name with the unamed tags
        markers_from_entity = defaultdict(None)
        for k, v in self._ent_tags.iteritems():
            dim = gmsh_shape[k.split("_")[0]].dimension()
            for marker in np.unique(v):
                markers_from_entity[self.marker_mapping[marker]] = (None, dim)

        if hasattr(self, 'marker_name'):
            markers_from_entity.update(self.marker_name)
        self.marker_name = markers_from_entity

        # Update entities in gmshfile to new markers 
        for codim in range(4):
            if self.entity_markers(codim) is not None:
                for l in range(len(self.entity_markers(codim))):
                    self.entity_markers(codim)[l] = self.marker_mapping[\
                        self.entity_markers(codim)[l]]

        # fix point_{order} (always 1 by default)
        newpt = "point_{}".format(self.mesh_order)
        if 'point_1' in self._entities:
            self._entities[newpt] = self._entities.pop("point_1")
        if 'point_1' in self._ent_tags:
            self._ent_tags[newpt] = self._ent_tags.pop("point_1")

    def print_info(self):
        """FIXME"""
        print("Mesh info:")
        print("  # vertices = %d" % len(self.vertices()))
        print("  # dofs     = %d" % self._coords.shape[0])
        print("  # cells    = %d" % len(self._entities[self.cell_type]))
        print("  order      = %d" % self.mesh_order)
        print("  embed dim  = %d" % self.geom_dim)
        print("  topol dim  = %d" % self.topo_dim)

        if not len(self._ent_tags): return
        print("Entity markers:")
        for shape, v in self._ent_tags.iteritems():
            # counts the number of marked entities for each tag
            dim = gmsh_shape[shape.split("_")[0]].dimension()
            keys = np.unique(v)
            counts = np.bincount(keys.searchsorted(v))
            names = [ self.marker_name[k][0] for k in keys ]
            print("  {}d: {}".format(dim, 
                "\n       ".join("{} => {} ({} entities)".format(k, n, c) 
                      for k, c, n in zip(keys, counts, names))))

    def _digest_file(self, fhandle):

        # implemented section handlers
        sections = [ s.split("_")[1] for s in dir(self) \
                if re.match(r"^parse_([a-zA-Z])+$", s) ]

        # parse as csv
        csv_file = csv.reader(fhandle, 
                delimiter = ' ', 
                skipinitialspace = True)
        # re for matching sections
        re_sect = re.compile(r"^\$(|End)([a-zA-Z]+)")
        
        # read data from input file
        cur_sect = None
        for l in csv_file:
            # skip empty lines
            if not len(l): continue
            # look for a tag
            tag_sect = re_sect.match(l[0])
            if tag_sect:
                if tag_sect.group(1) == "End":
                    cur_sect = None
                else:
                    cur_sect = tag_sect.group(2)
            else:
                # call the correct method
                getattr(self, "_parse_%s" % cur_sect)(l)

    def _parse_MeshFormat(self, l):
        version_number, file_type, data_type = l
        if version_number != "2.2" and file_type != 0:
            raise Exception("File not supported!")
        
        if self.verbose: print("Found supported GMSH header")

    def _parse_Nodes(self, l):

        # first line is the number of nodes
        if len(l) == 1:
            # reset the index
            self._cindex = 0
            totnodes = int(l[0])
            self._coords  = np.zeros((totnodes, 3), dtype=np.float)
            self._nodeids = {}
            if self.verbose: print("Expecting %d nodes" % totnodes)
            return

        # first number is the gmsh id
        # NB: not necessary in increasing order!
        gmsh_id = int(l[0])
        self._nodeids[gmsh_id] = self._cindex

        # then the coordinates (always 3d)
        self._coords[self._cindex,:] = map(float, l[1:])
        self._cindex += 1

    def _parse_Elements(self, l):

        # first line is the number of entities
        if len(l) == 1:
            totelms = int(l[0])
            self._entities = defaultdict(lambda: [])
            self._ent_tags = defaultdict(lambda: [])
            if self.verbose: print("Expecting %d entities" % totelms)
            return

        # A gmsh entity is represented by
        # - id:    ignored, since not necessary increasing
        # - type:  the shape of the entity (see shapes)
        # - ntags: number of following tags (always at least 1)
        # - tags:  an integer marker
        # - dofs:  nodes representing the entity

        # we pick up ony the first tag
        _, elm_type, num_tags, elm_tag = map(int, l[0:4])

        # check if element is supported
        try:
            ufc_shape, elm_order = msh2ufc_shape[elm_type]
        except KeyError:
            raise Exception("Shape currently not supported!")

        # key signature of the element
        entity_signature = "{}_{}".format(ufc_shape, elm_order)

        # dofs
        dofs = map(lambda dof: self._nodeids[int(dof)], l[3 + num_tags:])

        # updating the list
        self._entities[entity_signature].append(dofs)

        # The first tag is the geometric entity id, used
        # during the definition of the geometry in gmsh
        # If physical markers are present, the geometric
        # one is ignored
        elm_tag = elm_tag if num_tags >= 2 else 0
        self._ent_tags[entity_signature].append(elm_tag)

    def _parse_PhysicalNames(self, l):
        
        # first line is the number of names
        if len(l) == 1:
            if self.verbose: print("Expecting {} physical names".format(l[0]))
            self.marker_name = defaultdict(None)
            return

        # physical_{dim, id, name}
        phy_dim, phy_id, phy_name = int(l[0]), int(l[1]), l[2]

        # Update with user given id for a corresponding domain name
        phy_id_old = phy_id
        phy_id = self.user_marker_ids.get(phy_name, phy_id)
        
        self.marker_mapping[phy_id_old] = phy_id
        self.marker_name[phy_id] = (phy_name, phy_dim)

    def _shape_from_codim(self, codim):
        dim = self.topo_dim
        cell_shape = self.cell_shape()
        
        if dim < codim:
            return None
        elif codim == 0:
            enti_shape = cell_shape
        else:
            enti_shape = gmsh_shape_entity[cell_shape][dim-codim]

        return "{}_{}".format(enti_shape, self.mesh_order)

    def _entity_exists(self, codim):
        
        enti_type = self._shape_from_codim(codim)
        enti_shape = enti_type.split("_")[0]

        if enti_type is None:
            return False

        if enti_type not in self._entities.keys():
            return False

        return True

    def entities(self, codim, dofs = False):
        """FIXME:"""
        if not self._entity_exists(codim):
            return None

        enti_type = self._shape_from_codim(codim)
        enti_shape = enti_type.split("_")[0]

        if dofs:
            ent = self._entities[enti_type]
        else:
            vert = gmsh_shape[enti_shape].num_vertices()
            ent = self._entities[enti_type][:, 0:vert]

        return ent

    def entity_markers(self, codim):
        """FIXME:"""
        enti_type = self._shape_from_codim(codim)
        if enti_type in self._ent_tags:
            return self._ent_tags[enti_type]
        else:
            return None

    def cells(self, dofs = False):
        """FIXME:"""
        return self.entities(0, dofs)

    def dof_coords(self):
        """FIXME:"""
        return self._coords

    def vertices(self):
        """FIXME:"""
        return np.unique(self.cells().ravel())

    def cell_shape(self):
        """FIXME:"""
        return self.cell_type.split("_")[0]
