import sys
from textwrap import dedent


def asint(s):
    try:
        return int(s), ''
    except ValueError:
        return sys.maxint, s


body = dedent(
              """<?xml version="1.0"?>
              <Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
              <Domain>
              {body}
              </Domain>
              </Xdmf>""")


series = dedent(
                """<Grid Name="{name}" GridType="Collection" CollectionType="Temporal">
                <Time TimeType="List">
                <DataItem Format="XML" Dimensions="{N}"> {lst}</DataItem>
                </Time>
                {entry}
                </Grid>""")


entry_single = dedent(
                      """<Grid Name="time_{iter}" GridType="Uniform">
                      {frame}
                      </Grid>
                      """)

entry = dedent(
               """<Grid Name="time_{iter}" GridType="Uniform">
               {frame}
               </Grid>
               """)


topology = dedent(
                  """<Topology NumberOfElements="{ncells}" TopologyType="{cell}">
                  <DataItem Dimensions="{ncells} {dim}" Format="HDF">{h5name}:/{h5group}</DataItem>
                  </Topology>""")


topology_polyvert = dedent(
                           """<Topology TopologyType="Polyvertex" NodesPerElement="{nverts}">
                           </Topology>""")


geometry = dedent(
                  """<Geometry GeometryType="{coords}">
                  <DataItem Dimensions="{nverts} {dim}" Format="HDF">{h5name}:/{h5group}</DataItem>
                  </Geometry>""")


vector_attribute = dedent(
                          """<Attribute Name="{name}" AttributeType="Vector" Center="{center}">
                          <DataItem Format="HDF" Dimensions="{nverts} {dim}">{h5name}:/{h5group}</DataItem>
                          </Attribute>""")


scalar_attribute = dedent(
                          """<Attribute Name="{name}" AttributeType="Scalar" Center="{center}">
                          <DataItem Format="HDF" Dimensions="{nverts} {dim}">{h5name}:/{h5group}</DataItem>
                          </Attribute>""")


if __name__ == "__main__":
    pass
