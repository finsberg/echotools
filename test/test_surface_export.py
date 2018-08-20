import os
import echotools

curdir = os.path.abspath(os.path.dirname(__file__))

echo = echotools.EchoFile(os.path.join(curdir, 'test_data.h5'))

polydata = echotools.export.to_polydata(echo.endo_verts(0),
                                        echo.endo_faces)
echotools.export.write_ply(os.path.join(curdir, 'test.ply'), polydata)

grid = echotools.export.to_grid(echo.endo_verts(0),
                                echo.endo_faces)
echotools.export.write_vtu(os.path.join(curdir, 'test.vtu'), grid)

