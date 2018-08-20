import os

def get_status_output(cmd, inp=None, cwd=None, env=None):
    from subprocess import Popen, PIPE, STDOUT
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=inp)
    assert not errout

    status = pipe.returncode

    return (status, output)


curdir = os.path.abspath(os.path.dirname(__file__))
test_data = os.path.join(curdir, 'test_data.h5')
# fail, output = get_status_output('echo2dolfin "test/test_data.h5"')


for outformat in ['ply', 'vtk', 'vtu']:
    for time in [0, -1]:

fail, output = get_status_output('echoexport "{}"'.format(test_data))

from pprint import pprint
pprint(output)
