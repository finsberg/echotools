import os
import echotools

curdir = os.path.abspath(os.path.dirname(__file__))


echo = echotools.EchoFile(os.path.join(curdir,'test_data.h5'))
from IPython import embed; embed()
