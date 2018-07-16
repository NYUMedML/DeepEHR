# logging utlities

import logging
from datetime import datetime

# define the script global logger
def ext_print (name):
	tnow = datetime.now()
	name = '['+str(tnow)+'] ' +name
	return name



#Usage:
#from utility.log import strd_logger
#from cgi import log
#log = strd_logger ('XXXX')
#log.error ('XXX')
#log.info ('XXX')

# define the script global logger
def strd_logger (name):
	log = logging.getLogger (name)
	log.setLevel (logging.INFO)
	#formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
	formatter = logging.Formatter('[%(asctime)s.%(msecs)d %(levelname)s] %(message)s','%Y-%m-%d,%H:%M:%S')
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	log.addHandler(handler)
	return log
