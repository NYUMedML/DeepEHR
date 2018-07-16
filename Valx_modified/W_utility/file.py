# set of utilities to interact with files

# @author: rm3086 (at) columbia (dot) edu

import csv, shutil, os, sys, glob, pickle
#csv.field_size_limit(sys.maxint)
from W_utility.log import strd_logger
import string

# logger
global log
log = strd_logger ('file')

# check if a file exist
def file_exist (fname):
	try:
		open(fname,'r')
		return True
	except IOError:
		return False
	

# create directory if not existing
def mkdir (dirname):
	try:
		os.makedirs(dirname)
	except OSError:
		pass
	except Exception as e:
		log.error (e)
		return False
	return True


# create directory (delete if one with the same name already exists)
def mk_new_dir (dirname):
	try:
		os.makedirs (dirname)
	except OSError:
		shutil.rmtree(dirname)
		os.makedirs (dirname)
	except Exception as e:
		log.error (e)
		return False
	return True


# copy a file from "source" to "destination"
def fcopy (source, destination):
	try:
		shutil.copy2 (source, destination)
	except Exception as e:
		log.error(e)
		return False
	return True


# return the files of a directory with extension "ext"
def flist (directory, ext):
	try:
		os.chdir(directory)
		if ext[0:2] != '*.':
			ext = '*.' + ext
		data = []
		for f in glob.glob(ext):
			data.append(f.strip())
		return data
	except Exception as e:
		log.error(e)
		return None


### read operations ###
   
# read a text file
# @param struct: save data to (1) list, (2) set, (3)string
def read_file (filename, struct = 1, logout = True):
	try:
		fid = open(filename, 'r')
		if struct == 2:
			# set
			data = set()
			for line in fid:
				if len(line) > 0:			
					data.add (line.strip())
		elif struct == 1:
			# default - list
			data = []
			for line in fid:			
				if len(line) > 0:
					data.append (line.strip())
		else:
			data = fid.read() 
							
		fid.close()
		return data
	except Exception as e:
		if logout is True:
			log.error(e)
		return None

# read a text file, break lines according to skip
# @param skip: character to skip (default ' ')
def read_file_tokenized (filename, skip = ' ', logout = True):
	try:
		data = []
		fid = open (filename, 'r')
		for line in fid:
			line = line.strip()
			data.append (line.split(skip))
		fid.close()
		return data
	except Exception as e:
		if logout is True:
			log.error(e)
		return None

# read text
def read_text (filename, logout = True):
	try:
		fid = open (filename,'r')
		data = fid.read ()
		data = data.replace ('\n',' ').replace('\t',' ')
		data = ' '.join(data.split()).strip()
		fid.close()
		return data
	except Exception as e:
		if logout is True:
			log.error(e)
		return None

# read data from a csv file    
def read_csv (filename, logout = True):
	try:
		reader = csv.reader (open(filename, "r"))
		data = []
		for r in reader:
			data.append(r)
		return data
	except Exception as e:
		if logout is True:
			log.error(e)
		return None

# read a dictionary from a csv file
# @param iKey: column to consider as key (default 0)
# @param iData: column to consider as data (default 1)
def read_csv_as_dict (filename, iKey = 0, iData = 1, logout = True):
	try:
		reader = csv.reader (open(filename, "r"))
		data = {}
		for r in reader:
			data[r[iKey].strip()] = r[iData].strip()
		return data
	except Exception as e:
		if logout is True:
			log.error(e)
		return None

# read a dictionary from a csv file (column '0' is the keys)
def read_csv_as_dict_with_multiple_items (filename, logout = True):
	try:
		reader = csv.reader (open(filename, "r"))
		data = {}
		for r in reader:
			if len(r) >= 2:
				data[r[0].strip()] = r[1:len(r)]
		return data
	except Exception as e:
		if logout is True:
			log.error(e)
		return None

# read an object (list, dictionary, set) from a serialized file
def read_obj (filename, logout = True):
	try:
		data = pickle.load (open(filename, 'rb'))
		return data
	except Exception as e:
		if logout is True:
			log.error(e)
		return None
	
	
### write operations ###

# write data in format of [(x1,y1,z1),(x2,y2,z2)] to a csv file
def write_csv (filename, data, logout = True):
	try:
		doc = csv.writer (open(filename, 'wb'), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
		for d in data:
			doc.writerow (d)
		return True			
	except Exception as e:
		if logout is True:
			log.error(e)
		return False


# write list data in format of [[x1,y1,z1],[x2,y2,z2]] to a csv file
def write_csv_list (filename, data, logout = True):
	try:
		doc = csv.writer (open(filename, 'wb'))
		doc.writerows(data)
		return True			
	except Exception as e:
		if logout is True:
			log.error(e)
		return False
	

# write data to a text file
def write_file (filename, data, logout = True):
	try:
		fid = open (filename,'w')
		for d in data:
			#fid.write('%s\n' % d.encode('utf-8'))
			fid.write('%s\n' % d)
		fid.close()
		return True
	except Exception as e:
		if logout is True:
			log.error(e)
		return False

# write text
def write_text (filename, data, logout = True):
	try:
		fid = open (filename,'w')
		fid.write('%s' % data.encode('utf-8'))
		fid.close()
		return True
	except Exception as e:
		if logout is True:
			log.error(e)
		return False

	

# write a dictionary to a csv file (first item of each row is the key)
def write_dict_to_csv (filename, data, logout = True):
	try:
		f = open(filename, 'wb')
		csv.writer(f).writerows((k,) + v for k, v in data.iteritems())
		f.close()
		return True
	except Exception as e:
		if logout is True:
			log.error(e)
		return False

# write a dictionary to a csv file 2 (first item of each row is the key)
def write_dict_to_csv_2items (filename, data):
	f = open(filename,'wb')
	w = csv.writer(f)
	for key, value in data.iteritems():
	   w.writerow([key, value[0], value[1]])	   
	f.close()
	return True


# write an object (list, set, dictionary) to a serialized file
def write_obj (filename, data, logout = True):
	try:
		pickle.dump(data, open(filename, 'wb'))
		return True
	except Exception as e:
		if logout is True:
			log.error(e)
		return False



# load input files either a specific file or a directory
# only support text files and CSV files
def load_files (fin):
	texts = []
	# judge a single file or a directory
	if fin.endswith('.txt'):
		texts = read_file (fin, 1, False)
	elif fin.endswith('.csv'):
		texts = read_csv (fin, False)
		
	else: # is a directory
		for root, dir, files in os.walk(fin):
		   for filename in files:
		     f = os.path.join(root, filename)
		     if filename.endswith('.txt'):
		     	text = read_file (f, 1, False)
		     elif filename.endswith('.csv'):
	     		text = read_csv (f, False)
		     else:
	     		continue
		     texts.extend(text)
	
	return texts



def read_settings(filename):
    """Read the content of filename and put flags and values in a
    dictionary. Each line in the file is either an empty line, a line
    starting with '#' or a attribute-value pair separated by a '='
    sign. Returns the dictionary."""
    file = open(filename, 'r')
    settings = {}
    for line in file:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        (flag, value) = map(string.strip, line.split('='))
        settings[flag] = value
    file.close()
    return settings

def write_settings(settings, filename):
    """Write a dictionary to a file, with one line per entry and with the
    key and value separated by an '=' sign."""
    os.rename(filename, filename+'.org')
    file = open(filename, 'w')
    for (flag, value) in settings.items():
        file.write("%s=%s\n" % (flag, value))
    file.close()
