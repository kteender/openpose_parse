import os
import json
from itertools import groupby, chain
from operator import itemgetter
from collections import OrderedDict
from scipy import signal
import shutil
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

def setup_logging(log_dir, log_level):
	filename = os.path.join(log_dir, 'postprocessing.log')
	print('calling setup logging')
	print('filename: %s' % filename)
	logger.setLevel(log_level)
	#logging.basicConfig(filename=filename, encoding='utf-8', level=log_level)
	handler = logging.FileHandler(filename)
	formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

def test_logging():
	logger.debug('a debug')
	logger.warning('a warning')
	return

def close_logging():
	logger.handlers.clear()

def get_ease_method_names():
	return [method_name for method_name in dir(KeyPointTracker) if method_name.startswith('ease')]

def get_ease_method(method_name):
	return getattr(KeyPointTracker, method_name)

def split_files(input_directory, output_directory, group_size=4000):
    """
    Splits files in the input directory into groups of specified size and creates
    subfolders in the output directory for each group.
    
    Args:
        input_directory (str): Path to the input directory containing files.
        output_directory (str): Path to the output directory where subfolders will be created.
        group_size (int, optional): Number of files per group. Defaults to 4000.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get a list of all files in the input directory
    file_list = os.listdir(input_directory)

    # Split files into groups
    for i, file_name in enumerate(file_list):
        group_number = i // group_size
        group_folder = os.path.join(output_directory, f"Group_{group_number}")

        # Create subfolder for the group if it doesn't exist
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        # Move the file to the appropriate group subfolder
        source_path = os.path.join(input_directory, file_name)
        target_path = os.path.join(group_folder, file_name)
        shutil.move(source_path, target_path)

def split_files_into_groups():
	#this isn't gonna work on unix system!
    input_dir = "D:/_CURRENT/retirement_video/openpose_outputs/120_fps/output"
    output_dir = "D:/_CURRENT/retirement_video/openpose_outputs/120_fps_split"

    split_files(input_dir, output_dir)
    logger.info("Files have been split into groups and saved in the output directory.")


def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

#this returns a generator
def list_spans_generator(lst):
	for k, g in groupby(enumerate(lst), lambda i_x:i_x[0]-i_x[1]):
		yield list(map(itemgetter(1), g))

def angle_between(v1, v2):
	# Compute the dot product
	dot_product = np.dot(v1, v2)
	
	# Compute the magnitudes of the vectors
	mag_v1 = np.linalg.norm(v1)
	mag_v2 = np.linalg.norm(v2)
	
	# Compute the cosine of the angle
	cos_angle = dot_product / (mag_v1 * mag_v2)
	
	# Convert to angle in radians and then convert to degrees
	angle = np.arccos(np.clip(cos_angle, -1, 1))
	angle_in_degrees = np.rad2deg(angle)
	
	return angle_in_degrees

def rotate_vector_90_degrees(vector, rotate_clockwise=False):
	if rotate_clockwise:
		return np.array([vector[1], -vector[0]])
	else:
		return np.array([-vector[1], vector[0]])
	
def vector_angle(vector):
	angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
	return angle

def rot_mat_from_degree(theta, is_degrees=True):
	# Angle of rotation in radians
	if is_degrees:
		theta = np.deg2rad(theta)
	# 2D rotation matrix
	R = np.array([[np.cos(theta), -np.sin(theta)],
				[np.sin(theta),  np.cos(theta)]])
	return R


def vector_between_keypoints(dkp1, dkp2):
	logger.debug('\t\tkp1 is %s' % dkp1)
	logger.debug('\t\tkp2 is %s' % dkp2)
	p1 = np.array([dkp1.x, dkp1.y])
	p2 = np.array([dkp2.x, dkp2.y])

	return np.subtract(p2, p1)

def construct_vector(magnitude, angle_in_degrees):
    angle_in_radians = np.deg2rad(angle_in_degrees)
    return magnitude * np.array([np.cos(angle_in_radians), np.sin(angle_in_radians)])

def ensure_dir(dir):
	if not os.path.isdir(dir):
		os.makedirs(dir)
	return dir

def construct_2d_rotation_matrix(theta, is_degrees=True):
	# Angle of rotation in radians
	if is_degrees:
		theta = np.deg2rad(theta)
	# 2D rotation matrix
	R = np.array([[np.cos(theta), -np.sin(theta)],
				[np.sin(theta),  np.cos(theta)]])
	return R

class KeyPointVector(object):
	def __init__(self, x, y):
		self.np_vec = np.array([x, y])
		self.x = self.np_vec[0]
		self.y = self.np_vec[1]
		self.mag = np.linalg.norm(self.np_vec)
		self.angle = self.vector_angle()
		self.R = construct_2d_rotation_matrix(self.angle)
		return

	def vector_angle(self, convert_to_degrees=True):
		angle = np.arctan2(self.np_vec[1], self.np_vec[0])
		if convert_to_degrees:
			angle = np.rad2deg(angle)
		return angle
	
	def rotate_90_degrees(self, rotate_clockwise=False):
		vector = self.np_vec
		if rotate_clockwise:
			return np.array([vector[1], -vector[0]])
		else:
			return np.array([-vector[1], vector[0]])
	
	@staticmethod
	def angle_between(vec1, vec2, convert_to_degrees=True):
		if isinstance(vec1, KeyPointVector):
			vec1 = vec1.np_vec
		if isinstance(vec2, KeyPointVector):
			vec2 = vec2.np_vec
		# Compute the dot product
		dot_product = np.dot(vec1, vec2)
		
		# Compute the magnitudes of the vectors
		mag_v1 = np.linalg.norm(vec1)
		mag_v2 = np.linalg.norm(vec2)
		
		# Compute the cosine of the angle
		cos_angle = dot_product / (mag_v1 * mag_v2)
		
		# Convert to angle in radians and then convert to degrees
		angle = np.arccos(np.clip(cos_angle, -1, 1))
		if convert_to_degrees:
			angle = np.rad2deg(angle)
		return angle

	@classmethod
	def construct_between_keypoints(cls, kp1, kp2):
		p1 = np.array([kp1.x, kp1.y])
		p2 = np.array([kp2.x, kp2.y])
		vec = np.subtract(p2, p1)
		return cls(vec[0], vec[1])
	
	@classmethod
	def construct_from_mag_dir(cls, magnitude, angle, is_degrees=True):
		if is_degrees:
			angle = np.deg2rad(angle)
		vec = magnitude * np.array([np.cos(angle), np.sin(angle)])
		return cls(vec[0], vec[1])
	
	def __repr__(self):
		return('Key Point Vector %s %s' % (self.x, self.y))
	
#IF YOU EVER COMBINE THEM, THIS IS OVERLOADED WITH SOME STUFF IN NODE
class KeyPoint(object):
	def __init__(self, x, y, c, num, source=""):
		self.x = x
		self.y = y
		self.c = c
		self.num = num
		self.source = source
		return
	
	def __repr__(self):
		return ('%s key point no. %s x: %s y: %s c: %s' % (self.source, self.num, self.x, self.y, self.c))

class DetectedKeyPoint(KeyPoint):
	
	def __init__(self, x, y, c, num):
		super(DetectedKeyPoint, self).__init__(x, y, c, num, source="detected")

		return
	
#consider not using pointID at all
#This class is for single-frame key-point inferences
class ConstructedKeyPoint(KeyPoint):
	def __init__(self, x, y, c, num, source="constructed"):
		#self.tk_points = tracked_key_points
		#self.num_tk_points = len(tracked_key_points)
		#self.tk_nums = [tkp[0].num for tkp in tracked_key_points]
		#self.point_id = self.point_id()
		super(ConstructedKeyPoint, self).__init__(x, y, c, num, source=source)

	@classmethod
	def construct_from_multipoint_average(cls, contributors, num, conf_val=0.1):
		num_intended_contributors = len(contributors)
		usable_contributors = [cont for cont in contributors if cont.c >= conf_val]
		num_usable_contributors = len(usable_contributors)

		if float(num_usable_contributors)/float(num_intended_contributors) >= 0.5:
			avg_x = sum([cont.x for cont in usable_contributors])/num_usable_contributors
			avg_y = sum([cont.y for cont in usable_contributors])/num_usable_contributors
			#reconsider this? Your idea here is that you'll last confidence fill afterwards
			conf = 1.0
		else:
			avg_x = 0.0
			avg_y = 0.0
			conf = 0.0

		return cls(avg_x, avg_y, conf, num, source="average")
	
	@classmethod
	def construct_from_vector(cls, p1, p2, theta, mag_percent, magn, num, reflector=None, reflector_conf=0.1):
		vec = vector_between_keypoints(p1, p2)
		print('\tConstructed a vector between %s and %s: %s' % (p1.num, p2.num, vec))
		
		mag = np.linalg.norm(vec)
		print('\tMagnitude is %s' % mag)

		vec_norm = vec/mag
		ang = vector_angle(vec_norm)
		print('\tSource vector angle is %s' % ang)
		#R = rot_mat_from_degree(ang) 

		#negative rotation dir is cw, positive rotation dir is ccw
		if reflector != None and reflector.c >= reflector_conf:
			print('\tUsing reflector %s' % reflector)
			refl_vec = vector_between_keypoints(p2, reflector)
			print('\tReflector vec is %s' % refl_vec)
			refl_vec_mag = np.linalg.norm(refl_vec)
			refl_vec_norm = refl_vec/refl_vec_mag
			refl_ang = vector_angle(refl_vec_norm)

			refl_R = rot_mat_from_degree(refl_ang)

			# refl_ang =  + ang
			# refl_vec_compare = construct_vector(1.0, refl_ang)

			print('\treflection angle is %s' % refl_ang)

			v_ccw = rotate_vector_90_degrees(vec_norm)
			if np.dot(v_ccw, refl_vec_norm) >= 0:
				print('After rotating %s 90 degrees ccw, dp >= 0')
				theta = np.abs(theta) * -1
			else:
				theta = np.abs(theta) * 1

		new_mag = mag*mag_percent
		new_mag = magn
		print('\tFinal angle is %s' % theta)

		R = rot_mat_from_degree(theta)
		new_vec = np.dot(R, vec_norm) * 1.0

		#new_vec = construct_vector(new_mag, theta)
		print('\tFinal vector is %s' % new_vec)
		scaled_vec = new_vec *new_mag
		print('\tFinal vector scaled is %s' % scaled_vec)

		final_point = np.array([p2.x+scaled_vec[0], p2.y+scaled_vec[1]])

		if reflector != None and reflector.c >= reflector_conf:
			print('Reflector check')
			print(np.dot(scaled_vec, refl_vec) < 0)

		return cls(float(final_point[0]), float(final_point[1]), 1.0, num, source="vector")
		#vec_R = 

		#new_vec = 


	
	@classmethod
	def construct_from_vector_old(cls, p1, p2, theta, mag_percent, num, reflector=None, reflector_conf=0.1):
		vec = vector_between_keypoints(p1, p2)
		print('\tConstructed a vector between %s and %s: %s' % (p1.num, p2.num, vec))
		
		mag = np.linalg.norm(vec)

		print('\tMagnitude is %s' % mag)

		new_mag = mag*mag_percent

		#create a transformation matrix where the x-axis is the vector and the y-axis is orthonormal
		basis_x = vec/mag
		basis_y = rotate_vector_90_degrees(basis_x)
		print('\tbasis_x is %s' % basis_x)
		print('\tbasis_y is %s' % basis_y)
		e = [[basis_x[0], basis_y[0]], [basis_x[1], basis_y[1]]]
		R = np.array(e)

		#construct the vector from the angle and magnitude
		new_vec = construct_vector(1.0, theta)

		if reflector != None and reflector.c >= reflector_conf:
			print('\tUsing reflector: %s' % reflector)
			#get the opposite in R space
			reflector_vec = vector_between_keypoints(p2, reflector)
			reflector_vec_mag = np.linalg.norm(reflector_vec)
			reflector_vec_norm = reflector_vec/reflector_vec_mag
			refector_vec_transf = np.dot(R, reflector_vec_norm)

			print('\tReflector in R space is %s' % refector_vec_transf)

			#if the y-signs are the same, flip the new vector over x-axis
			if (refector_vec_transf[1] >= 0) and (new_vec[1] >= 0):
				print('\tFlipping new vector %s over x-axis' % new_vec)
				new_vec_flipped = np.array([new_vec[0], -new_vec[1]])
				new_vec = new_vec_flipped

			#regardless we flip the x coordinate because we want to be pointing away from 
			#new_vec = np.array([new_vec[0]*-1, new_vec[1]])

		#transform new_vec back into worldspace
		R_inv = np.linalg.inv(R)
		final_vec = np.dot(new_vec, R) * new_mag

		print("\tFinal constructed vec is: %s" % final_vec)


		final_point = np.array([p2.x+final_vec[0], p2.y+final_vec[1]])

		print("\tFinal point is: %s" % final_point)

		#we are happy with this point, so confidence is 1.0
		return cls(float(final_point[0]), float(final_point[1]), 1.0, num, source="vector")

"""
	# def construct_from_average_old(self):
	# 	constructed_point = []
	# 	#this is the number of key points you have
	# 	for i in range(0, len(self.tk_points[0])):
	# 		contributors = [tk_point[i] for tk_point in self.tk_points if tk_point[i].c > 0.01]
	# 		num_contributors = len(contributors)
	# 		print('For frame %s, between points %s, %s usable points found for average' % (i, self.tk_nums, num_contributors))

	# 		if num_contributors > 0:
	# 			avg_x = sum([cont.x for cont in contributors])/num_contributors
	# 			avg_y = sum([cont.y for cont in contributors])/num_contributors
	# 			#reconsider this? Your idea here is that you'll last confidence fill afterwards
	# 			conf = 1.0
	# 		else:
	# 			avg_x = 0.0
	# 			avg_y = 0.0
	# 			conf = 0.0

	# 		constructed_point.append(KeyPoint(avg_x,avg_y,conf,self.num,source="average"))
	# 	return constructed_point
"""
class KeyPointConstructor(object):

	def __init__(self, num_key_points, num, point_type=""):
		logger.info('Creating a constructor for kp number %s for %s key points' % (num, num_key_points))
		self.num_key_points = num_key_points
		self.kp_num = num
		self.point_type = point_type
		self.constructed_key_points = []

	#this doesn't use any of the filling stuff, this is straight constructing data
	def vec_replace(self, tracker1, tracker2, scalar, theta):
		logger.info('Running vector replace for Key Point Constructor %s' % self)
		replaced = []
		tkp1 = tracker1.tracked_key_points
		tkp2 = tracker2.tracked_key_points
		if (len(tkp1) != len(tkp2)) or (len(tkp1) != self.num_key_points):
			e = 'Inequal number of key points for vector replace:'
			e += "\ntracker1: %s" % len(tkp1)
			e += "\ntracker2: %s" % len(tkp2)
			e += "\nkey point constructor: %s" % self.num_key_points
			logger.error(e)
			raise Exception(e)
		
		else:

			for kp1, kp2 in zip(tkp1, tkp2):
				vec1 = KeyPointVector.construct_between_keypoints(kp1, kp2)

				R = construct_2d_rotation_matrix(theta)
				vec2 = R.dot(vec1.np_vec) * scalar

				x = (kp1.x+vec1.x+vec2[0])
				y = (kp1.y+vec1.y+vec2[1])
				logger.debug('Constructed vector ( %s , %s )' % (x, y))

				kp = KeyPoint(x, y, 1.0, self.kp_num, source='vector_replace')
				replaced.append(kp)

			self.constructed_key_points = replaced
		return replaced
	
	def dummy_replace(self):
		logger.info('Running dummy replace for Key Point Constructor %s' % self)
		replaced = [KeyPoint(0.0, 0.0, 1.0, self.kp_num, source='dummy') for i in range(0, self.num_key_points)]
		self.constructed_key_points = replaced
		return replaced
	
	def write_source_info(self, outdir):
		write_dir = ensure_dir(outdir)
		write_path = os.path.join(write_dir, str(self.kp_num) + '.json')
		d = {i:self.constructed_key_points[i].source for i in range(0,len(self.constructed_key_points))}
		write_str = json.dumps(d, indent=4)
		f = open(write_path, 'w')
		f.write(write_str)
		f.close()
		return
	
	def __repr__(self):
		return 'Key Point Constructor for point %s. %s points long. Construction done: %s' % (self.kp_num, self.num_key_points, False == (self.constructed_key_points == []))


#This class is for multi-frame key point inferences
class KeyPointTracker(object):

	def __init__(self, key_points, num, point_type=""):
		logger.info('Creating a tracker for kp number %s out of %s detected key points' % (num, len(key_points)))
		self.key_points = key_points
		self.num_key_points = len(key_points)
		self.kp_num = num
		self.tracked_key_points = self.key_points
		self.point_type = point_type

	@property
	def tracked_np_vec(self):
		return np.array([(kp.x, kp.y) for kp in self.tracked_key_points])
	
	@property
	def tracked_confidence_vec(self):
		return np.array([(kp.c) for kp in self.tracked_key_points])

	def fill_executor(self, usable_indicies, start_operation_cb, tween_operation_cb, span_operation_cb, stop_operation_cb, min_span, **kwargs):
		filled = []
		logger.info('\tFill executor got %s usable indicies' % len(usable_indicies))
		logger.debug('\tUsable indicies: %s' % usable_indicies)

		#create a generator with spans
		usable_idx_spans = list(list_spans_generator(usable_indicies))

		usable_idx_spans = [span for span in usable_idx_spans if len(span) >= min_span ]

		#filled_confidences = []
		num_idx_spans = len(usable_idx_spans)

		for i in range(0, num_idx_spans):
			span = usable_idx_spans[i]
			start_idx = span[0]
			stop_idx = span[-1]

			if i == 0:
				filled += start_operation_cb(start_idx, **kwargs)

			elif num_idx_spans > 1:
				prev_span = usable_idx_spans[i-1]

				fill_indicies = list(range(prev_span[-1]+1, start_idx))

				filled += tween_operation_cb(prev_span, start_idx, fill_indicies, **kwargs)

			filled += span_operation_cb(start_idx, stop_idx, **kwargs)

			if i == num_idx_spans-1:
				if stop_idx < (self.num_key_points - 1):
					filled += stop_operation_cb(stop_idx, **kwargs)

		return filled
	
	def last_confidence_kp_fill(self, confidence_val, min_span=2, tween_func=None, start_copy=False):
		confidences = [kp.c for kp in self.key_points]
		usable_indices = [i for i,conf in enumerate(confidences) if conf >= confidence_val]
		d = {'tween_fun':tween_func, 'start_copy':start_copy}
		filled_key_points = self.fill_executor(usable_indices, self.kp_fill_start_cb, self.kp_fill_tween_cb, self.kp_fill_span_cb, self.kp_fill_stop_cb, min_span, **d)
		if filled_key_points != []:
			logger.info('Was able to performan last confidence kp fill')
			self.tracked_key_points = filled_key_points
		else:
			logger.warning('Was not able to perform last confidence kp fill. Reverting to raw data')
			self.tracked_key_points = self.key_points
		return filled_key_points
	
	def last_confidence_relative_kp_fill(self, confidence_val, other_tracker, min_span=3, tween_func=None, start_copy=False):
		confidences = [kp.c for kp in self.key_points]
		usable_indices = [i for i,conf in enumerate(confidences) if conf >= confidence_val]
		d = {'tween_fun':tween_func, 'start_copy':start_copy, 'other_tracker':other_tracker}
		filled_key_points = self.fill_executor(usable_indices, self.kp_fill_relative_start_cb, self.kp_fill_relative_tween_cb, self.kp_fill_relative_span_cb, self.kp_fill_relative_stop_cb, min_span, **d)
		self.tracked_key_points = filled_key_points
		return filled_key_points		
	
	
	def other_tracker_kp_fill(self, confidence_val, other_tracker, min_span=3, tween_func=None):
		confidences = [kp.c for kp in self.key_points]
		usable_indices = [i for i,conf in enumerate(confidences) if conf >= confidence_val]
		d = {'tween_fun':tween_func, 'other_tracker':other_tracker}
		filled_key_points = self.fill_executor(usable_indices, self.other_tracker_copy_start_cb, self.other_tracker_copy_tween_cb, self.other_tracker_copy_span_cb, self.other_tracker_copy_stop_cb, min_span, **d)
		self.tracked_key_points = filled_key_points
		return filled_key_points
	
	def last_confidence_vec_fill(self, confidence_val, other_tracker, min_span=3, tween_func=None):
		""" Create list of vectors between two points, with averaging in between """
		logger.info('Running last confidence vector fill-confidence: %s, min_span: %s' % (confidence_val, min_span))
		this_conf = [kp.c for kp in self.tracked_key_points]
		other_conf = [kp.c for kp in other_tracker.tracked_key_points]
		this_usable_indices = [i for i,conf in enumerate(this_conf) if conf >= confidence_val]
		other_usable_indices = [i for i,conf in enumerate(other_conf) if conf >= confidence_val]

		usable_indices = sorted(list(set(this_usable_indices).intersection(set(other_usable_indices))))
		logger.debug('Usable indicies were %s' % usable_indices)
		d = {'other_tracker':other_tracker, 'tween_func':tween_func}
		filled_vectors = self.fill_executor(usable_indices, self.vec_fill_start_cb, self.vec_fill_tween_cb, self.vec_fill_span_cb, self.vec_fill_stop_cb, min_span, **d)

		filled_points = []
		for i in range(0, self.num_key_points):
			kp1 = other_tracker.tracked_key_points[i]
			kpv = filled_vectors[i]
			global_vec = np.sum([np.array([kp1.x, kp1.y]), kpv.np_vec], axis=0)
			if i in usable_indices:
				c = this_conf[i]
			else:
				c = 0.0
			filled_points.append(KeyPoint(global_vec[0], global_vec[1], c, self.kp_num, source='vector_fill'))
		self.tracked_key_points = filled_points
		return filled_points
	

	def kp_fill_start_cb(self, start_idx, **kwargs):
		kp = self.key_points[start_idx]
		return [KeyPoint(kp.x, kp.y, 1.0, self.kp_num, source='filled_start') for i in range(0, start_idx)]
	
	def kp_fill_tween_cb(self, prev_span, start_idx, fill_indicies, **kwargs):
		tween_func = kwargs.get('tween_func', None)
		if tween_func == None:
			tween_func = KeyPointTracker.ease_linear
		start_copy = kwargs.get('start_copy', False)

		logger.debug('\t\tUsing %s as tween func' % tween_func)
		logger.debug('\t\tDoing start copy: %s' % start_copy)

		filled_kps = []

		if start_copy:
			x = self.key_points[prev_span[-1]].x
			y = self.key_points[prev_span[-1]].y
			for i in range(1, len(fill_indicies)+1):
				filled_kps.append(KeyPoint(x, y, 0.0, self.kp_num, source='filled_copy'))
		else:
			div = len(fill_indicies)+1
			
			x_start = self.key_points[prev_span[-1]].x
			x_stop = self.key_points[start_idx].x

			y_start = self.key_points[prev_span[-1]].y
			y_stop = self.key_points[start_idx].y

			x_interval = tween_func((x_stop - x_start)/div)
			y_interval = tween_func((y_stop - y_start)/div)

			filled_kps = []
			for i in range(1, len(fill_indicies)+1):
				interval = tween_func(i/div)
				x_fill = x_start + (x_stop - x_start)*interval
				y_fill = y_start + (y_stop - y_start)*interval
				filled_kps.append(KeyPoint(x_fill, y_fill, 0.0, self.kp_num, source='filled_average'))
		return filled_kps

		# return [KeyPoint(x_start+(i*x_interval), y_start+(i*y_interval), 1.0, self.kp_num, source='filled_average')
		# 			  				for i in range(1, len(fill_indicies)+1)]
	
	def kp_fill_span_cb(self, start_idx, stop_idx, **kwargs):
		return [self.key_points[i] for i in range(start_idx, stop_idx+1)]
	
	def kp_fill_stop_cb(self, stop_idx, **kwargs):
		kp = self.key_points[stop_idx]
		return [KeyPoint(kp.x, kp.y, 1.0, self.kp_num, source='filled_stop') 
						for i in range(stop_idx+1, self.num_key_points)]

	def kp_fill_relative_start_cb(self, start_idx, **kwargs):
		kp = self.key_points[start_idx]
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise Exception(e)
		other_kp = other_tracker.tracked_key_points[start_idx]

		vec = KeyPointVector.construct_between_keypoints(other_kp, kp)
		filled_kps = []
		for i in range(0, start_idx):
			ikp = other_tracker.tracked_key_points[i]
			filled_kps.append(KeyPoint(ikp.x+vec.x, ikp.y+vec.y, 1.0, self.kp_num, source='relative_filled_start'))
		return filled_kps
	
	def kp_fill_relative_tween_cb(self, prev_span, start_idx, fill_indicies, **kwargs):
		tween_func = kwargs.get('tween_func', None)
		if tween_func == None:
			tween_func = KeyPointTracker.ease_linear
		start_copy = kwargs.get('start_copy', False)
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise Exception(e)
		
		filled_kps = []
		if start_copy:
			x = self.key_points[prev_span[-1]].x
			y = self.key_points[prev_span[-1]].y
			kp = self.key_points[prev_span[-1]]
			other_kp = other_tracker.tracked_key_points[prev_span[-1]]
			vec = KeyPointVector.construct_between_keypoints(other_kp, kp)
			for i in range(1, len(fill_indicies)+1):
				idx = i + prev_span[-1]
				ikp = other_tracker.tracked_key_points[idx]
				filled_kps.append(KeyPoint(ikp.x+vec.x, ikp.y+vec.y, 1.0, self.kp_num, source='relative_filled_copy'))
		else:
			kp1_start = other_tracker.key_points[prev_span[-1]]
			kp2_start = self.key_points[prev_span[-1]]
			kpv_start = KeyPointVector.construct_between_keypoints(kp1_start, kp2_start)

			kp1_stop = other_tracker.tracked_key_points[start_idx]
			kp2_stop = self.tracked_key_points[start_idx]
			kpv_stop =  KeyPointVector.construct_between_keypoints(kp1_stop, kp2_stop)	
			
			#implement this later katie!
			pass
		return filled_kps
	
	def kp_fill_relative_span_cb(self, start_idx, stop_idx, **kwargs):
		return [self.key_points[i] for i in range(start_idx, stop_idx+1)]
	
	def kp_fill_relative_stop_cb(self, stop_idx, **kwargs):
		kp = self.key_points[stop_idx]
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise Exception(e)
		other_kp = other_tracker.tracked_key_points[stop_idx]

		vec = KeyPointVector.construct_between_keypoints(other_kp, kp)
		filled_kps = []
		for i in range(stop_idx+1, self.num_key_points):
			ikp = other_tracker.tracked_key_points[i]
			filled_kps.append(KeyPoint(ikp.x+vec.x, ikp.y+vec.y, 1.0, self.kp_num, source='relative_filled_stop'))
		return filled_kps

	def vec_fill_start_cb(self, start_idx, **kwargs):
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise(e)
		
		kp1 = other_tracker.tracked_key_points[start_idx]
		kp2 = self.tracked_key_points[start_idx]

		kpv = KeyPointVector.construct_between_keypoints(kp1, kp2)

		return [KeyPointVector(kpv.x, kpv.y) for i in range(0, start_idx)]

	def vec_fill_tween_cb(self, prev_span, start_idx, fill_indicies, **kwargs):
		other_tracker = kwargs.get('other_tracker', None)
		tween_func = kwargs.get('tween_func', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise Exception(e)
		
		if tween_func == None:
			tween_func = KeyPointTracker.ease_linear

		logger.debug('\t\tUsing %s as tween func' % tween_func)
		
		div = len(fill_indicies)+1
		
		kp1_start = other_tracker.tracked_key_points[prev_span[-1]]
		kp2_start = self.tracked_key_points[prev_span[-1]]
		kpv_start =  KeyPointVector.construct_between_keypoints(kp1_start, kp2_start)

		kp1_stop = other_tracker.tracked_key_points[start_idx]
		kp2_stop = self.tracked_key_points[start_idx]
		kpv_stop =  KeyPointVector.construct_between_keypoints(kp1_stop, kp2_stop)		

		#angle_start = kpv_start.vector_angle()
		#angle_stop = kpv_stop.vector_angle()
		#angle_interval = (angle_stop - angle_start)/div

		mag_start = kpv_start.mag
		mag_stop = kpv_stop.mag
		mag_interval = (mag_stop - mag_start)/div

		tweened_vectors = []
		for i in range(1, len(fill_indicies)+1):
			#a = angle_start+(i*angle_interval)
			interval_norm = tween_func(i/div)
			v = slerp(kpv_start.np_vec/np.linalg.norm(kpv_start.np_vec), kpv_stop.np_vec/np.linalg.norm(kpv_stop.np_vec), interval_norm)
			#m = mag_start+(i*mag_interval)
			#m = mag_start + (interval_norm*mag_interval)
			m = mag_start + (mag_stop - mag_start)*interval_norm
			v = v*m
			tweened_vectors.append(KeyPointVector(v[0],v[1]))
			#m = mag_start+(i*mag_interval)
			#tweened_vectors.append(KeyPointVector.construct_from_mag_dir(m, a))

		return tweened_vectors
	
	def vec_fill_span_cb(self, start_idx, stop_idx, **kwargs):
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise Exception(e)
		
		span_vectors = []
		for i in range(start_idx, stop_idx+1):
			kp1 = other_tracker.tracked_key_points[i]
			kp2 = self.tracked_key_points[i]
			kpv = KeyPointVector.construct_between_keypoints(kp1, kp2)

			span_vectors.append(kpv)

		return span_vectors
	
	def vec_fill_stop_cb(self, stop_idx, **kwargs):
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise(e)
		
		kp1 = other_tracker.tracked_key_points[stop_idx]
		kp2 = self.tracked_key_points[stop_idx]

		kpv = KeyPointVector.construct_between_keypoints(kp1, kp2)

		return [KeyPointVector(kpv.x, kpv.y) for i in range(stop_idx+1, self.num_key_points)]
	
	##YOU STOPPED HERE
	def other_tracker_copy_start_cb(self, start_idx, **kwargs):
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise(e)
		other_kp = other_tracker.tracked_key_points[start_idx]

		filled_kps = []
		for i in range(0, start_idx):
			okp = other_tracker.tracked_key_points[i]
			filled_kps.append(KeyPoint(okp.x, okp.y, 0.0, self.kp_num, source='other_tracker_copy_start'))
			#filled_kps.append(KeyPoint(ikp.x+vec.x, ikp.y+vec.y, 1.0, self.kp_num, source='relative_filled_start'))
		return filled_kps
	
	def other_tracker_copy_tween_cb(self, prev_span, start_idx, fill_indicies, **kwargs):
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise(e)
		other_kp = other_tracker.tracked_key_points[start_idx]
		filled_kps = []
		for i in range(1, len(fill_indicies)+1):
			idx = i + prev_span[-1]
			okp = other_tracker.tracked_key_points[idx]
			filled_kps.append(KeyPoint(okp.x, okp.y, 0.0, self.kp_num, source="other_tracker_copy_tween"))
		return filled_kps
	
	def other_tracker_copy_span_cb(self, start_idx, stop_idx, **kwargs):
		return [self.key_points[i] for i in range(start_idx, stop_idx+1)]
	
	def other_tracker_copy_stop_cb(self, stop_idx, **kwargs):
		kp = self.key_points[stop_idx]
		other_tracker = kwargs.get('other_tracker', None)
		if not isinstance(other_tracker, KeyPointTracker):
			e = '%s is not another KeyPointTracker!' % type(other_tracker)
			raise(e)
		other_kp = other_tracker.tracked_key_points[stop_idx]

		filled_kps = []
		for i in range(stop_idx+1, self.num_key_points):
			okp = other_tracker.tracked_key_points[i]
			filled_kps.append(KeyPoint(okp.x, okp.y, 1.0, self.kp_num, source='other_tracker_copy_stop'))
		return filled_kps

	def run_savgol_filter(self, window=None, order=3):
		if window == None:
			window = int(len(self.tracked_key_points) * 0.05)
		logger.debug('\tCreating smoothing for %s' % self)
		tracked_np_vec = self.tracked_np_vec
		logger.debug('\tWindow: %s' % window)
		logger.debug('\tOrder: %s' % order)
		logger.debug('\tTracked np vec:\n\t\t')
		logger.debug(tracked_np_vec)
		smoothed_np_vec = signal.savgol_filter(tracked_np_vec, window, order, axis=0)
		logger.debug('\tSmoothed np vec:\n\t\t')
		logger.debug(smoothed_np_vec)
		smoothed_key_points = []
		for i in range(0, len(self.tracked_key_points)):
			existing = self.tracked_key_points[i]
			np_arr = smoothed_np_vec[i]
			new = KeyPoint(np_arr[0], np_arr[1], existing.c, existing.num, source= existing.source+'_smooth')
			smoothed_key_points.append(new)
		self.tracked_key_points = smoothed_key_points
		return
	
	def run_savgol_filter_confidences(self, window=None, order=3):
		if window == None:
			window = int(len(self.tracked_key_points) * 0.05)
		tracked_conf_vec = self.tracked_confidence_vec
		smoothed_conf_vec = signal.savgol_filter(tracked_conf_vec, window, order, axis=0)
		key_points = []
		for i in range(0, len(self.tracked_key_points)):
			existing = self.tracked_key_points[i]
			np_arr = smoothed_conf_vec[i]
			new = KeyPoint(existing.x, existing.y, np_arr, existing.num, source= existing.source+'_conf_smooth')
			key_points.append(new)
		self.tracked_key_points = key_points

	def write_source_info(self, outdir):
		write_dir = ensure_dir(outdir)
		write_path = os.path.join(write_dir, str(self.kp_num) + '.json')
		d = {i:self.tracked_key_points[i].source for i in range(0,len(self.tracked_key_points))}
		write_str = json.dumps(d, indent=4)
		f = open(write_path, 'w')
		f.write(write_str)
		f.close()
		return
	
	#https://gist.github.com/robweychert/7efa6a5f762207245646b16f29dd6671#easeincubic
	@staticmethod
	def ease_in_quad(t):
		return t*t
	
	@staticmethod
	def ease_in_quart(t):
		return t*t*t*t
	
	@staticmethod
	def ease_in_expo(t):
		return math.pow(2, 10 * (t - 1))
	
	@staticmethod
	def ease_linear(t):
		return t

	def __repr__(self):
		return 'Key Point Tracker for point %s. %s points long. Additional tracking done: %s' % (self.kp_num, self.num_key_points, False == (self.key_points == self.tracked_key_points))

class OpenPoseFrame(object):
	def __init__(self, filepath):
		self.filepath = filepath

		self.frame_data = None
	
	#you would need to rewrite this if you wanted to do multi-person
	#and buff it out in general, because right now this is just getting the first person
	#helps to run openpose with the only one person argument
	def read(self, write=True):

		f = open(self.filepath, 'r')
		self.frame_data = json.loads(f.read())
		f.close()

		person_index = 0
		try:
			person = self.frame_data['people'][person_index]
		except IndexError:
			self.frame_data = None
			return {}

		#this has the pose info for the body points
		pk_2d = person['pose_keypoints_2d']
		#each body point has 3 coordinates
		n = 3

		kd = {}

		key_points = [pk_2d[i * n:(i + 1) * n] for i in range((len(pk_2d) + n - 1) // n )]
		for i in range(0, len(key_points)):
			key_point = key_points[i]

			x,y,c = key_point[0], key_point[1], key_point[2]
			kd[i] = DetectedKeyPoint(x,y,c,i)

		pk_2d = person['pose_keypoints_2d']
		hl_2d = person['hand_left_keypoints_2d']
		hr_2d = person['hand_right_keypoints_2d']

		n = 3
		kd = {}

		p_key_points = [pk_2d[i * n:(i + 1) * n] for i in range((len(pk_2d) + n - 1) // n )]
		hl_key_points = [hl_2d[i * n:(i + 1) * n] for i in range((len(hl_2d) + n - 1) // n )]
		hr_key_points = [hr_2d[i * n:(i + 1) * n] for i in range((len(hr_2d) + n - 1) // n )]
		p_dict = {}
		hl_dict = {}
		hr_dict = {}

		for i in range(0, len(p_key_points)):
			p_key_point = p_key_points[i]
			p_x,p_y,p_c = p_key_point[0], p_key_point[1], p_key_point[2]
			#we want to get rid of whatever c value Open Pose found
			#any sort of confidence-based processing should occur in the postprocessing script, not in the node
			p_dict[i] = DetectedKeyPoint(p_x,p_y,p_c,i)

		#hand and body have different number of key points, so we need to do loop
		for i in range(0, len(hl_key_points)):
			hl_key_point = hl_key_points[i]
			hl_x, hl_y, hl_c = hl_key_point[0], hl_key_point[1], hl_key_point[2]
			hl_dict[i] = DetectedKeyPoint(hl_x, hl_y, hl_c, i)

			hr_key_point = hr_key_points[i]
			hr_x, hr_y, hr_c = hr_key_point[0], hr_key_point[1], hr_key_point[2]
			hr_dict[i] = DetectedKeyPoint(hr_x, hr_y, hr_c, i)	

		kd = {
			'pose_keypoints_2d':p_dict,
			'hand_left_keypoints_2d':hl_dict,
			'hand_right_keypoints_2d':hr_dict,
		}


		
		#we return a dictionary of the BODY_25 point number to the point found by OpenPose
		#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
		return kd
	
	#you would need to rewrite this if you wanted to do multiperson
	def write_pose_data(self, write_dir, new_pose_points, new_hl_points, new_hr_points, hl_rest_points, hr_rest_points):
		concat_pose = list(chain(*[[np.x, np.y, np.c] for np in new_pose_points]))
		concat_hl = list(chain(*[[np.x, np.y, np.c] for np in new_hl_points]))
		concat_hr = list(chain(*[[np.x, np.y, np.c] for np in new_hr_points]))

		concat_hl_rest = list(chain(*[[np.x, np.y, np.c] for np in hl_rest_points]))
		concat_hr_rest = list(chain(*[[np.x, np.y, np.c] for np in hr_rest_points]))

		out_data = self.frame_data
		out_data["people"][0]['pose_keypoints_2d'] = concat_pose
		out_data["people"][0]['hand_left_keypoints_2d'] = concat_hl
		out_data["people"][0]['hand_right_keypoints_2d'] = concat_hr
		out_data["people"][0]['hand_left_keypoints_2d_rest'] = concat_hl_rest
		out_data["people"][0]['hand_right_keypoints_2d_rest'] = concat_hr_rest

		fileroot = os.path.basename(self.filepath).split('.')[0]
		ensure_dir(write_dir)
		outpath = os.path.join(write_dir, 'processed_'+fileroot+'.json')
		f = open(outpath, 'w')
		f.write(json.dumps(out_data))
		f.close()
		return
	
class OpenPoseDataProcessor(object):

	def __init__(self, dir, **kwargs):
		self.dir = dir
		self.files = [os.path.join(dir, filename) for filename in sorted(os.listdir(self.dir))]
		#remember that this returns dict of form point number:DetectedKeyPoint
		self.frames = [OpenPoseFrame(filepath) for filepath in self.files]
		frame_data = [frame.read() for frame in self.frames]
		self.pose_frame_data = [data['pose_keypoints_2d'] for data in frame_data if data != {}]
		self.hl_frame_data = [data['hand_left_keypoints_2d'] for data in frame_data if data != {}]
		self.hr_frame_data = [data['hand_right_keypoints_2d'] for data in frame_data if data != {}]

		#THIS SHOULD ALLOW YOU TO SKIP PROCESSING SOME BY SETTING PROCESSED TO RAW. BUT NOT TESTED
		self.per_frame_pose_processed = self.pose_frame_data
		self.per_frame_hl_processed = self.hl_frame_data
		self.per_frame_hr_processed = self.hr_frame_data
		self.per_frame_hl_rest_processed = self.hl_frame_data
		self.per_frame_hr_rest_processed = self.hr_frame_data

		self.trackers = {'pose':{},'l_hand':{},'r_hand':{},'l_hand_rest':{},'r_hand_rest':{}}

		kwarg_defaults = {
			'process_hands': True,
			'pose_smooth': True,
			'hand_smooth': True,
			'pose_smooth_window': 3,
			'hand_smooth_window': 8,
			'pose_tween': KeyPointTracker.ease_linear,
			'hand_tween': KeyPointTracker.ease_linear,
			'pose_confidence': 0.7,
			'hand_confidence': 0.3,
			'pose_min_span': 2,
			'hand_min_span': 2,
		}

		kwarg_keys = kwargs.keys()
		for k in kwarg_defaults.keys():
			if k in kwarg_keys:
				logger.info('setting kwarg %s to %s from input' % (k, kwargs[k]))
				setattr(self, k, kwargs[k])
			else:
				logger.info('setting kwarg %s to %s from default' % (k, kwarg_defaults[k]))
				setattr(self, k, kwarg_defaults[k])

		self.kwarg_defaults = kwarg_defaults

		return
	
	def process(self, outdir="", write=True):
		logger.info(('****Processing Pose Data******'))
		self.process_pose_data()
		if self.process_hands:
			self.process_both_hand_data()
			self.create_both_hand_rest_data()
		if write:
			logger.info('Writing processed data to directory: %s' % outdir)
			self.write_processed_data(outdir=outdir)
		logger.debug('info was')
		for k in self.kwarg_defaults.keys():
			logger.debug(k, ' : ', getattr(self,k))
	
	#REMEMBER THAT EACH POINT HAS DATA FOR EACH FRAME! 
	#NEED TO MAKE SURE DON'T LOSE POINTS
	def process_pose_data(self):

		combined_key_points = {}

		for d in self.pose_frame_data:
			#this should always be 25
			key_points = len(d.keys())

			for i in range(0, key_points):
				if not i in combined_key_points.keys():
					combined_key_points[i] = [d[i]]
				else:
					combined_key_points[i] = combined_key_points[i] + [d[i]]

		processed_key_points = {}

		for k, v in combined_key_points.items():
			tracker = KeyPointTracker(v, k, point_type='pose')
			tracker.last_confidence_kp_fill(self.pose_confidence, min_span=self.pose_min_span, tween_func=self.pose_tween)
			if self.pose_smooth and len(tracker.tracked_np_vec >= self.pose_smooth_window):
				tracker.run_savgol_filter(window=self.pose_smooth_window, order=2)
			logger.critical('tracker.tracked_key_points %s' % len(tracker.tracked_key_points ))
			processed_key_points[k] = tracker.tracked_key_points
			self.trackers['pose'][k] = tracker

		#if the lists in processed_key_point are not all the same length, the zip won't work and this list will be empty/None
		self.per_frame_pose_processed = list(zip(*processed_key_points.values()))

	#POSE PROCESSED NEEDS TO BE DONE BEFORE HAND PROCESSED
	def process_both_hand_data(self):
		logger.info('****Processing Right Hand Data******')
		self.process_single_hand_data()
		logger.info('****Processing Left Hand Data******')
		self.process_single_hand_data(left_hand=True)

	def process_single_hand_data(self, left_hand=False):
		#Katie is right handed so this func defaults to right hand
		hand_frame_data = self.hr_frame_data
		point_type = 'r_hand'
		if left_hand:
			point_type = 'l_hand'
			hand_frame_data = self.hl_frame_data

		combined_key_points = {}

		for d in hand_frame_data:
			key_points = len(d.keys())

			for i in range(0, key_points):
				if not i in combined_key_points.keys():
					combined_key_points[i] = [d[i]]
				else:
					combined_key_points[i] = combined_key_points[i] + [d[i]]
		"""
		hand_average_groups = {
			4:[3,4,6,7],
			12:[7,8,11,12,15,16],
			20:[19,20,15,16]
		}
		"""
		hand_average_groups = {
			4:[3,4,6,7],
			12:[7,8,11,12,15,16],
			20:[19,20,15,16]
		}

		#this is the setup for the right hand. 3,4 are right elbow and right wrist respectively
		#use an ordered dict so we can prioritize thumb(4) over pinky(20), since 4 & 20 are e/o reflectors

		#option for improving -- look at the previous frames rotation between thumb and pinky, and use
		#it to determine the sign for 4 and 20 angle (index 3)

		hand_vector_groups = OrderedDict()
		hand_vector_groups[4] =  [3, 4, 20,  0.3, 20]
		hand_vector_groups[8] =  [3, 4, 10,  0.4, 20]
		hand_vector_groups[12] = [3, 4,  0,  0.5, None]
		hand_vector_groups[20] = [3, 4, -5, 0.2, 4]
		hand_vector_groups[17] = [3, 4, -5,  0.1, 2]
		hand_vector_groups[2] =  [3, 4,  5, 0.15, 20]
		hand_vector_groups[9] =  [3, 4,  0,  0.2, None]

		#sub out appropriate pose joints if it's the left hand
		if left_hand:
			l_hand_vector_groups = OrderedDict()
			for k, v in hand_vector_groups.items():
				#6,7 are left elbow and left wrist, respectively
				l_hand_vector_groups[k] = [6,7] + v[2:]
			hand_vector_groups = l_hand_vector_groups
		
		"""
		temp_d_1 = {}
		for key_point_num, key_points in combined_key_points.items():
			if key_point_num in hand_average_groups.keys():
				avg_processed_points = []
				for frame_num in range(0, len(key_points)):
					group = hand_average_groups[key_point_num]
					contributors = [combined_key_points[g][frame_num] for g in group]
					cp = ConstructedKeyPoint.construct_from_multipoint_average(contributors, key_point_num, conf_val=0.6)
					avg_processed_points.append(cp)
				temp_d_1[key_point_num] = avg_processed_points
			else:
				temp_d_1[key_point_num] = key_points
		"""
		#REMEMBER THAT YOU REMOVED THE AVERAGING
		temp_d_2 = {}
		temp_d_1 = combined_key_points
		processed_key_points = {}
		hand_fill_order = sorted(temp_d_1.keys())
		vector_tracker_dict ={
			12:9,
			8:5,
			20:17,
			16:13,
			4:1,
		}
		#temp_d_1 = combined_key_points
		for key_point_num in hand_fill_order:
			key_points = temp_d_1[key_point_num]
			if key_point_num in [0,1,5,9,13,17]:
				if left_hand:
					other_tracker = self.trackers['pose'][7]
				else:
					other_tracker = self.trackers['pose'][4]
			elif key_point_num in vector_tracker_dict.keys():
				o = vector_tracker_dict[key_point_num]
				other_tracker = self.trackers[point_type][o]
			else:
				other_tracker = self.trackers[point_type][(key_point_num-1)]
		#for key_point_num, key_points in temp_d_1.items():
			tracker = KeyPointTracker(key_points, key_point_num)
			if left_hand:
				other_tracker = self.trackers['pose'][7]
			else:
				other_tracker = self.trackers['pose'][4]
			tracker.last_confidence_vec_fill(self.hand_confidence, other_tracker, min_span=self.hand_min_span, tween_func=self.hand_tween)
			if self.hand_smooth:
				tracker.run_savgol_filter(window=self.hand_smooth_window)
			tracker.run_savgol_filter_confidences(window=self.hand_smooth_window)
			processed_key_points[key_point_num] = tracker.tracked_key_points

			self.trackers[point_type][key_point_num] = tracker
		"""
		#copy over the middle finger points
		point_mapping = {
			9:[0,1,5,13,17],
			10:[2,6,14,18],
			11:[3,7,15,19],
			12:[4,8,16,20]
		}

		chains = [
			[1,2,3,4],
			[5,6,7,8],
			[9,10,11,12],
			[13,14,15,16],
			[17,18,19,20]
		]

		for main_point, dep_points in point_mapping.items():
			for dep_point in dep_points:
				if dep_point == 0 :
					finger_chain = chains[0]
				for chain in chains:
					if dep_point in chain:
						finger_chain = chain
						break
				points = processed_key_points[dep_point]
				main_points = processed_key_points[main_point]
				l  = []
				for i in range(0, len(points)):
					check_point = points[i]
					conf_points = 0
					for chain_point in finger_chain:
						check_chain_point = processed_key_points[chain_point][i]
						if check_chain_point.c >= self.hand_confidence:
							conf_points += 1
					#if check_point.c <= self.hand_confidence:
					if conf_points < 4:
						p = main_points[i]
						kp = KeyPoint(p.x, p.y, 0.0, dep_point, source='copy')
						l.append(kp)
					else:
						l.append(check_point)
				processed_key_points[dep_point] = l
		"""

		"""
		processed_key_points = {}
		for key_point_num, key_points in temp_d_2.items():
			if left_hand:
				point_type = 'l_hand'
			else:
				point_type = 'r_hand'
			tracker = KeyPointTracker(key_points, key_point_num, point_type)
			processed_key_points[key_point_num] = tracker.last_confidence_kp_fill(0.1)
			self.trackers[point_type][key_point_num] = tracker
		"""

		if left_hand:
			logger.debug('PROCESSED KEY POINTS L HAND')
			logger.debug(processed_key_points)
			self.per_frame_hl_processed = list(zip(*processed_key_points.values()))
		else:
			self.per_frame_hr_processed = list(zip(*processed_key_points.values()))

	def process_single_hand_data02(self, left_hand=False):
		hand_frame_data = self.hr_frame_data
		point_type = 'r_hand'
		if left_hand:
			point_type = 'l_hand'
			hand_frame_data = self.hl_frame_data

		combined_key_points = {}
		processed_key_points = {}

		main_points = [9,10,11,12]
		main_trackers = {}
		for d in hand_frame_data:
			key_points = len(d.keys())

			for i in range(0, key_points):
				if not i in combined_key_points.keys():
					combined_key_points[i] = [d[i]]
				else:
					combined_key_points[i] = combined_key_points[i] + [d[i]]
		for main_point in main_points:
			main_key_points = combined_key_points[main_point]
			main_tracker = KeyPointTracker(main_key_points, main_point)	
			if left_hand:
				other_tracker = self.trackers['pose'][7]
			else:
				other_tracker = self.trackers['pose'][4]
			main_tracker.last_confidence_vec_fill(self.hand_confidence, other_tracker, min_span=self.hand_min_span, tween_func=self.hand_tween)	
			if self.hand_smooth:
				main_tracker.run_savgol_filter(window=self.hand_smooth_window)
			main_trackers[main_point] = main_tracker
			self.trackers[point_type][main_point] = main_tracker
			processed_key_points[main_point] = main_tracker.tracked_key_points

		point_mapping = {
			9:[0,1,5,13,17],
			10:[2,6,14,18],
			11:[3,7,15,19],
			12:[4,8,16,20]
		}
		for main_point, dep_points in point_mapping.items():
			for dep_point in dep_points:
				dep_key_points = combined_key_points[dep_point]
				other_tracker = main_trackers[main_point]
				tracker = KeyPointTracker(dep_key_points, dep_point)
				tracker.other_tracker_kp_fill(self.hand_confidence, other_tracker, min_span=self.hand_min_span, tween_func=self.hand_tween)
				if self.hand_smooth:
					tracker.run_savgol_filter(window=self.hand_smooth_window)
				self.trackers[point_type][dep_point] = tracker
				processed_key_points[dep_point] = tracker.tracked_key_points

		if left_hand:
			self.per_frame_hl_processed = list(zip(*processed_key_points.values()))
		else:
			self.per_frame_hr_processed = list(zip(*processed_key_points.values()))
			
		return

	def create_both_hand_rest_data(self):
		logger.info('****Creating Right Hand Rest Data******')
		#self.create_single_hand_rest_data()
		self.create_single_hand_rest_data()
		logger.info('****Creating Left Hand Rest Data******')
		#self.create_single_hand_rest_data(left_hand=True)	
		self.create_single_hand_rest_data(left_hand=True)

	def create_single_hand_rest_data02(self, left_hand=False):
		hand_frame_data = self.hr_frame_data
		point_type = 'r_hand_rest'
		if left_hand:
			point_type = 'l_hand_rest'
			hand_frame_data = self.hl_frame_data

		processed_key_points = {}
		combined_key_points = {}

		for d in hand_frame_data:
			key_points = len(d.keys())

			for i in range(0, key_points):
				if not i in combined_key_points.keys():
					combined_key_points[i] = [d[i]]
				else:
					combined_key_points[i] = combined_key_points[i] + [d[i]]

		if left_hand:
			#elbow_tracker = self.trackers['pose'][6]		
			wrist_tracker = self.trackers['pose'][7]
		else:
			#elbow_tracker = self.trackers['pose'][3]
			wrist_tracker = self.trackers['pose'][4]

		hand_rest_points = {
			4: [0.5, -30],
			8: [0.7, -10],
			12:[0.8, 0],
			16:[0.7, 10],
			20:[0.5, 20]
		}

		temp_d_1 =combined_key_points
		#just get a random tracker to see how many key points the dummy will need
		num_key_points = wrist_tracker.num_key_points

		for key_point_num in range(0, 21):
			if key_point_num in hand_rest_points.keys():
				use_num = 12
				key_points = temp_d_1[use_num]

				tracker = KeyPointTracker(key_points, key_point_num)
				tracker.last_confidence_relative_kp_fill(self.hand_confidence, wrist_tracker, min_span=self.hand_min_span, tween_func=self.hand_tween, start_copy=True)
				processed_key_points[key_point_num] = tracker.tracked_key_points
				self.trackers[point_type][key_point_num] = tracker
			else:
				constructor = KeyPointConstructor(num_key_points, i, point_type="dummy")
				constructor.dummy_replace()
				processed_key_points[key_point_num] = constructor.constructed_key_points
				self.trackers[point_type][key_point_num] = constructor

		if left_hand:
			logger.debug('PROCESSED KEY POINTS L HAND REST')
			logger.debug(processed_key_points)
			print('im in here')
			self.per_frame_hl_rest_processed = list(zip(*processed_key_points.values()))
		else:
			print('im in there')
			self.per_frame_hr_rest_processed = list(zip(*processed_key_points.values()))

		return	

	def create_single_hand_rest_data(self, left_hand=False):
		#Even tho we're creating hand data
		#We'll be inferring from elbow and wrist, which are pose data points
		pose_frame_data = self.pose_frame_data
		point_type = 'r_hand_rest'
		if left_hand:
			point_type = 'l_hand_rest'

		if left_hand:
			elbow_tracker = self.trackers['pose'][6]		
			wrist_tracker = self.trackers['pose'][7]
		else:
			elbow_tracker = self.trackers['pose'][3]
			wrist_tracker = self.trackers['pose'][4]
		
		#dictionary with, for each hand tip the magnitude and direction to use in constructing vector
		hand_rest_points = {
			4: [0.5, -30],
			8: [0.7, -10],
			12:[0.8, 0],
			16:[0.7, 10],
			20:[0.5, 20]
		}

		#all of the trackers should have the same number of key points
		#there's checking in the vec_replace func anyway
		num_key_points = elbow_tracker.num_key_points

		#there's 20 hand key points
		constructed_key_points = {}
		for i in range(0, 21):
			key_point_num = i
			if i in hand_rest_points.keys():
				t1 = elbow_tracker
				t2 = wrist_tracker

				#you can uncomment this if you want only the middle finger in rest vec
				#mag, theta = hand_rest_points[12]
				mag, theta = hand_rest_points[i]
				constructor = KeyPointConstructor(num_key_points, i, point_type="vec_replaced")
				constructor.vec_replace(t1, t2, mag, theta)

				

			else:
				constructor = KeyPointConstructor(num_key_points, i, point_type="dummy")
				constructor.dummy_replace()
			
			constructed_key_points[key_point_num] = constructor.constructed_key_points
			self.trackers[point_type][key_point_num] = constructor

		if left_hand:
			self.per_frame_hl_rest_processed = list(zip(*constructed_key_points.values()))
		else:
			self.per_frame_hr_rest_processed = list(zip(*constructed_key_points.values()))
			

	def write_processed_data(self, outdir=""):
		#write out the trackers into folders so can look at the point info per frame
		for k, v in self.trackers.items():
			for t, tracker in v.items():
				tracker.write_source_info(os.path.join(outdir,'trackers',k))
		
		#for i in range(0, len(self.frames)):
		logger.info('Writing %s processed frame jsons' % len(self.per_frame_pose_processed))
		for i in range(0, len(self.per_frame_pose_processed)):
			# if i % 4 != 0:
			# 	continue
			frame = self.frames[i]
			if frame.frame_data == None:
				continue
			p = self.per_frame_pose_processed[i]
			hl = self.per_frame_hl_processed[i]
			hr = self.per_frame_hr_processed[i]
			hl_rest = self.per_frame_hl_rest_processed[i]
			hr_rest = self.per_frame_hr_rest_processed[i]
			logger.debug('per frame hl processed')
			logger.debug(self.per_frame_hl_processed)
			logger.debug('per frame hl rest processed')
			logger.debug(self.per_frame_hl_rest_processed)
			frame.write_pose_data(os.path.join(outdir,'output'), p, hl, hr, hl_rest, hr_rest)
			