from obspy.geodetics.base import locations2degrees, degrees2kilometers, kilometers2degrees
from scipy import interpolate
from datetime import datetime
import numpy as np
from distance_calculations import calculate_3d_distance_path_dependent, calculate_3d_distance, guess_depth_and_change_to_km
from tqdm import tqdm
from IPython.display import clear_output
from waveform_timeshift import WaveformShifter
from nearestdistance import ClosestEarthquake
import psutil

def interpolate_waveform(waveform, old_dt, new_dt):
    """ this function interpolates a waveform with sampling spacing of old_dt to a new sample spacing of new_dt
    param::
    waveform :: np.ndarray
    old_dt :: float :: the sample spacing of the input
    new_dt :: float :: the sample spacing of the output
    """
    
    old_dt = float(old_dt)
    new_dt = float(new_dt)
    # get the sample length
    
    waveform_sample_length = waveform.shape[0]
    
    # get the maximum value of the range
    original_x_stop = waveform_sample_length * old_dt
    
    # get the base range
    original_x = np.arange(0, original_x_stop, old_dt)
    
    # create the range for interpolation (do not include last value, interpolation 
    interpolated_x_stop = original_x_stop - old_dt
    interpolated_x = np.arange(0, interpolated_x_stop, new_dt)
    
    # because the last value is not included in the upsampling, it needs to be padded
    additional_pad_length = int(old_dt/new_dt)
    
    # define the interpolator and interpolate
    f_interpolator = interpolate.interp1d(original_x, waveform, kind="cubic", fill_value="extrapolate")     
    interpolated_waveform = f_interpolator(interpolated_x)
    interpolated_waveform = np.pad(interpolated_waveform, (0,additional_pad_length))
    
    return interpolated_waveform

def shift_waveform_by_time(waveform, time_shift, original_sampling_interval):
    """ this function time-shifts the a waveform given its original sampling interval.
    the waveform is upsampled so that a suitable version is found for upsampling.
    the time shift can be negative or positive
    """
    original_waveform_length = waveform.shape[0]
    
    # store the direction of the time shift and the absolute value of it
    direction = np.sign(time_shift)
    time_shift = np.abs(time_shift)
      
    # this way of doing it takes care of shift values above the original sampling intervall as well
    upsampled_interval = original_sampling_interval/512.
    if upsampled_interval > time_shift and time_shift!=0:
        upsampled_interval = time_shift / 16
    
    upsampled_waveform = interpolate_waveform(waveform, old_dt=original_sampling_interval, new_dt = upsampled_interval)
    
    # calculate the shiftvalue and shift the upsampled waveform
    shift_value = int(direction*time_shift/upsampled_interval)
    shifted_waveform = np.roll(upsampled_waveform,  shift_value)
    
    # downsampling: calculate downsample step and sample accordingly, up to the length of the original waveform
    downsample_step = int(original_sampling_interval/upsampled_interval)
    downsampled_waveform = shifted_waveform[::downsample_step][:original_waveform_length]
    
    return downsampled_waveform


def create_new_locations(base_latitude, base_longitude, base_depth, distance, number_elements):
    """ This function creates a new set of locations starting from the base values up to the allowed distance (in km)
    """
    # take new elements on both sides
    total_elements = 2*number_elements + 1
    
    # convert the given distance (in kilometers) to degrees, then split the allowed distance up into steps
    converted_degrees_distance = kilometers2degrees(distance)
    degree_steps = 2*converted_degrees_distance / (total_elements-1)
    distance_steps = 2*distance / (total_elements -1)
    # print(f"distance_steps = {distance_steps}, distance = {distance}, base_depth = {base_depth}")
    
    # locations2degrees(lat1, long1, lat2, long2)
    new_locations = []
    for i in range(total_elements):
        for j in range(total_elements):
            for k in range(total_elements):
                new_latitude = base_latitude - converted_degrees_distance + i*degree_steps
                new_longitude = base_longitude - converted_degrees_distance +  j*degree_steps
                new_depth = base_depth - distance + distance_steps*k
                if new_depth > 0.:
                    new_locations.append([new_longitude, new_latitude, new_depth])
    new_locations = np.array(new_locations)
                             
    return new_locations
    

def create_set_of_shifted_waveforms(waveform, earthquake_location, receiver_location,
                                    new_elements=10,
                                    allowed_distance = 10.0, 
                                    velocity = 2.7,
                                    method_of_calculation="split_path",
                                    original_sampling_interval=5.):
    """ This function creates a set of shifted waveform based on a base waveform.
    It creates new waveforms with the number of new_elements in each direction of 3d space.
    The earthquake location is given in degrees, the allowed_distance is given in kilometers.
    
    param::
    waveform:: np.ndarray
    earthquake_location:: np.ndarray in format [longitude, latitude, depth]
    receiver_location :: array in format [longitude, latitude]
    new_elements:: int
    allowed_distance:: float
    """
    receiver_longitude = receiver_location[0]
    receiver_latitude = receiver_location[1]
    
    longitude = earthquake_location[0]
    latitude = earthquake_location[1]
    depth = earthquake_location[2]
    

    # change the depth values if they are in [m]
    depth_km = guess_depth_and_change_to_km(depth)

    
    # print(psutil.virtual_memory())
    
    new_locations = create_new_locations(base_latitude=latitude, 
                                        base_longitude=longitude, 
                                        base_depth=depth_km, 
                                        distance=allowed_distance,
                                        number_elements=new_elements)
    
    # calculate distance from original location to receiver
    original_to_receiver = calculate_3d_distance_path_dependent(latitude, longitude, depth, receiver_latitude, receiver_longitude)
    
    # calculate distance from new locations to receiver
    distances = calculate_3d_distance_path_dependent(new_locations[:,1], new_locations[:,0] , new_locations[:,2]*1000,
                                                     receiver_latitude, receiver_longitude)

    
    distance_difference =distances - original_to_receiver

    # calculate time shift based on velocity value
    calculated_time_shift = distance_difference / velocity
    # print(psutil.virtual_memory())
    
    # create waveforms with time shifts
    
    """
    new_waveforms = []
    for time_shift_s in tqdm(calculated_time_shift, leave=False):
        new_waveform = \
            shift_waveform_by_time(waveform=waveform, time_shift = time_shift_s, original_sampling_interval=original_sampling_interval)
        new_waveforms.append(new_waveform)
        
    new_waveforms = np.array(new_waveforms)
    """
    # instead with the more efficient WaveformShifter class 
    waveform_shifter = WaveformShifter(waveform, original_sampling_interval)
    new_waveforms = waveform_shifter.shift_by_time_bulk(calculated_time_shift)
    # print(psutil.virtual_memory())
    
    return new_locations, new_waveforms

def create_distance_matrix(latitude, longitude, depth, method_of_calculation="split_path"):
    print("starting to calculate the distance matrix")
    num_elements = latitude.shape[0]
    distance_matrix = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(i):
            if (latitude[i] == latitude[j]) and (longitude[i]==longitude[j]) and (depth[i] == depth[j]):
                distance = 0
            else:
                distance = calculate_3d_distance(lat1 = latitude[i], lon1 = longitude[i], depth1 = depth[i],
                                                 lat2 = latitude[j], lon2 = longitude[j], depth2 = depth[j])
            distance_matrix[i,j] = distance
            distance_matrix[j,i] = distance
    return distance_matrix

def create_shifted_dataset(waveforms, earthquake_parameters, receiver_location,
                                    new_elements=10,
               #                     allowed_distance = 10.0, this is unnecessary?
                                    velocity = 2.7,
                                    method_of_calculation="split_path",
                                    allowed_distance_ratio = 0.25,
                                    original_sampling_interval = 5.):
    """ this function creates both sets of data: earthquake parameters and waveforms. it calculates
    the maximum allowed distance with a factor to the nearest neighbouring earthquake
    
    new elements is the number of new elements in each direction, the total number of elements along one axis
    is 2*new_elements + 1
    velocity is the velocity used for inferring time shifts
    allowed_distance_ratio is the ratio of distance between earthquakes up to which new earthquakes will be created,
    so if the closest nearby earthquake is 10km away a ratio 0.25 will only lead to new earthquakes 0.25 in that direction
    
    input:
    waveforms:: np.array of waveforms
    earthquake_parameters :: np.array of parameters ([longitude, latitude, depth, m_rr_norm, m_rt_norm, m_rp_norm])
    allowed_distance_ratio :: float
    """
    # unpack
    [longitude, latitude, depth, m_rr_norm, m_rt_norm, m_rp_norm] = [earthquake_parameters[:,0],
                                                                  earthquake_parameters[:,1],
                                                                  earthquake_parameters[:,2],
                                                                  earthquake_parameters[:,3],
                                                                  earthquake_parameters[:,4],
                                                                  earthquake_parameters[:,5]]
    original_number_earthquakes = longitude.shape[0]
    
    
    earthquake_distance = ClosestEarthquake(earthquake_parameters[:,:3])

    new_earthquake_list = []
    new_waveforms_list = []
    
    for i in tqdm(range(original_number_earthquakes)): 
        # choose the base value for the calculation in this iteration
        earthquake_location = np.array([longitude[i], latitude[i], depth[i]])
        moment_tensor = np.array([m_rr_norm[i], m_rt_norm[i], m_rp_norm[i]])
        waveform = waveforms[i]
        
        # if there is only one earthquake given, set the allowed distance to 100
        if original_number_earthquakes == 1:
            allowed_distance = 100.
        # define the allowed distance
        else:
            closest_distance = earthquake_distance.get_nearest_distance(
                    longitude[i], latitude[i], depth[i])
            allowed_distance = closest_distance * allowed_distance_ratio        
        
        new_locations, new_waveforms = create_set_of_shifted_waveforms(
                waveform, earthquake_location, receiver_location,
                                        new_elements=new_elements,
                                        allowed_distance = allowed_distance, 
                                        velocity = velocity,
                                        method_of_calculation="split_path",
                                        original_sampling_interval = original_sampling_interval)
        # generate a moment tensor array to be stacked to the location parameters
        moment_tensor_stack = np.zeros(new_locations.shape) + moment_tensor
        new_earthquake_parameters = np.hstack((new_locations, moment_tensor_stack))
        
        # append the items to the list
        new_earthquake_list.append(new_earthquake_parameters)
        new_waveforms_list.append(new_waveforms)
        
        # print(allowed_distance)
        # clear_output()
        # print(f"Step {i+1} of {original_number_earthquakes} done.")
        
    new_earthquakes = np.vstack(new_earthquake_list)
    new_waveforms = np.vstack(new_waveforms_list)
    
    return new_earthquakes, new_waveforms

class ShiftAugmentor(object):
    """docstring for ShiftAugmentor."""

    def __init__(self, waveforms = None, earthquake_parameters = None, receiver_location =None, new_elements=2,
                 velocity = 2.7, method_of_calculation="split_path", allowed_distance_ratio = 0.25, original_sampling_interval = 5.,
                 depth_output = "m"):
        super(ShiftAugmentor, self).__init__()
        self.waveforms = waveforms
        self.earthquake_parameters = earthquake_parameters
        self.receiver_location = receiver_location
        
        self.__check_variable_state()
        
        # assign data values which have default values
        self.new_elements = new_elements
        self.velocity = velocity
        self.method_of_calculation = method_of_calculation
        self.allowed_distance_ratio = allowed_distance_ratio
        self.original_sampling_interval = original_sampling_interval
        self.depth_output = depth_output
        
        # print(f"new_elements is {new_elements}")

                
            
    def __check_variable_state(self):
        # check if values for inputs are set and set state variable accordingly
        if (self.waveforms is None) or (self.earthquake_parameters is None) or \
            (self.receiver_location is None):
            self.all_inputs_set = False
            print("Warning: Inputs missing")
        else:
            self.all_inputs_set = True
            
    def __create_output_string(self):
        """ this private method creates an output string of the internal variables
        """
        try:
            waveforms_string = self.waveforms.shape
        except:
            waveforms_string = self.waveforms
        try:
            earthquake_parameters_string = self.earthquake_parameters.shape
        except:
            earthquake_parameters_string = self.earthquake_parameters
        output_string = \
        "---- Inputs ----\n" + \
        f"Waveforms = {waveforms_string}\n" + \
        f"Earthquake Parameters = {earthquake_parameters_string}\n" + \
        f"Receiver Location = {self.receiver_location}\n" + \
        f"---- Augmentation Properties ----\n" + \
        f"New elements = {self.new_elements}\n" + \
        f"Velocity = {self.velocity}\n" + \
        f"Method of calculation = {self.method_of_calculation}\n" + \
        f"Allowed distance ratio = {self.allowed_distance_ratio}\n"+\
        f"Original sampling interval = {self.original_sampling_interval}\n"+\
        f"Depth output unit = [{self.depth_output}]"
        return output_string
            
    def create_shifted_dataset(self, waveforms = None, earthquake_parameters = None, receiver_location = None,
                                       new_elements = None, velocity = None, method_of_calculation = None):
        """ this public method creates the augmented waveforms. Checks if augmentation parameters
        and input variables are given. 
        """ 
        # if new inputs have been given, overwrite the inputs into the instance properties 
        if (waveforms is not None) or (earthquake_parameters is not None) or (receiver_location is not None):
            self.set_input(waveforms, earthquake_parameters, receiver_location)
        # same for augmentation properties
        if (new_elements is not None) or (velocity is not None) or (method_of_calculation is not None):
            self.set_augmentation_parameters(new_elements, velocity, method_of_calculation)
            
        self.__check_variable_state()
        # initialize outputs for when function is called without parameters set 
        new_earthquakes, new_waveforms = None, None

        if self.all_inputs_set:
            new_earthquakes, new_waveforms = create_shifted_dataset(waveforms = self.waveforms, 
                                                                    earthquake_parameters=self.earthquake_parameters, 
                                                                    receiver_location=self.receiver_location,  
                                                                    new_elements=self.new_elements,
                                                                    velocity=self.velocity,
                                                                    allowed_distance_ratio = self.allowed_distance_ratio,
                                                                    original_sampling_interval = self.original_sampling_interval)
        else: # create error message
            print("Error: Not all input values are set. \n"+ \
                  "Please define the inputs: waveforms, earthquake parameters and receiver location")
        if self.depth_output == "m":
            # convert to meters
            new_earthquakes[:,2] = new_earthquakes[:,2]*1000.
        return new_earthquakes, new_waveforms
    
    def set_input(self, waveforms = None, earthquake_parameters = None, receiver_location = None):
        """ this public method sets the input parameters, if they have been given
        """
        if (waveforms is not None):
            self.waveforms = waveforms
        if (earthquake_parameters is not None):
            self.earthquake_parameters = earthquake_parameters
        if (receiver_location is not None):
            self.receiver_location = receiver_location
    
    def set_augmentation_parameters(self, new_elements = None, velocity = None, method_of_calculation = None, allowed_distance_ratio=None):
        """ this public method sets the parameters of the augmentation method
        """
        if new_elements is not None:
            self.new_elements = new_elements
        if velocity is not None:
            self.velocity = velocity
        if method_of_calculation is not None:
            self.method_of_calculation = method_of_calculation
        if allowed_distance_ratio is not None:
            self.allowed_distance_ratio = allowed_distance_ratio
    
    def store_settings(self, filename, ID=None):
        """ this public method stores all the internal variables for input and 
        augmentation settings in the given filename; if a ID code is given, it is 
        written to the file as well
        """
        # store all settings in a plain text file,
        # store also the data and time of creation, and possibly with corresponding data
        output_string = self.__create_output_string()
        current_time = datetime.now()
        output_string += f"\n---- Time of creation ----\n{current_time}"
        
        if ID is not None:
            output_string = f"---- ID ----\nID ={ID}\n" + \
                output_string
        with open(filename, 'w') as f:
            f.write(output_string)
    
    def display_properties(self):
        """ this method displays the stored values
        """        
        output_string= self.__create_output_string()
        print(output_string)