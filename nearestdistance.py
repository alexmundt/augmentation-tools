from distance_calculations import calculate_3d_distance_path_dependent, calculate_3d_distance, guess_depth_and_change_to_km
import numpy as np
from tqdm import tqdm

class ClosestEarthquake(object):
    """ This class handles the task of providing the distance to the nearest earthquake given
    an earthquake at a specific location as input.
    """ 
    def __init__(self, earthquake_parameters):
        self.earthquake_parameters = earthquake_parameters[:,:3]
        
        self.__remove_duplicates()
        self.__calculate_distance()
        pass
    
    def __remove_duplicates(self):
        unique_parameters = np.unique(self.earthquake_parameters, axis=0)
        self.earthquake_parameters = unique_parameters
        
    def __calculate_distance(self):
        num_elements = self.earthquake_parameters.shape[0]
        earthquake_parameters = self.earthquake_parameters
        print("Starting to calculating distances...")
        
        # order of the earthquake parameters is defined as longitudes, latitudes, depths
        longitude, latitude, depth = earthquake_parameters[:,0], earthquake_parameters[:,1], earthquake_parameters[:,2]
        dict_distances = {}
        for i in range(num_elements):
            minimum_i = np.inf
            for j in range(num_elements):
                distance = calculate_3d_distance(lat1 = latitude[i], lon1 = longitude[i], depth1 = depth[i],
                                                 lat2 = latitude[j], lon2 = longitude[j], depth2 = depth[j])
                # print(f"distance at {i,j} at {distance}")
                if distance < minimum_i and distance > 0 :
                    minimum_i = distance
                    
            key_string = str(earthquake_parameters[i].tolist())
            dict_distances[key_string] = minimum_i
    
        print("Calculation done.")
        self.distances = dict_distances
        
    def get_nearest_distance(self, longitude, latitude, depth):
        """ this method gets the distance to the nearest earthquake 
        of the earthquake at the given location
        """
        earthquake_location = np.array([longitude, latitude, depth])
        key_string = str(earthquake_location.tolist())
        value = self.distances[key_string]
        
        return value