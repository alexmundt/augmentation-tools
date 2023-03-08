import numpy as np
from obspy.geodetics.base import locations2degrees, degrees2kilometers, kilometers2degrees


def change_from_m_to_km(value):
    return value/1000.

def change_from_km_to_m(value):
    return value*1000.

def guess_depth_and_change_to_km(depth):
    """ This functions guesses if a value or array is in m and changes it to km.
    input:
    param: depth :: float or np.array
    """
    depth = np.array(depth)
    # guess depth (m or km?) - deepest earthquake recorded about 700km:
    if depth.mean() > 700.:
        # The mean of the depth is above 700. This value is therefore likely to be in m.
        depth_dimension = "m"
        depth_km = change_from_m_to_km(depth)
    else:
        # The mean of the depth is below 700. This value is therefore likely to be in km
        depth_dimension ="km"
        depth_km = depth
    return depth_km

def calculate_3d_distance(lat1, lon1, depth1, lat2, lon2, depth2):
    """ calculates the distance between two locations within the earth
    returns distance in km
    assumes a spherical earth and that the pythagorean theorem can be used, i.e. small distances without taking curvature of the earth into account
    """
    depth1 = guess_depth_and_change_to_km(depth1)
    depth2 = guess_depth_and_change_to_km(depth2)    
    
    # need to calculate distance in km
    dist_degrees = locations2degrees(lat1, lon1, lat2, lon2)
    dist_km = degrees2kilometers(dist_degrees)
    
    # calculate the distance based on pythagoras theorem
    dist_pythagoras = np.sqrt(dist_km**2.+((depth2-depth1))**2.)
    # print(dist_degrees, dist_km, dist_pythagoras)
    
    return dist_pythagoras

def calculate_3d_distance_path_dependent(lat1, lon1, depth1, lat2, lon2):
    """ calculates the distance between two points within or on the Earth, the function assumes
    a path change during traveling. at first the wave travels with an angle of 45° to the surface normal
    to the Earth's surface, then it travels along the surface. The second station is assumed to be on the surface
    returns distance in km
    assumes a spherical earth and that the pythagorean theorem can be used, i.e. small distances without taking curvature of the earth into account
    
    param::
    depth1::float in [m] or [km]
    
    output: float in km
    """
    depth1 = guess_depth_and_change_to_km(depth1)  
    
    hypocenter_path = locations2degrees(lat1, lon1, lat2, lon2)
    # convert to kilometers
    hypocenter_path = degrees2kilometers(hypocenter_path)
    # calculate the distance the 45° upwards travelling wave passes
    upwards_path = (depth1)*np.sqrt(2.)
    # combine, note: the wave has already travelled the depth across the hypocenter_path (assuming a 45° angle)
    real_path = upwards_path + hypocenter_path - depth1
    
    return real_path