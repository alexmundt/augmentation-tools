import numpy as np
from tslearn.metrics import dtw_path
from tqdm import tqdm


def align_waveforms(base_waveform, align_waveform):
    """ this method aligns two waveforms based on dtw matching of pairs.
    when indexes in the to be aligned waveform remain in place, then the corresponding value is simply copied.
    when indexes in the base waveform remain in place the values of the be-aligned waveforms are averaged
    param:
    waveform:: time series array
    """
    aligned_waveform = np.zeros(len(base_waveform))
    dtw_path_values, dtw_similarity = dtw_path(base_waveform, align_waveform) #global_constraint="sakoe_chiba", sakoe_chiba_radius = 100)

    last_index = None
    for i in range(len(dtw_path_values)):
        dtw_index_pair = dtw_path_values[i]
        
        # get the corresponding index-pair of the time warped path
        current_index = dtw_index_pair[0]
        align_index = dtw_index_pair[1]
        
        # check if the time warp path assigns to a new value in the time series
        if current_index != last_index:
            aligned_waveform[current_index] = align_waveform[align_index]
            # aligned_waveform[current_index] = 1
            
            # reset the count for the sum over the same values
            sum_count = 1
            sum_align = align_waveform[align_index]
        elif current_index == last_index: 
            sum_count += 1
            
            # sum up the previous value
            sum_align += align_waveform[align_index] 
            new_value = sum_align / sum_count
            aligned_waveform[current_index] = new_value
            # aligned_waveform[current_index] = 1 
    
        last_index = current_index
    return aligned_waveform
    

def create_aligned_waveform_cluster_array(cluster, base_waveform):
    """ This function aligns all the waveforms in the cluster to the base_waveform.
    This is the array version expecting a cluster in array shape (n,1000)
    param::
    cluster :: np.ndarray(n,1000)
    base_waveform :: np.array
    """
    list_of_dtw_waveforms = []
    cluster_length = cluster.shape[0]
    for i in range(cluster_length):
        waveform = cluster[i]
        new_waveform = align_waveforms(base_waveform, waveform)
        list_of_dtw_waveforms.append(new_waveform)
    return list_of_dtw_waveforms    

def base_time_series_inversion_numpy(data, coefficients):
    """ The code inverts the data in a model for a set of base time series. Output is the set of base time series.
    param
    data:: np.array of two dimensions with (n,m) shape, where n is the number of time series and m the number of samples in the time series
    coefficients:: 2-dimensional array with (n,c) shape where m is the number of time series 
    """
    
    #  normal equation is (X^T X)^-1*X^T
  
    sample_length = data.shape[1]
    solution = []
    for i in range(sample_length):
        # inversion_result = np.dot(factor, get_data_value(data, i))
        (solution_values, residuals, rank, singular_values) = np.linalg.lstsq(coefficients, data[:,i], rcond=None)

        solution.append(solution_values)
    solution = np.array(solution)
    
    return solution



def generate_from_base_waveform(base_waveforms, base_mt_parameters, parameter_variation=0.1,number_of_new_data=20, clip_values=True):
    """
    param:
    base_waveforms:: np.array of shape (1000, 3)
    base_mt_parameters:: np.array of shape(3)
    parameter_variation:: either np.array of shape(3) or single float or int
    number_of_new_data:: int
    """
    if isinstance(parameter_variation, float):
        # assign values
        x_var, y_var, z_var = parameter_variation, parameter_variation, parameter_variation
    elif isinstance(parameter_variation, np.ndarray) or isinstance(parameter_variation, list):
        if len(parameter_variation) < 3:
            print("Error: Not enough parameter values given, please specify 3 parameter variation values. Using standard value of 0.1")
            x_var, y_var, z_var = 0.1, 0.1, 0.1
        else:
            x_var, y_var, z_var = parameter_variation[0], parameter_variation[1], parameter_variation[2]
    else:
        print("Error: No proper parameter variation specified. Picking standard value of 0.1")
        x_var, y_var, z_var = 0.1, 0.1, 0.1

    x_mid, y_mid, z_mid = base_mt_parameters[0], base_mt_parameters[1], base_mt_parameters[2]
    
    # generate new values
    x_values = np.linspace(x_mid-x_var*0.5, x_mid + x_var*0.5, num=number_of_new_data)
    y_values = np.linspace(y_mid-y_var*0.5, y_mid + y_var*0.5, num=number_of_new_data)
    z_values = np.linspace(z_mid-z_var*0.5, z_mid + z_var*0.5, num=number_of_new_data)

    # clip the values:
    if clip_values == True:
        # x_values = np.clip(x_values, -1., 1.)
        # y_values = np.clip(y_values, -1., 1.)
        # z_values = np.clip(z_values, -1., 1.)
        x_values = np.delete(x_values, np.argwhere( (x_values > 1.)  ))
        y_values = np.delete(y_values, np.argwhere( (y_values > 1.) ))
        z_values = np.delete(z_values, np.argwhere( (z_values > 1.) ))
        x_values = np.delete(x_values, np.argwhere( (x_values < -1.) ))
        y_values = np.delete(y_values, np.argwhere( (y_values < -1.) ))
        z_values = np.delete(z_values, np.argwhere( (z_values < -1.) ))
    # mesh_result = np.meshgrid(x_values, y_values, z_values)
    # print(y_values)
    new_parameter_list = []
    # print(x_values)
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            for k in range(len(z_values)):
                new_parameter_list.append([x_values[i], y_values[j], z_values[k]])
    
    # print(mesh_result.shape)
    # print(z_values)
    new_parameter_array = np.array(new_parameter_list)
    
    new_waveforms = np.dot(base_waveforms,new_parameter_array.T)
    return new_parameter_array, new_waveforms

def combine_earthquake_parameters(moment_tensor, earthquake_location):
    """ this function combines moment tensor values and earthquake location
    output format should be:
    ['longitude', 'latitude', 'depth', 'm_rr_norm','m_rt_norm', 'm_rp_norm']
    input param:
    earthquake_location:: np.ndarray of shape (n,3)
    moment_tensor:: np.ndarray of shape (n,3)
    """
    # check if both input parameters have the same length
    if moment_tensor.shape != earthquake_location.shape:
        print("Error: incorrect input shapes")
        output = None
    else:
        output= np.hstack((moment_tensor, earthquake_location))
    
    return output



def augment_cluster(waveforms, earthquake_parameters, parameter_variation=0.25, 
            number_of_new_data=10, clip_values=True, additional_weight = True):
    """ this function augments the waveforms of a cluster by creating new waveforms
    based on a waveform inversion method that takes moment tensor values into account
    param waveforms:: np.ndarray of shape (n, 1000) where 1000 is the waveform sample size and n the number of waveforms
    param earthquake_parameters:: np.ndarray of shape (n, 6) where n is the number of waveforms, the first 3 are locations, the last 3 moment tensor values
    """
    
    output = None
    print(f"Waveform shape = {waveforms.shape}")
    print(f"Parameter shape = {earthquake_parameters.shape}")
    
    num_waveforms = waveforms.shape[0]
    
    mt_values = earthquake_parameters[:,3:]
    earthquake_locations = earthquake_parameters[:,:3]
    
    print(f"Moment tensor data shape = {mt_values.shape}")
    
    new_parameter_list = []
    new_waveform_list = []
    
    for i in tqdm(range(num_waveforms)):
        # pick first waveform for the time warping and aligning
        time_warp_base_waveform = waveforms[i]
        base_mt_parameters = mt_values[i]
        
        # create aligned cluster
        aligned_cluster = create_aligned_waveform_cluster_array(cluster = waveforms, base_waveform= time_warp_base_waveform)
        
        aligned_cluster_array = np.array(aligned_cluster)
        
        # add additional weight
        if additional_weight == True:
            dupl_waveforms = aligned_cluster_array
            dupl_mt_values = mt_values
        
            for j in range(100):
                dupl_waveforms = np.vstack((dupl_waveforms, time_warp_base_waveform))
                dupl_mt_values = np.vstack((dupl_mt_values, base_mt_parameters))
                
#             aligned_cluster_array = dupl_waveforms
#             mt_values = dupl_mt_values
            # print(aligned_cluster_array.shape)
            # print(mt_values.shape)
            waveforms_to_invert = dupl_waveforms
            parameters_to_invert = dupl_mt_values
        else:
            waveforms_to_invert = aligned_cluster_array
            parameters_to_invert = mt_values

        # invert
        inverted_baseforms = base_time_series_inversion_numpy(data=waveforms_to_invert, coefficients=parameters_to_invert)
        
        # create new values
        added_parameter_array, added_waveforms = generate_from_base_waveform(
            base_waveforms = inverted_baseforms, base_mt_parameters=base_mt_parameters, parameter_variation=parameter_variation, 
            number_of_new_data=number_of_new_data, clip_values=clip_values)
        
        # create duplicates of the earthquake location parameters to attach to the new parameter array
        parameter_length = added_parameter_array.shape[0]
        duplicate_locations = earthquake_locations[i]*np.ones((parameter_length, 3))
              
        # attach to form the full stack of parameters
        new_full_parameters = np.hstack((duplicate_locations, added_parameter_array))
        
        new_parameter_list.append(new_full_parameters)
        new_waveform_list.append(added_waveforms.T)
        
    
    new_waveforms = np.vstack(new_waveform_list)
    new_parameters = np.vstack(new_parameter_list)
    
    return new_parameters, new_waveforms