import numpy as np
from tqdm import tqdm
from tools import augment_cluster


class InversionAugmentor(object):
    """docstring for InversionAugmentor."""

    def __init__(self, waveforms = None, earthquake_parameters = None, new_elements=2, parameter_variation=0.2, additional_weight = False):
        super(InversionAugmentor, self).__init__()
        self.waveforms = waveforms
        self.earthquake_parameters = earthquake_parameters
        
        self.__check_variable_state()
        
        # assign data values which have default values
        self.new_elements = new_elements
        self.parameter_variation = parameter_variation
        self.additional_weight = additional_weight

        # print(f"new_elements is {new_elements}")                
            
    def __check_variable_state(self):
        # check if values for inputs are set and set state variable accordingly
        if (self.waveforms is None) or (self.earthquake_parameters is None):
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
        f"---- Augmentation Properties ----\n" + \
        f"New elements = {self.new_elements}\n" + \
        f"Parameter variation = {self.parameter_variation}\n" + \
        f"Additional weight = {self.additional_weight}"

        return output_string
            
    def create_new_dataset(self, waveforms = None, earthquake_parameters = None, new_elements = None, parameter_variation = None,
                            additional_weight = None):
        """ this public method creates new waveforms for augmentation. Checks if augmentation parameters
        and input variables are given and, if given, uses them instead of the properties of the instance. 
        """ 
        # if new inputs have been given, overwrite the inputs into the instance properties 
        if (waveforms is not None) or (earthquake_parameters is not None):
            self.set_input(waveforms, earthquake_parameters)
        # same for augmentation properties
        if (new_elements is not None) or (parameter_variation is not None) or (
            additional_weight is not None):
            self.set_augmentation_parameters(new_elements, parameter_variation, additional_weight)
            
        self.__check_variable_state()
        # initialize outputs for when function is called without parameters set 
        new_earthquakes, new_waveforms = None, None

        if self.all_inputs_set:       
        
            # the actual generation is done here
            new_earthquakes, new_waveforms = augment_cluster(
                waveforms= self.waveforms,
                earthquake_parameters =  self.earthquake_parameters,
                parameter_variation=self.parameter_variation, 
                number_of_new_data= self.new_elements, clip_values=True,
                additional_weight = self.additional_weight)
        else: # create error message
            print("Error: Not all input values are set. \n"+ \
                  "Please define the inputs: waveforms and/or earthquake parameters")

        return new_earthquakes, new_waveforms
    
    def set_input(self, waveforms = None, earthquake_parameters = None):
        """ this public method sets the input parameters, if they have been given
        """
        if (waveforms is not None):
            self.waveforms = waveforms
        if (earthquake_parameters is not None):
            self.earthquake_parameters = earthquake_parameters
    
    def set_augmentation_parameters(self, new_elements = None, parameter_variation = None, additional_weight = None):
        """ this public method sets the parameters of the augmentation method
        """
        if new_elements is not None:
            self.new_elements = new_elements
        if parameter_variation is not None:
            self.parameter_variation = parameter_variation
        if additional_weight is not None:
            self.additional_weight = additional_weight
    
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