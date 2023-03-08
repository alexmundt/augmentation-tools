import numpy as np
from scipy import interpolate
from tqdm import tqdm


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


class WaveformShifter(object):
    """ docstring for waveform_shift"""
    
    def __init__(self, waveform, original_sampling_interval, number_new_samples = 1024):
        
        # bind the attributes to be stored within the class
        self.waveform = waveform
        self.original_sampling_interval = original_sampling_interval
        
        # store the original waveform length for downsampling down the line
        self.original_waveform_length = waveform.shape[0]
        
        # define the interval to be upsampled to (check later if the time shift is smaller than the interval)
        self.upsampled_interval = original_sampling_interval/number_new_samples
        
        self.__upsample()
    
    def __upsample(self, upsampled_interval =None):
        if upsampled_interval == None:
            upsampled_interval = self.upsampled_interval
        # generate waveform
        upsampled_waveform = interpolate_waveform(self.waveform, old_dt=self.original_sampling_interval, new_dt = upsampled_interval)
        # bind to instance
        self.upsampled_waveform = upsampled_waveform
        self.upsampled_interval = upsampled_interval
        
    def get_upsampled_waveform(self):
        return self.upsampled_waveform

    
    def shift_by_time(self, time_shift):
        # store the direction of the time shift and the absolute value of it
        direction = np.sign(time_shift)
        time_shift = np.abs(time_shift)
        
        # calculate the shiftvalue and shift the upsampled waveform
        shift_value = int(direction*time_shift/self.upsampled_interval)
        shifted_waveform = np.roll(self.upsampled_waveform,  shift_value)

        # downsampling: calculate downsample step and sample accordingly, up to the length of the original waveform
        downsample_step = int(self.original_sampling_interval/self.upsampled_interval)
        downsampled_waveform = shifted_waveform[::downsample_step][:self.original_waveform_length]

        return downsampled_waveform
    
    def shift_by_time_bulk(self, time_shifts):
        """ this method shifts the waveform by a set of time shifts
        """
        num_shifts = time_shifts.shape[0]
        # get the minimum time shift of the set to adapt the upsampling interval
        
        shifted_waveforms = []
        for i in range(num_shifts):
            current_time_shift = time_shifts[i]
            
            # check if the current time shift is too small for the interpolated waveform
            if np.abs(current_time_shift) < self.upsampled_interval:
                new_waveform = self.waveform
            else:
                new_waveform = self.shift_by_time(current_time_shift)
                
            shifted_waveforms.append(new_waveform)
        shifted_waveforms = np.array(shifted_waveforms)
        
        return shifted_waveforms