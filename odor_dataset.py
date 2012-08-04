import pickle
import numpy as np
import matplotlib.pyplot as plt
import fit_data.fit_data as fit_data

class Dataset():
    def __init__(self):
        self.odor_traces = {}
    def get_odor_trace(self, n=0):
        key = self.odor_traces.keys()[n] 
        return self.odor_traces[key]
        
class Odor_Trace(object):
    def __init__(self, position):
        if type(position) == list:
            position = np.array(position)
        self.position = position
        self.trace = None
        self.timestamps = None
        
        # for real data will need to interpolate to nicer values
        self.raw_trace = None
        self.raw_timestamps = None
        
def get_traces_along_axis(dataset, axis, position, max_error=0.01):
    # find all 

    keys = []    
    for key, odor_trace in dataset.odor_traces.items():
        err = np.abs(odor_trace.position - position)
        err[axis] = 0
        errsum = np.sum(err)
        if errsum < max_error:
            keys.append(key)
    
    return keys
    
def calc_peak_of_odor_trace(odor_trace):
    odor_trace.peak_frame = np.argmax(odor_trace.trace)
    odor_trace.peak_time = odor_trace.timestamps[odor_trace.peak_frame]
    
def prep_data(dataset):
    for key, odor_trace in dataset.odor_traces.items():
        calc_peak_of_odor_trace(odor_trace)
        
def save_dataset(dataset, name):
    fd = open(name, 'w')
    pickle.dump(dataset, fd)
    fd.close()
    
    
def calc_windspeed(dataset, axis=0, position=[0,.165], max_error=0.001):
    keys = get_traces_along_axis(dataset, axis, position, max_error)
    
    peak_times = []
    positions = []
    
    for key in keys:
        odor_trace = dataset.odor_traces[key]
        peak_times.append(odor_trace.peak_time)
        positions.append(odor_trace.position[axis])
        
    peak_times = np.array(peak_times)
    positions = np.array(positions)
    
    slope, intercept = fit_data.linear_fit(positions, peak_times, plot=True)
    
    dataset.windspeed = slope
    
    return positions, peak_times
    
    
    
    
    
    
def plot_odor_traces(dataset, axis=0, position=[0,.165], max_error=0.001):
    keys = get_traces_along_axis(dataset, axis, position, max_error)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for key in keys:
        odor_trace = dataset.odor_traces[key]
        ax.plot(odor_trace.timestamps, odor_trace.trace)
        
    plt.show()
    
    
    
    
