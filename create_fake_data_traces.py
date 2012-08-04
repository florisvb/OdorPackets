import numpy as np
import matplotlib.pyplot as plt

def point_to_pixel(data_point, size, resolution):
    pixel = (np.asarray(data_point)*resolution + np.array([0, size[1]/2.*resolution])).astype(int) 
    return pixel    
    
    
    
class Odor_Trace:
    def __init__(self, position, trace, dt):
        self.position = position
        self.trace = trace
        self.time = np.arange(0, len(self.trace))*dt
        
class Odor_Dataset:
    def __init__(self):
        self.traces = {}
        
        
def make_odor_trace(pixel, odor_movie, dt):
    odor_trace = odor_movie[:,pixel[1], pixel[0]]
    return odor_trace 
    
def make_odor_dataset(odor_movie, size, resolution, dt):
    data_points = [[0, 0], [.1, 0], [.2, 0], [.3, 0], [.4, 0], [.5, 0], [.6, 0]]
    
    odor_dataset = Odor_Dataset()
    key = 0
    
    for data_point in data_points:
        pixel = point_to_pixel(data_point, size, resolution)
        odor_vals = make_odor_trace(pixel, odor_movie, dt)
        odor_trace = Odor_Trace( np.asarray(data_point), odor_vals, dt)
        odor_dataset.traces.setdefault(key, odor_trace)
        key += 1
    
    return odor_dataset
    
    
    
def do_stuff(odor_dataset):
    
    wind_speed = 0.4
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    # find all the traces within the range of the master trace, and shift those and overlay them to see how accurate the assumption that traces don't change too much over time is
    
    master_key = 3
    
    master_odor_trace = odor_dataset.traces[master_key]
    position = master_odor_trace.time*wind_speed
    ax.plot(position, master_odor_trace.trace)
    
    odor_trace_before = odor_dataset.traces[master_key-1]
    position = odor_trace_before.time*wind_speed
    ax.plot(position, odor_trace_before.trace)
    
        
    plt.show()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
