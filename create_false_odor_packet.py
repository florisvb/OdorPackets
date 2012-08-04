#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import odor_dataset as od

def make_gaussian(size, center, fwhm = 20):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[0], 1, np.float32)
    y = np.arange(0, size[1], 1, np.float32)[:,np.newaxis]
    x0 = center[0]
    y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def mm_to_pixel(val):
    # blank function for future use
    return int(val*1000)

def make_odor_movie(wind_speed):
    
    framerate = 100.
    resolution = 1000 # 1mm
    size = [1.,0.33]
    
    n_secs = size[0] / wind_speed
    n_frames = int(n_secs*framerate)
    
    pixels = (np.array(size)*resolution).astype(int)
    
    odor_movie = np.zeros([n_frames, pixels[1], pixels[0]])
    t = 0
    dt = 1/framerate
    timestamps = []
    
    for i in range(n_frames):
        
        x_center = (0 + wind_speed*t)*resolution
        y_center = size[1]/2.*resolution
        
        center = [x_center, y_center]
        frame = make_gaussian(pixels, center, fwhm=100)
        odor_movie[i,:,:] = frame
        
        timestamps.append(t)
        t += dt
    timestamps = np.array(timestamps)
        
    return odor_movie, timestamps
    
    
def make_default_odor_movie():
    wind_speed = 0.4
    odor_movie, timestamps = make_odor_movie(wind_speed)
    return odor_movie, timestamps
    
def play_movie(odor_movie=None):
    if odor_movie is None:
        odor_movie = get_default_odor_movie()

    fig = plt.figure()

    frame = 0

    im = plt.imshow(odor_movie[frame,:,:], cmap=plt.get_cmap('jet'))

    def updatefig(*args):
        global frame
        frame += 1
        
        if frame >= odor_movie.shape[0]:
            frame = 0
            
        im.set_array(odor_movie[frame,:,:])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()
            
            
def make_false_odor_trace(odor_movie, timestamps, position):
    
    y = mm_to_pixel(position[0])
    x = mm_to_pixel(position[1])
    print x, y
    trace = odor_movie[:,x,y]
    
    odor_trace = od.Odor_Trace(position)
    odor_trace.trace = trace
    odor_trace.timestamps = timestamps
    
    return odor_trace
    

def make_false_odor_dataset(odor_movie=None, timestamps=None, positions=None):
    
    if odor_movie is None:
        odor_movie, timestamps = make_default_odor_movie()
    
    if positions is None:
        positions = [[0, .165], [.1, .165], [.2, .165], [.3, .165], [.4, .165], [.5, .165], [.6, .165], 
                     [0, .175], [.1, .175], [.2, .175], [.3, .175], [.4, .175], [.5, .175], [.6, .175],
                     [0, .155], [.1, .155], [.2, .155], [.3, .155], [.4, .155], [.5, .155], [.6, .155]]
        for position in positions:
            position = np.array(positions)
            
    odor_dataset = od.Dataset()
            
    key = 0
    for position in positions:
        odor_trace = make_false_odor_trace(odor_movie, timestamps, position)
        odor_dataset.odor_traces.setdefault(key, odor_trace)
        key += 1
    
    return odor_dataset 
       
    
        
        
        
        
        
        
        
        
