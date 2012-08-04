#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    
    for i in range(n_frames):
        
        x_center = (0 + wind_speed*t)*resolution
        y_center = size[1]/2.*resolution
        
        center = [x_center, y_center]
        frame = make_gaussian(pixels, center, fwhm=100)
        odor_movie[i,:,:] = frame
        
        t += dt
        
    return odor_movie
    
    
wind_speed = 0.4
odor_movie = make_odor_movie(wind_speed)
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
        
        
        
        
        
        
        
        
        
        
        
