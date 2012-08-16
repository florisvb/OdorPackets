import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import copy

import data_fit
import odor_dataset as od
import floris_plot_lib as fpl



def play_movie_from_model(gm=None):

    fig = plt.figure()
    anim_params = {'t': 0, 'xlim': [0,1], 'ylim': [0,.33], 't_max': 3., 'dt': 0.05, 'resolution': 0.01}
    
    array, extent = gm.get_array_2d(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'])
    im = plt.imshow( array, cmap=plt.get_cmap('jet'))
    

    def updatefig(*args):
        anim_params['t'] += anim_params['dt']
        if anim_params['t'] > anim_params['t_max']:
            anim_params['t'] = 0
                        
        array, extent = gm.get_array_2d(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'])
                    
        im.set_array(array)
        return im,

    ani = animation.FuncAnimation(fig, updatefig, anim_params, interval=50, blit=True)
    plt.show()
    

def make_false_odor_trace(gm, timestamps, position):
    
    x = position[0]
    y = position[1]
    inputs = [timestamps, [x, y]] 
    trace = gm.get_val(inputs)
    
    odor_trace = od.Odor_Trace(position)
    odor_trace.trace = trace
    odor_trace.timestamps = timestamps
    
    return odor_trace
    

def make_false_odor_dataset(gm=None, timestamps=None, positions=None):
    
    if gm is None:
        parameters = {  'mean_0_intercept': 0,
                        'mean_0_slope':     .2,
                        'mean_1_intercept': 0.16,
                        'mean_1_slope':     0,
                        'std_0_intercept':  0.2,
                        'std_0_slope':      0.05,
                        'std_1_intercept':  0.05,
                        'std_1_slope':      0.02,
                        'magnitude':        1,
                        }
        gm = data_fit.models.GaussianModel2D_TimeVarying(parameters=parameters)
    
        
    if timestamps is None:
        t_max = 3
        dt = 0.002
        timestamps = np.arange(0,t_max,dt)
    
    if positions is None:
        if 0:
            positions = [[0, .165], [.1, .165], [.2, .165], [.3, .165], [.4, .165], [.5, .165], [.6, .165], 
                         [0, .175], [.1, .175], [.2, .175], [.3, .175], [.4, .175], [.5, .175], [.6, .175],
                         [0, .135], [.1, .135], [.2, .135], [.3, .135], [.4, .135], [.5, .135], [.6, .135],
                         [0, .195], [.1, .195], [.2, .195], [.3, .195], [.4, .195], [.5, .195], [.6, .195],
                         [0, .155], [.1, .155], [.2, .155], [.3, .155], [.4, .155], [.5, .155], [.6, .155]]
        if 1:
            positions = []
            x_pos = np.arange(0,1,.05).tolist()
            y_pos = np.arange(0, .33, .05).tolist()
            for i, x in enumerate(x_pos):
                for j, y in enumerate(y_pos):
                    positions.append( [x,y] )
                    print positions[-1]
        
        for position in positions:
            position = np.array(positions)
            
    odor_dataset = od.Dataset()
            
    key = 0
    for position in positions:
        odor_trace = make_false_odor_trace(gm, timestamps, position)
        odor_dataset.odor_traces.setdefault(key, odor_trace)
        key += 1
    
    return odor_dataset 
    
    
    
def fit_2d_gaussian(odor_dataset, t=0.3, plot=False):
    #od.calc_windspeed(odor_dataset, position=[0, 0.16])
    
    # guess for center:
    #x0_guess = t*odor_dataset.windspeed + 0        
    x0_guess = 0
    y0_guess = 0.165
        
    x = []
    y = []
    odor = []
    for key, odor_trace in odor_dataset.odor_traces.items():
        x.append( odor_trace.position[0] )
        y.append( odor_trace.position[1] )
        index_at_t = np.argmin( np.abs(odor_trace.timestamps - t) )
        odor.append( odor_trace.trace[index_at_t] )
        
    x = np.array(x)
    y = np.array(y)
    odor = np.array(odor)   
        
    # now fit gaussian to data
    gm = data_fit.models.GaussianModel2D()
    inputs = [x,y]
    gm.fit(odor, inputs)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        array, extent = gm.get_array(xlim=[0,1], ylim=[0,.33], resolution=0.001)
        ax.imshow(array, extent=extent)
        
        fpl.scatter(ax, x, y, color=odor, radius=.003, edgecolors='black')
    
    return gm
    
    
def fit_2d_gaussian_moving(odor_dataset):
    
    t_list = np.arange(0, 2, .1)
    
    gm_list = []
    for t in t_list:
        gm = fit_2d_gaussian(odor_dataset, t=t, plot=False)
        gm_list.append(gm)
        
    mean_0_list = np.zeros_like(t_list)
    std_0_list = np.zeros_like(t_list)
    mean_1_list = np.zeros_like(t_list)
    std_1_list = np.zeros_like(t_list)
    magnitude_list = np.zeros_like(t_list)
    for i, gm in enumerate(gm_list):
        mean_0_list[i] = gm.parameters['mean_0']
        std_0_list[i] = gm.parameters['std_0']
        mean_1_list[i] = gm.parameters['mean_1']
        std_1_list[i] = gm.parameters['std_1']
        magnitude_list[i] = gm.parameters['magnitude']
        
    parameter_list = [mean_0_list, std_0_list, mean_1_list, std_1_list, magnitude_list]
    lm_list = []
    
    
    for i, param in enumerate(parameter_list):
        lm = data_fit.models.LinearModel(parameters={'slope': 1, 'intercept': 0})
        print parameter_list[i]
        lm.fit(parameter_list[i], t_list)
        print lm.parameters
        print
        lm_list.append(copy.copy(lm))
    
        
        
    return gm_list, parameter_list, lm_list
    
    
    
def fit_2d_gaussian_moving_builtin(odor_dataset):
    
    timestamps = np.arange(0, 2, .1)
    
    def get_positions(odor_dataset):
        x = []
        y = []
        for key, odor_trace in odor_dataset.odor_traces.items():
            x.append( odor_trace.position[0] )
            y.append( odor_trace.position[1] )
        return [np.array(x), np.array(y)]        
    
    def get_odor(timestamps, odor_dataset):
        odor = []
        for t in timestamps:
            odor_at_t = []
            for key, odor_trace in odor_dataset.odor_traces.items():
                index_at_t = np.argmin( np.abs(odor_trace.timestamps - t) )
                odor_at_t.append( odor_trace.trace[index_at_t] )
            odor.append(np.array(odor_at_t))
        return odor
        
    positions = get_positions(odor_dataset)
    odor = get_odor(timestamps, odor_dataset)
            
    gm = data_fit.models.GaussianModel2D_TimeVarying()
    gm.fit(timestamps, odor, positions)
    
    return gm    
    
