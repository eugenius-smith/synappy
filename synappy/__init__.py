# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:51:03 2016

@author: michaellynn
"" SynapPy is a rapid data visualization and quantification tool for single-cell electrophysiologists.


It takes .abf files as inputs, detects poststim synaptic events, 
and automatically computes a variety of statistics on them including:
    .height
    .height_norm
    .latency             [by max_height, max_slope, 80_20_line (calcuate epsp foot)]
    .decay
    .baseline
    .failure rate

        
The raw values are stored as attributes in the synwrapper object
(eg event_stats.height_norm[neuron] gives arrays of norm heights for each [neuron])

    
It also intelligently filters out bad data if events are not above 4*SD(baseline),
or if their decay constant (tau) is nonsensical. These events are masked but kept 
in the underlying data structure.


The main dependencies are: numpy, scipy, matplotlib and neo (ver 0.4+ recommended)




---Typical usage---:

    import synappy as syn

    event1 = syn.load(['15d20004.abf', '15d20007.abf', '15d20020.abf'])
     
            #Give a list of filenames as first argument
            #can also specify trial ranges [default:all], input channels [default:first]
            #stim channels [default:last] and a downsampling ratio for analog signals [default:2] 
            #(this last property to help with rapid analysis)
     
     
    event1.add_all(event_direction = 'down', latency_method = 'max_slope') 
    
            #automatically adds all relevant stats. Many options here to change stat properties.
            #Note: includes filtering out of unclamped aps; and filtering out of events with nonsensical decay 
    
    
    event1.height_norm[neuron]
    
            #fetch normalized height stats for that neuron. dim0 = trials, dim1 = stims.
            #The raw data behind each attribute can be fetched this way.



---Plotting tools---:

    event1.plot('height')  
    
            #main data visualization tool. Plots event attribute.  
            #Makes a separate figure for each neuron, then plots stim_num on x-axis and attribute on y-axis.
            #plots are color-coded (blue are early trials, red are late trials, grey are fails)
    
    
    event1.plot_corr('height, 'decay')
    
            #plots correlation between two attributes within event1.
            #same format/coloring as event1.plot.
    
    
    syn.plot_events(event1.attribute, event2.attribute)
    
            #compare attributes from two data groups (different conditions, cell types, etc.)
    
    
    syn.plot_statwrappers(stat1, stat2, ind = 0, err_ind = 2)
    
            #compare statfiles on two events, and can also give dim1indices of statfiles to plot.
            #eg to plot means +-sterr, syn.plot_statwrappers(stat1, stat2, ind = 0, err_ind = 2)
    


---Useful built-in general functions---:

    syn.pool(event_1.attribute)
    
        #Pools this attribute over [stims, :]
    
    
    syn.get_stats(event_1.attribute, byneuron = False)
    
        #Gets statistics for this attribute over stimtrain (out[stim,:]) or neuron if byneuron is True (out[neuron,:])
        #dim2: 1->mean, 2->std, 3->sterr, 4->success_rate_mean, 5->success_rate_stdev
        #eg if byneuron = True, out[neuron, 3] would give sterr for that neuron, calculated by pooling across all trials/stims.
    

---Useful methods on the synwrapper class---:
    synwrapper.propagate_mask(): propagate synwrapper.mask through to all other attributes.
    synwrapper.add_heights() adds .height and .latency
    synwrapper.add_sorting() adds .mask and sorts [.height, .latency]
    synwrapper.add_invertedsort() adds .height_fails
    synwrapper.add_normalized() adds .height_norm
    synwrapper.add_decay() adds .decay
    synwrapper.remove_unclamped_aps() identifies events higher than 5x (default) amp.
                                    It updates the .mask with this information, and then .propagate_mask()
    
    synwrapper.add_all() is a general tool to load all stats.
   
   
   
   
~~~~~~~Data filtering and manipulation~~~~~~
    
Data is automatically filtered in two ways:
    1) height for each event must be above 4*std(baseline) for that trial.
    2) decays must not be nonsensical (tau > 0, tau not way larger than rest of set)
        - This decay-based mask update can be turned off during synwrapper.add_decay()
    
    Success/fail filter is stored in a global synwrapper.mask attribute 
    that can be propagated to all other attributes via the synwrapper.propagate_mask() method.

"""

import neo
import numpy as np
import scipy as sp
import scipy.signal as sp_signal
import scipy.stats as sp_stats
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt



###Define synwrapper: A wrapper for synaptic event attributes (eg height, latency) from one dataset.
class synwrapper(object):
    def __init__(self):
        pass
    
    def plot(self, arr_name, yax = False, ylim = False,
                              by_neuron = False, ind = 0, hist = False):
        
        arr = self.__getattribute__(arr_name)
        num_neurons = len(arr)
    
        if 'height' is arr_name:
            yax = 'Event amplitude (pA)'
        elif 'height_norm' is arr_name:
            yax = 'Normalized event amplitude'
        elif 'baseline' is arr_name:
            yax = 'Baseline holding current (pA)'
        elif 'decay' is arr_name:
            yax = 'tau (s)'
            hist = True
        elif 'latency' is arr_name:
            yax = 'Latency (s)'
                        
            
        if yax  is False:
            yax = ' '
            
        if hist is False:
    
            for neuron in range(num_neurons):
                print('\nNeuron: ', neuron)
                x_1 = range(1, len(arr[neuron][0,:,0]) + 1)
                num_trials = len(arr[neuron][:,0,0])
            
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                plt.hold(True)
                for trial in range(num_trials):
                    ratio_thistrial = trial / (num_trials)
                    red_thistrial = 1 / (1 + np.exp( -5 * (ratio_thistrial - 0.5)))
                    color_thistrial = [red_thistrial, 0.2, 1 - red_thistrial]
                    if type(arr[neuron]) is np.ma.core.MaskedArray:             
                        ax.plot(x_1, arr[neuron][trial, :, ind].filled(np.nan),'.', color = color_thistrial, alpha = 0.6)
                        ma = arr[neuron][trial, :, ind].mask
                        inv_ma = np.logical_not(ma)
                        new_pse = np.ma.array(np.array(arr[neuron][trial,:,ind]), mask = inv_ma)
                        ax.plot(x_1, new_pse.filled(np.nan),'.', color = '0.7', alpha = 0.6)
                    else: 
                        ax.plot(x_1, arr[neuron][trial,:,ind],'.', color = color_thistrial, alpha = 0.6)
                        
                
                ax.set_xlabel('Stimulation number')
                ax.set_ylabel(yax)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                xlim_curr = ax.get_xlim()
                ylim_curr = ax.get_ylim()                
                ax.set_xlim([xlim_curr[0], xlim_curr[1] + 1])
                if arr_name is 'latency' or arr_name is 'height_norm':
                    ax.set_ylim([0, ylim_curr[1]])
                
                

                
                if ylim is not False:
                    ax.set_ylim(ylim)
                #name = name_5ht + '_' + name_gaba + '_' name_freq + '.jpg'
                plt.hold(False)
                plt.show()
                
        elif hist is True:
            
            for neuron in range(num_neurons):
                
                print('\nNeuron: ', neuron)
                num_trials = len(arr[neuron][:,0,0])
                array_to_plot = arr[neuron][:, :, ind]
                            
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)                
                plt.hold(True)
                if type(arr[neuron]) is np.ma.core.MaskedArray:   
                    histpool_thisneuron = array_to_plot.compressed()                    
                    ax.hist(histpool_thisneuron, bins = 30, facecolor = [0.2, 0.4, 0.8], normed = True, alpha = 0.6, linewidth = 0.5)
                else: 
                    histpool_thisneuron = array_to_plot.flatten()    
                    ax.hist(histpool_thisneuron, n = 30, facecolor = [0.2, 0.4, 0.8], normed = True, alpha = 0.6, linewidth = 0.5)
                        
                
                ax.set_xlabel('Decay (tau) (s)')
                ax.set_ylabel('Number')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                xlim_curr = ax.get_xlim()
                

                plt.hold(False)
                plt.show()
            
            
    def plot_corr(self, arr_name, arr_name2, ind1 = 0, ind2 = 0, xlabel = False, ylabel = False):
        arr1 = self.__getattribute__(arr_name)
        arr2 = self.__getattribute__(arr_name2)
        
        if xlabel is False:
            xlabel = arr_name
        if ylabel is False:
            ylabel = arr_name2
        
        num_neurons = len(arr1)        

        for neuron in range(num_neurons):
            print('\nNeuron: ', neuron)
            
            plt.figure()
            plt.hold(True)
            num_trials = arr1[neuron].shape[0]
            for trial in range(num_trials):
                ratio_thistrial = trial / num_trials
                red_thistrial = 1 / (1 + np.exp( -5 * (ratio_thistrial - 0.5)))
                color_thistrial = [red_thistrial, 0.2, 1 - red_thistrial]

                if type(arr1[neuron]) is np.ma.core.MaskedArray or type(arr2[neuron]) is np.ma.core.MaskedArray:
                    plt.plot(arr1[neuron][trial,:,ind1].filled(np.nan), arr2[neuron][trial,:,ind2].filled(np.nan), '.', color = color_thistrial) 
                      
                    inv_ma = np.logical_not(arr1[neuron][trial, :, ind1].mask)
                    new_arr1 = np.ma.array(np.array(arr1[neuron][trial, :, ind1]), mask = inv_ma)
                    new_arr2 = np.ma.array(np.array(arr2[neuron][trial, :, ind2]), mask = inv_ma)                    
                    plt.plot(new_arr1.filled(np.nan), new_arr2.filled(np.nan),'.', color = '0.6')
                                
                else:
                    plt.plot(arr1[neuron][trial,:,ind1], arr2[neuron][trial,:,ind2], '.r')
                
                plt.xlabel(arr_name)
                plt.ylabel(arr_name2)   
            plt.show()

                
            
    def add_heights(self, event_direction = 'up',
                                       baseline_lower = 4, baseline_upper = 0.2, 
                                       PSE_search_lower = 5, PSE_search_upper = 30,
                                       smoothing_width = False, latency_method = 'max_height'):
                                           
        analog_signals = self.analog_signals
        stim_on = self.stim_on
        times = self.times                       
                                                       
        num_neurons = len(analog_signals)
        baseline = np.empty(num_neurons, dtype = np.ndarray)
            
        #Determine direction of postsynaptic events
        if event_direction is 'up' :
            event_direction = 1
        if event_direction is 'down':
            event_direction = -1
                 
        #postsynaptic_event stores data for each post-synaptic event     
        postsynaptic_event = np.empty(num_neurons, dtype = np.ndarray)
        postsynaptic_event_latency = np.empty(num_neurons, dtype = np.ndarray)
    
        for neuron in range(num_neurons):
            sample_rate = np.int32(np.round(1 / (times[neuron][1] - times[neuron][0])))   
            if smoothing_width == False:
                smoothing_width = 2
                smoothing_width_ind = np.int32(smoothing_width * sample_rate / 1000) + 1
            else:
                smoothing_width_ind = np.int32(smoothing_width * (sample_rate / 1000)) + 1
    
                                                   
            num_trials = len(analog_signals[neuron])  
            num_stims = len(stim_on[neuron])
    
            #convert baseline and PSE search bounds to indices                                  
            baseline_lower_index = np.int32(baseline_lower * sample_rate / 1000)
            baseline_upper_index = np.int32(baseline_upper * sample_rate / 1000)
        
            PSE_search_lower_index = np.int32(PSE_search_lower * sample_rate / 1000)
            PSE_search_upper_index = np.int32(PSE_search_upper * sample_rate / 1000)     
    
            baseline[neuron] = np.empty([num_trials, num_stims, 2], dtype = np.ndarray)
            postsynaptic_event[neuron] = np.empty([num_trials, num_stims, 4], dtype = np.ndarray)
            postsynaptic_event_latency[neuron] = np.empty([num_trials, num_stims, 2], dtype = np.ndarray)
            
    
            #postsynaptic_event[neuron][trial, stim, b], 
                #b = 0: index of max, b = 1: val of max, b = 2: normalized val of max
                #b = 3: latency
    
            for stim in range(num_stims):   
                baseline_lower_thistrial = np.int32(stim_on[neuron][stim] - baseline_lower_index)
                baseline_upper_thistrial = np.int32(stim_on[neuron][stim] - baseline_upper_index)
                
                PSE_search_lower_thistrial = np.int32(stim_on[neuron][stim] + PSE_search_lower_index)
                PSE_search_upper_thistrial = np.int32(stim_on[neuron][stim] + PSE_search_upper_index) 
        
                #calculate mean baseline for this trial, stim. Store in baseline[neuron][trial][stim, 0]
                #baseline[neuron][trial][stim, 1] stores stdev.
                for trial in range(num_trials):
        
                    baseline[neuron][trial, stim, 0] = np.mean(analog_signals[neuron][trial, baseline_lower_thistrial:baseline_upper_thistrial])
                    baseline[neuron][trial, stim, 1] = np.std(analog_signals[neuron][trial, baseline_lower_thistrial:baseline_upper_thistrial])              
                
                #Use boxcar-moving-avg to smooth analog signal. Calculate this in analog_smoothed
                #and the derivative in 
                
                analog_presmoothed_input = analog_signals[neuron][:, PSE_search_lower_thistrial:PSE_search_upper_thistrial]              
                analog_smoothed = sp_signal.savgol_filter(analog_presmoothed_input, smoothing_width_ind, 3)
            
                #calculate max PSE height [stim,0] and its index [stim,1] for this trial, stim
                if event_direction == 1:
                    postsynaptic_event[neuron][:, stim, 1] = np.argmax(analog_smoothed, axis = -1)
                elif event_direction == -1:
                    postsynaptic_event[neuron][:, stim, 1] = np.argmin(analog_smoothed, axis = -1)
                #correct index back to analog_signal reference
                postsynaptic_event[neuron][:, stim, 1] += PSE_search_lower_thistrial                        
                postsynaptic_event[neuron][:, stim, 0] = [analog_signals[neuron][i, np.int32(postsynaptic_event[neuron][i,stim,1])] for i in range(num_trials)]
                postsynaptic_event[neuron][:, stim, 0] -=  baseline[neuron][:, stim, 0]      #correct EPSP val by subtracting baseline measurement               
                #store time of max_height latency in [stim,2]
                postsynaptic_event[neuron][:, stim, 2] =  [times[neuron][np.int32(postsynaptic_event[neuron][trial][stim,1])] - times[neuron][np.int32(stim_on[neuron][stim])] for trial in range(num_trials)]
        
                #derivative calcs. Go to trial indexing due to uneven size of arays from stim-on to max-height.
                for trial in range(num_trials):
                    max_height_smoothed_ind = np.int32(postsynaptic_event[neuron][trial,stim,1] - PSE_search_lower_thistrial)

                    if max_height_smoothed_ind < 2:
                        max_height_smoothed_ind = 2
                    analog_smoothed_deriv = np.gradient(analog_smoothed[trial, 0:max_height_smoothed_ind])
        
                    if event_direction == 1:
                        max_deriv_ind = np.argmax(analog_smoothed_deriv)
                        postsynaptic_event[neuron][:, stim, 3] = analog_smoothed_deriv[max_deriv_ind] * (sample_rate/1000)                   
                    elif event_direction == -1:
                        max_deriv_ind = np.argmin(analog_smoothed_deriv)
                        postsynaptic_event[neuron][:, stim, 3] = analog_smoothed_deriv[max_deriv_ind] * (sample_rate/1000)                   
            
            
                    #Based on latency_method, determine latency and store in postsynaptic_event_latency
                    if latency_method == 'max_height': 
                        event_time_index = postsynaptic_event[neuron][trial, stim, 1]
                        stim_time_index = stim_on[neuron][stim]
                        
                        postsynaptic_event_latency[neuron][trial, stim,0] =  times[neuron][event_time_index] - times[neuron][stim_time_index]
                        postsynaptic_event_latency[neuron][trial, stim,1] = postsynaptic_event[neuron][trial, stim, 1]
            
                    elif latency_method == 'max_slope':     
                        event_time_index = np.int32(max_deriv_ind + PSE_search_lower_thistrial)
                        stim_time_index = stim_on[neuron][stim]
            
                        postsynaptic_event_latency[neuron][trial, stim, 0] = times[neuron][event_time_index] - times[neuron][stim_time_index]               
                        postsynaptic_event_latency[neuron][trial, stim, 1] = event_time_index                  
                        
                    elif latency_method == 'baseline_plus_4sd':                 
                        signal_base_diff = ((analog_smoothed[trial,0:max_height_smoothed_ind] - (baseline[neuron][trial, stim, 0] + 4 * baseline[neuron][trial, stim, 1])) ** 2 ) > 0
                        signal_base_min_ind = find_last(signal_base_diff, tofind = 0)
                        postsynaptic_event_latency[neuron][trial, stim,0] = times[neuron][signal_base_min_ind + PSE_search_lower_index]
                        postsynaptic_event_latency[neuron][trial, stim,1] = signal_base_min_ind + PSE_search_lower_index
                              
                    elif latency_method == '80_20_line':
                        value_80pc = 0.8 * (postsynaptic_event[neuron][trial,stim,0]) + baseline[neuron][trial,stim,0]         
                        value_20pc = 0.2 * (postsynaptic_event[neuron][trial,stim,0]) + baseline[neuron][trial,stim,0]         
                        value_80pc_sizeanalog =  value_80pc * np.ones(len(analog_smoothed[trial, 0:max_height_smoothed_ind]))                   
                        value_20pc_sizeanalog =  value_20pc * np.ones(len(analog_smoothed[trial, 0:max_height_smoothed_ind]))                   
                 
#                        diff_80pc = (analog_smoothed[trial, 0:max_height_smoothed_ind] - value_80pc_sizeanalog) > 0
#                        diff_20pc = (analog_smoothed[trial, 0:max_height_smoothed_ind] - value_20pc_sizeanalog) > 0
                        diff_80pc = (analog_presmoothed_input[trial, 0:max_height_smoothed_ind] - value_80pc_sizeanalog) > 0
                        diff_20pc = (analog_presmoothed_input[trial, 0:max_height_smoothed_ind] - value_20pc_sizeanalog) > 0

                        
                        if event_direction is 1:
                            ind_80cross = find_last(diff_80pc, tofind = 0)
                            ind_20cross = find_last(diff_20pc, tofind = 0)
                        elif event_direction is -1:
                            ind_80cross = find_last(diff_80pc, tofind = 1)
                            ind_20cross = find_last(diff_20pc, tofind = 1)
                                                    
                        if ind_20cross > ind_80cross or ind_80cross == 0:                   
                            ind_80cross = np.int32(1)
                            ind_20cross = np.int32(0)                        
                        
                        val_80cross = analog_smoothed[trial, ind_80cross]                    
                        val_20cross = analog_smoothed[trial, ind_20cross]
                        
                        
                        slope_8020_line = (val_80cross - val_20cross) / (ind_80cross - ind_20cross)
                        
                        vals_8020_line = np.zeros(len(analog_smoothed[trial, 0:ind_80cross + 1]))
                        vals_8020_line = [(val_80cross - (ind_80cross - i)*slope_8020_line) for i in range(ind_80cross)]
                        
                        vals_baseline = baseline[neuron][trial,stim,0] * np.ones(len(analog_smoothed[trial, 0:ind_80cross]))
                        #diff_sq_8020_line = (vals_baseline - vals_8020_line) ** 2 + (analog_smoothed[trial, 0:ind_80cross] - vals_8020_line) ** 2
                        diff_sq_8020_line = (vals_baseline - vals_8020_line) ** 2 + (analog_presmoothed_input[trial, 0:ind_80cross] - vals_8020_line) ** 2
                                                
                        intercept_8020_ind = np.argmin(diff_sq_8020_line)
                        
                        event_time_index = intercept_8020_ind + PSE_search_lower_thistrial
                        stim_time_index = stim_on[neuron][stim]
                        postsynaptic_event_latency[neuron][trial,stim,0] = times[neuron][event_time_index] - times[neuron][stim_time_index]               
                        postsynaptic_event_latency[neuron][trial,stim,1] = event_time_index                  
                                
        self.height =  postsynaptic_event 
        self.latency = postsynaptic_event_latency
        self.baseline = baseline
#        self.upslope = 
        
        print('\nAdded height. \nAdded latency. \nAdded baseline.')              
                                                       
                                               
        return
        
        
        
    def add_sorting(self, thresh = False, thresh_dir = False):
        postsynaptic_event = self.height
        postsynaptic_event_latency = self.latency
        baseline = self.baseline
        
        num_neurons = len(postsynaptic_event)
        
        self.mask = np.empty(num_neurons, dtype = np.ndarray)
    
              
        postsynaptic_event_successes = np.empty(num_neurons, dtype = np.ndarray)
        postsynaptic_event_latency_successes = np.empty(num_neurons, dtype = np.ndarray)   
        
        dynamic_thresholding = False
        if thresh == False:
            dynamic_thresholding = True
        
        for neuron in range(num_neurons):
            postsynaptic_event_successes[neuron] = np.ma.array(np.copy(postsynaptic_event[neuron]))
            postsynaptic_event_latency_successes[neuron] = np.ma.array(np.copy(postsynaptic_event_latency[neuron]))
    
            height_tocompare = np.abs(postsynaptic_event[neuron][:, :, 0])
            
            if dynamic_thresholding is True:
                thresh_tocompare = 4 * baseline[neuron][:,:,1]
            else:
                thresh_tocompare = np.abs(thresh) * np.ones_like(baseline[neuron][:,:,1])
                
            diff_tocompare = height_tocompare - thresh_tocompare  
            if thresh_dir is False:
                mask_tocompare = diff_tocompare < 0 
            elif thresh_dir is True:
                mask_tocompare = diff_tocompare > 0
            
            mask_tocompare_full_pes = np.ma.empty([mask_tocompare.shape[0], mask_tocompare.shape[1], postsynaptic_event[neuron].shape[2]])
            for shape_3d in range(postsynaptic_event[neuron].shape[2]):
                mask_tocompare_full_pes[:,:,shape_3d] = mask_tocompare
                
            mask_tocompare_full_lat = np.ma.empty([mask_tocompare.shape[0], mask_tocompare.shape[1], postsynaptic_event_latency[neuron].shape[2]])
            for shape_3d in range(postsynaptic_event_latency[neuron].shape[2]):
                mask_tocompare_full_lat[:,:,shape_3d] = mask_tocompare
          
       
            postsynaptic_event_successes[neuron].mask = mask_tocompare_full_pes
            postsynaptic_event_latency_successes[neuron].mask = mask_tocompare_full_lat
            
            self.height[neuron] = np.ma.masked_array(self.height[neuron], mask = mask_tocompare_full_pes)
            self.latency[neuron] = np.ma.masked_array(self.latency[neuron], mask = mask_tocompare_full_pes[:,:,0:2])
            self.mask[neuron] = mask_tocompare_full_pes[:,:,0]
            self.baseline[neuron] = np.ma.masked_array(self.baseline[neuron], mask = mask_tocompare_full_pes[:,:,0:2])

        print('\nAdded succ/fail sorting to: \n\theight \n\tlatency, \n\tbaseline')              
                                                                                               
        return
    
     
    def add_invertedsort(self):
        postsynaptic_event_successes = self.height
        num_neurons = len(postsynaptic_event_successes)
        
        self.height_fails = np.empty(num_neurons, dtype = np.ndarray)
    
        
        postsynaptic_event_failures = np.empty(num_neurons, dtype = np.ndarray)
        
        for neuron in range(num_neurons):
            success_mask = np.ma.getmask(postsynaptic_event_successes[neuron])
            failure_mask = np.logical_not(success_mask)
                
            postsynaptic_event_failures[neuron] = np.ma.array(np.copy(postsynaptic_event_successes[neuron]), mask = failure_mask)
            self.height_fails[neuron] = np.ma.array(np.copy(postsynaptic_event_successes[neuron]), mask = failure_mask)
    
        #print('\nAdded height_fails.')              

        return
             
    def add_normalized(self):
         postsynaptic_event = self.height
         
         num_neurons = len(postsynaptic_event)
         
         postsynaptic_event_normalized = np.empty(num_neurons, dtype = np.ndarray)
         self.height_norm = np.empty(num_neurons, dtype = np.ndarray)
         avg_ampl = np.empty(num_neurons)
         
         for neuron in range(num_neurons):
             postsynaptic_event_normalized[neuron] = np.ma.copy(postsynaptic_event[neuron])
             self.height_norm[neuron] = np.ma.copy(postsynaptic_event[neuron])
    
             avg_ampl = np.mean(postsynaptic_event[neuron][:,0,0])         
             
             if type(postsynaptic_event[neuron]) is np.ma.core.MaskedArray:
                 current_neuron_mask = postsynaptic_event[neuron][:,:,0].mask
             
                 postsynaptic_event_normalized_temp = np.array(postsynaptic_event[neuron][:,:,0]) / avg_ampl
                 postsynaptic_event_normalized[neuron][:,:,0] = np.ma.array(postsynaptic_event_normalized_temp, mask = current_neuron_mask)
             else:
                 postsynaptic_event_normalized[neuron][:,:,0] /= avg_ampl
                 
             self.height_norm[neuron] = postsynaptic_event_normalized[neuron]
         
         print('\nAdded height_norm.')              
    
    
         return
         
    
    def pool(self, name, pool_index = 0, mask = 'suc'):
        if type(name) is str:
            postsynaptic_event = self.__getattribute__(name)
        else:
            postsynaptic_event = name

            
        num_neurons = len(postsynaptic_event)
    
        #Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > postsynaptic_event[neuron][:,:,:].shape[1] :
                common_stims = postsynaptic_event[neuron][:,:,:].shape[1]
            
        #Pool data     
        stimpooled_postsynaptic_events = np.ma.array(np.empty([common_stims,0]))
    
        for neuron in range(num_neurons):
            if mask is 'suc':
                stimpooled_postsynaptic_events = np.ma.append(stimpooled_postsynaptic_events, np.ma.transpose(postsynaptic_event[neuron][:,0:common_stims, pool_index]), axis = 1)
            elif mask is 'all':
                stimpooled_postsynaptic_events = np.ma.append(stimpooled_postsynaptic_events, np.ma.transpose(postsynaptic_event[neuron][:,0:common_stims, pool_index].data), axis = 1)

    
    def add_sucrate(self, byneuron = False):
        postsynaptic_event = self.mask
        
        num_neurons = len(postsynaptic_event)
        
        #Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > len(postsynaptic_event[neuron][0,:,0]):
                common_stims = len(postsynaptic_event[neuron][0,:,0])
    
        success_rate_neur = np.zeros([common_stims, num_neurons])
        success_rate = np.zeros([common_stims, 3])    
    
        for neuron in range(num_neurons):
            count_fails_temp = np.sum(postsynaptic_event[neuron][:,0:common_stims,0].mask, axis = 0)   
            count_total_temp = postsynaptic_event[neuron].shape[0]
            success_rate_neur[:, neuron] = (count_total_temp - count_fails_temp) / count_total_temp
        
        success_rate[:,0] = np.mean(success_rate_neur, axis = 1)
        success_rate[:,1] = np.std(success_rate_neur, axis = 1)
        success_rate[:,2] = np.std(success_rate_neur, axis = 1) / np.sqrt(np.sum(success_rate_neur, axis = 1))
        
        
        if byneuron is True:   
            success_rate = []
            success_rate = success_rate_neur
            
        self.success_rate = success_rate
        #print('\nAdded success_rate.') 

             
        
            
        return self
        
    def special_sucrate(self, byneuron = False):
        mask = self.mask
        
        num_neurons = len(mask)
        
        #Calculate min stims:
        common_stims = 10000
        for neuron in range(num_neurons):
            if common_stims > len(mask[neuron][0,:]):
                common_stims = len(mask[neuron][0,:])
    
        success_rate_neur = np.zeros([common_stims, num_neurons])
        success_rate = np.zeros([common_stims, 3])    
    
        for neuron in range(num_neurons):
            count_fails_temp = np.sum(mask[neuron][:,0:common_stims], axis = 0)   
            count_total_temp = mask[neuron].shape[0]
            success_rate_neur[:, neuron] = (count_total_temp - count_fails_temp) / count_total_temp
        
        success_rate[:,0] = np.mean(success_rate_neur, axis = 1)
        success_rate[:,1] = np.std(success_rate_neur, axis = 1)
        success_rate[:,2] = np.std(success_rate_neur, axis = 1) / np.sqrt(np.sum(success_rate_neur, axis = 1))
        
        
        if byneuron is True:   
            success_rate = []
            success_rate = success_rate_neur
            
        self.success_rate = success_rate
        #print('\nAdded success_rate.') 

             
        
            
        return success_rate        
    
    ###Define function ephys_summarystats which takes a postsynaptic event and outputs (mean, sd, sterr, numevents)
    def get_stats(self, arr_name, pooling_index = 0, mask = 'suc'):
        postsynaptic_event = self.__getattribute__(arr_name)
                
        stimpooled_postsynaptic_events = self.pool(postsynaptic_event, pooling_index, mask = mask)   
        if type(postsynaptic_event[0]) is np.ma.core.MaskedArray:
            success_rate = self.special_sucrate(self)   
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 5])
            
            for stim in range(num_stims):
                num_nonmasked_stims = len(stimpooled_postsynaptic_events[stim,:]) - np.ma.count_masked(stimpooled_postsynaptic_events[stim,:])        
                
                stats_postsynaptic_events[stim,0] = np.mean(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,1] = np.std(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,2] = np.std(stimpooled_postsynaptic_events[stim, :]) / np.sqrt(num_nonmasked_stims)
                stats_postsynaptic_events[stim,3] = success_rate[stim, 0]
                stats_postsynaptic_events[stim,4] = success_rate[stim, 1]
        else:
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 3])
            
            for stim in range(num_stims):
                num_nonmasked_stims = len(stimpooled_postsynaptic_events[stim,:]) - np.ma.count_masked(stimpooled_postsynaptic_events[stim,:])        
                
                stats_postsynaptic_events[stim,0] = np.mean(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,1] = np.std(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,2] = np.std(stimpooled_postsynaptic_events[stim, :]) / np.sqrt(num_nonmasked_stims)
    
            
        return (stats_postsynaptic_events)
     
    def get_median_filtered(signal, threshold=3):
        
        if type(signal) is not np.ma.core.MaskedArray:
            signal = signal.copy()
            difference = np.abs(signal - np.median(signal))
            median_difference = np.median(difference)
            if median_difference == 0:
                s = 0
            else:
                s = difference / float(median_difference)
                
            mask = s > threshold
            mask_2 = signal < 0 
            signal[mask] = 0
            signal[mask_2] = 0
            
        else:
            original_mask = signal.mask
            
            signal = np.array(signal.copy())
            difference = np.abs(signal - np.median(signal))
            median_difference = np.median(difference)
            if median_difference == 0:
                s = 0
            else:
                s = difference / float(median_difference)
            mask = s > threshold
            signal[mask] = np.median(signal)
            
            mask = s > threshold
            mask_2 = signal < 0 
            signal[mask] = 0
            signal[mask_2] = 0
            
            combined_mask_1 = np.ma.mask_or(mask, mask_2)
            combined_mask_2 = np.ma.mask_or(combined_mask_1, original_mask)
            
            signal = np.ma.array(signal, mask = combined_mask_2)
        
        return signal
     
    def propagate_mask(self):
        postsynaptic_events = self.height
        mask = self.mask
        num_neurons = len(postsynaptic_events) 
        for neuron in range(num_neurons):
            for lastind in range(postsynaptic_events[neuron].shape[2]):
                self.height[neuron].mask[:, :, lastind] = mask[neuron]
                self.height_norm[neuron].mask[:, :, lastind] = mask[neuron]
                self.height_fails[neuron].mask[:, :, lastind] = ~mask[neuron]
                
                if lastind < self.latency[neuron].shape[2]:
                    self.latency[neuron].mask[:, :, lastind] = mask[neuron]
                if lastind < self.baseline[neuron].shape[2]:
                    self.baseline[neuron].mask[:, :, lastind] = mask[neuron]
                if lastind < self.decay[neuron].shape[2]:
                    self.decay[neuron].mask[:, :, lastind] = mask[neuron]
                    
    def remove_unclamped_aps(self, thresh_ratio = 5):
        height_norm = self.height_norm
        height = self.height
        latency = self.latency
        
        num_neurons = len(height_norm)
        
        for neuron in range(num_neurons):
            to_replace = np.argwhere(height_norm[neuron][:,:,0] > thresh_ratio)

            height_norm[neuron][to_replace[:, 0], to_replace[:, 1] ,:] = np.nan
            height_norm[neuron].mask[to_replace[:, 0], to_replace[:, 1] ,:] = True

            height[neuron][to_replace[:, 0], to_replace[:, 1] ,:] = np.nan
            height[neuron].mask[to_replace[:, 0], to_replace[:, 1] ,:] = True

            latency[neuron][to_replace[:, 0], to_replace[:, 1] ,:] = np.nan
            latency[neuron].mask[to_replace[:, 0], to_replace[:, 1] ,:] = True            
            
            self.mask[neuron] = np.ma.mask_or(height_norm[neuron][:,:,0].mask, self.height[neuron][:,:,0].mask)
            
        self.propagate_mask()
        
        print('\nMasked APs')

    
    def add_decays(self, prestim = 0, poststim = 10, plotting = False, fn = 'monoexp_normalized_plusb', update_mask = False):
        
        analog_signals = self.analog_signals
        postsynaptic_events = self.height
        baseline = self.baseline
        times = self.times
            
        num_neurons = len(postsynaptic_events)
    
        def biexp_normalized_plusb(time_x, lambda1, lambda2, vstart2, b):
            y = np.exp(time_x * (-1) * lambda1) + vstart2 * np.exp(time_x * (-1) * lambda2) + b#+  np.exp(time_x * (-1) * lambda2)
            return y                 
             
        def monoexp_normalized_plusb(time_x, lambda1, b):
            y = np.exp(time_x * (-1) * lambda1) + b #+  np.exp(time_x * (-1) * lambda2)
            return y                
       
        num_vars = 2
    
        fitted_vars = np.empty(num_neurons, dtype = np.ndarray)
        fitted_covari = np.empty_like(fitted_vars)
        
        for neuron in range(num_neurons):
            sample_rate = np.int32(np.round(1 / (times[neuron][1] - times[neuron][0])))  
            poststim_ind = np.int32(poststim * sample_rate / 1000)
    
            
            num_trials = postsynaptic_events[neuron].shape[0]
            num_stims = postsynaptic_events[neuron].shape[1]
    
            if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                fitted_vars[neuron] = np.ma.array(np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray))
                fitted_vars[neuron].mask = postsynaptic_events[neuron].mask     
                fitted_covari[neuron] = np.ma.array(np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray))
                fitted_covari[neuron].mask = postsynaptic_events[neuron].mask     
    
            else:
                fitted_vars[neuron] = np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray)
                fitted_covari[neuron] = np.empty([num_trials, num_stims, num_vars], dtype = np.ndarray)
    
            
            for trial in range(num_trials):
                for stim in range(num_stims):
                    
                    if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray and  postsynaptic_events[neuron][trial, stim, 0] is not np.ma.masked: 
                        event_ind_min = postsynaptic_events[neuron][trial,stim,1] - prestim      
                        event_ind_max = event_ind_min + poststim_ind
                        
                        postsynaptic_curve = analog_signals[neuron][trial, event_ind_min : event_ind_max] - baseline[neuron][trial, stim, 0]
                        postsynaptic_curve /= np.mean(postsynaptic_curve[0:2])
        
                        vars_guess = [100, 0]
                        times_forfit = times[neuron][0:poststim_ind]
                        try:
                            [popt, pcov] = sp_opt.curve_fit(monoexp_normalized_plusb, times_forfit, postsynaptic_curve, p0 = vars_guess) #Dfun = jacob_exp)#, maxfev = 250) #p0 = [100, postsynaptic_events[neuron][trial,stim,0]]) #0.5*postsynaptic_events[neuron][trial,stim,0]])
                        except RuntimeError:
                            popt = np.ones(num_vars) * 10000
                            pcov = 10000
                        except ValueError:
                            print(postsynaptic_curve, 'neuron: ', neuron, 'trial: ', trial, 'stim: ', stim)
                        fitted_vars[neuron][trial,stim, :] = popt[:]
                    elif type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray and postsynaptic_events[neuron][trial, stim, 0] is np.ma.masked: 
                        fitted_vars[neuron][trial,stim, :] = np.ones(num_vars) * 10000
                        fitted_vars[neuron][trial,stim, :].mask = np.ones(num_vars, dtype = np.bool)
                    elif type(postsynaptic_events[neuron]) is not np.ma.core.MaskedArray:
                        event_ind_min = postsynaptic_events[neuron][trial,stim,1] - prestim      
                        event_ind_max = event_ind_min + poststim_ind                 
                        
                        postsynaptic_curve = analog_signals[neuron][trial, event_ind_min : event_ind_max] - baseline[neuron][trial, stim, 0]
                        postsynaptic_curve /= postsynaptic_curve[0]
        
                        vars_guess = [100, 0]
                        times_forfit = times[neuron][0:poststim_ind]
                        try:
                            [popt, pcov] = sp_opt.curve_fit(monoexp_normalized_plusb, times_forfit, postsynaptic_curve, p0 = vars_guess) #, Dfun = jacob_exp)#, maxfev = 250) #p0 = [100, postsynaptic_events[neuron][trial,stim,0]]) #0.5*postsynaptic_events[neuron][trial,stim,0]])
                        except RuntimeError:
                            popt = np.ones(num_vars) * 10000
                            #pcov = 10000

                            
                        fitted_vars[neuron][trial,stim, :] = popt[:]
                        
                        
            if plotting is True:
                                
                postsynaptic_curve = analog_signals[neuron][0, postsynaptic_events[neuron][0,0,1] - prestim : postsynaptic_events[neuron][0,0,1] + poststim_ind] - baseline[neuron][0, 0, 0]
                y_fitted = postsynaptic_events[neuron][0,0,0] * monoexp_normalized_plusb(times[neuron][0:poststim_ind + prestim], fitted_vars[neuron][0,0,0], fitted_vars[neuron][0,0,1])          
                plt.figure()
                plt.hold(True)
                plt.plot(times[neuron][0:poststim_ind + prestim], y_fitted, 'r')
                plt.plot(times[neuron][0:poststim_ind + prestim], postsynaptic_curve, 'b')
                  
            #convert from lambda to tau
            fitted_ones = np.ones([fitted_vars[neuron][:,:,0].shape[0], fitted_vars[neuron][:,:,0].shape[1]])
            
            if type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                fitted_vars[neuron][:,:,0] = fitted_ones / np.array(fitted_vars[neuron][:,:,0])
                fitted_vars[neuron][:,:,0] = get_median_filtered(fitted_vars[neuron][:,:,0], threshold=10)
                fittedvarmask = np.ma.mask_or(postsynaptic_events[neuron][:,:,0].mask, fitted_vars[neuron][:,:,0].mask)
                fitted_vars[neuron][:,:,0].mask = fittedvarmask
    
            else:
                fitted_vars[neuron][:,:,0] = fitted_ones / fitted_vars[neuron][:,:,0]            
                fitted_vars[neuron][:,:,0] = get_median_filtered(fitted_vars[neuron][:,:,0], threshold=10)
                
            if update_mask is True and type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
                self.mask[neuron] = np.ma.mask_or(postsynaptic_events[neuron][:,:,0].mask, fitted_vars[neuron][:,:,0].mask)
        self.decay = fitted_vars
        
        print('\nAdded decay')              
                
        if update_mask is True and type(postsynaptic_events[neuron]) is np.ma.core.MaskedArray:
            self.propagate_mask()
            print('\n\nAdded round2 of succ/fail sorting to: \n\theight \n\theight_norm, \n\tlatency, \n\tbaseline')              

        return
                
        
    def add_all(self, event_direction = 'down', latency_method = 'max_height', ap_filter = True, sort_thresh = False, keep_above = False, delay_mask_update = True, upper_search = 30):
        self.add_heights(event_direction = event_direction, latency_method = latency_method, PSE_search_upper = upper_search)
        self.add_sorting(thresh = sort_thresh, thresh_dir = keep_above)
        self.add_normalized()
        self.add_invertedsort()
        #self.add_sucrate()
        self.add_decays(update_mask = delay_mask_update)
        if ap_filter is True:
            self.remove_unclamped_aps()
        
        
        
### ---- synappy functions ----
###     core functionality is syn.find_stims and syn.load.
def find_stims(stim_signals, thresh):
    num_neurons = len(stim_signals)
    stim_on = np.empty(num_neurons, dtype = np.ndarray)
    

    #Find start of stim-on in stim_signals[neuron][:,stim_channel[neuron],:] for each trial. 
    #Store in light_on(a): a>trial number, light_on_ind(a)>time index of light-on for that trial
    for neuron in range(num_neurons):
        stim_on[neuron] = np.zeros(1)
        all_crossings = np.where(stim_signals[neuron] > thresh)[0]        
        stim_on[neuron][0] = all_crossings[0]
        
        for crossing_ind in np.arange(1,len(all_crossings)):
            if all_crossings[crossing_ind-1] != all_crossings[crossing_ind] - 1:
                stim_on[neuron] = np.append(stim_on[neuron], all_crossings[crossing_ind])                    

    return (stim_on)
    
def load(files, trials = None, input_channel = None, stim_channel = None, downsampling_ratio = 2, stim_thresh = 3):    
    print('\n\n----New Group---')

    num_neurons = len(files)
    neurons_range = np.int32(range(num_neurons))

    block = np.empty(num_neurons, dtype = np.ndarray)
    analog_signals = np.empty(num_neurons, dtype = np.ndarray)
    stim_signals= np.empty(num_neurons, dtype = np.ndarray)
    
    times = np.empty(num_neurons, dtype = np.ndarray)   

    
    block = [neo.AxonIO(filename = files[i]).read()[0] for i in neurons_range]
 
    #Check for presence of optional variables, create them if they don't exist
    if trials is None:
        trials = np.empty(num_neurons, dtype = np.ndarray)
        for neuron in range(num_neurons):
            trials[neuron] = [1, len(block[neuron].segments) + 1]
            
    if input_channel is None:
        input_channel = np.int8(np.zeros(num_neurons))
    
    if stim_channel is None:
        stim_channel = np.int8((-1) * np.ones(num_neurons))
        


    #Populate analog_signals and times from raw data in block
    for neuron in range(num_neurons):
        num_trials = trials[neuron][1] -  trials[neuron][0]
        numtimes_full = len(block[neuron].segments[0].analogsignals[0].times)  
        numtimes_wanted =  np.int32(numtimes_full /  downsampling_ratio)

        times[neuron] = np.linspace(block[neuron].segments[0].analogsignals[0].times[0].magnitude, block[neuron].segments[0].analogsignals[0].times[-1].magnitude, num = numtimes_wanted)
            
        analog_signals[neuron] = np.empty((num_trials, numtimes_wanted))
        stim_signals[neuron] = block[neuron].segments[0].analogsignals[np.int8(stim_channel[neuron])][:]        

       
        for trial_index, trial_substance in enumerate(block[neuron].segments[trials[neuron][0]-1:trials[neuron][1]-1]):
            analog_signals[neuron][trial_index,:] = sp_signal.decimate(trial_substance.analogsignals[np.int8(input_channel[neuron])][:], int(downsampling_ratio), zero_phase = True).squeeze()

    #Find stim onsets
    stim_on = find_stims(stim_signals, stim_thresh)
    for neuron in range(num_neurons):        
        stim_on[neuron] /= downsampling_ratio
        stim_on[neuron] = np.int32(stim_on[neuron])
    
    #str_name = 'postsynaptic_events_' + name
    
    synaptic_wrapper = synwrapper()
    synaptic_wrapper.analog_signals = analog_signals
    synaptic_wrapper.stim_on = stim_on
    synaptic_wrapper.times = times
    #synaptic_wrapper.name = str_name
    print('\nInitialized. \nAdded analog_signals. \nAdded stim_on. \nAdded times.')              

    
    return (synaptic_wrapper)
				

### ---- Other useful functions ----

#Find last index in an array which has value of tofind.
def find_last(arr, tofind = 1):
    for ind, n in enumerate(reversed(arr)):
        if n == tofind or ind == len(arr) - 1:
            return (len(arr) - ind)

#def jacob_exp(pars, x, y, monoexp_normalized_plusb):
#    deriv = np.empty([len(x), 2])
#    deriv[:, 0] = -1 * x * np.exp(-1 * pars[0] * x)
#    deriv[:, 1] = 1 * np.ones(len(x))
#    return deriv



#syn.pool takes an attribute and pools it across stims: out[stim,:]
def pool(synaptic_wrapper_attribute, pool_index = 0):
    #Returns a stim x m matrix where stim is stim-num and [stim,:] is raw data for that n across all neurons, trials
          
    postsynaptic_event = synaptic_wrapper_attribute
    num_neurons = len(postsynaptic_event)
    
     #Calculate min stims:
    common_stims = 10000
    for neuron in range(num_neurons):
        if common_stims > postsynaptic_event[neuron][:,:,:].shape[1] :
            common_stims = postsynaptic_event[neuron][:,:,:].shape[1]
                            
    #Pool data     
    stimpooled_postsynaptic_events = np.ma.array(np.empty([common_stims, 0]))

    for neuron in range(num_neurons):
        stimpooled_postsynaptic_events = np.ma.append(stimpooled_postsynaptic_events, np.ma.transpose(postsynaptic_event[neuron][:,0:common_stims, pool_index]), axis = 1)
     
    return stimpooled_postsynaptic_events
    
    
#syn.get_sucrate takes an attribute and returns success rate stats over the stim train
    #out[stim,0] = mean. out[stim,1] is stdev. out[stim,2] is sterr.
def get_sucrate(synaptic_wrapper_attribute, byneuron = False):
    num_neurons = len(synaptic_wrapper_attribute)
    
    #Calculate min stims:
    common_stims = 10000
    for neuron in range(num_neurons):
        if common_stims > len(synaptic_wrapper_attribute[neuron][0,:,0]):
            common_stims = len(synaptic_wrapper_attribute[neuron][0,:,0])

    success_rate_neur = np.zeros([common_stims, num_neurons])
    success_rate = np.zeros([common_stims, 3])    

    for neuron in range(num_neurons):
        count_fails_temp = np.sum(synaptic_wrapper_attribute[neuron].mask[:,0:common_stims,0], axis = 0)   
        count_total_temp = synaptic_wrapper_attribute[neuron].mask.shape[0]
        success_rate_neur[:, neuron] = (count_total_temp - count_fails_temp) / count_total_temp
    
    success_rate[:,0] = np.mean(success_rate_neur, axis = 1)
    success_rate[:,1] = np.std(success_rate_neur, axis = 1)
    success_rate[:,2] = np.std(success_rate_neur, axis = 1) / np.sqrt(np.sum(success_rate_neur, axis = 1))
    
    
    if byneuron is True:   
        success_rate = []
        success_rate = success_rate_neur
                
    return success_rate

    
###syn.get_sucrate takes an attribute and returns general stats over dim0 = stim# (or dim0 = neuron# if byneuron is True)
    #out[:,0] = mean. out[:,1] is stdev. out[:,2] is sterr. . out[:,3] is success_rate mean. out[:,4] is success_rate stdev
    
def get_stats(synaptic_wrapper_attribute, pooling_index = 0, byneuron = False):
    #If byneuron = False: get_stats returns a (stim x 5) array where stim is the stim number and [stim,0:4] is mean, std, sterr, mean success_rate, std success_rate.
    #If byneuron = True: get_stats returns a (neur x 3) array where neur is the neuron number and [neur,0:2] is mean, std, median.
    
    postsynaptic_event = synaptic_wrapper_attribute
    
    if byneuron is False:
        stimpooled_postsynaptic_events = pool(postsynaptic_event, pooling_index)   
        if type(postsynaptic_event[0]) is np.ma.core.MaskedArray:
            success_rate = get_sucrate(postsynaptic_event)   
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 5])
            
            for stim in range(num_stims):
                num_nonmasked_stims = len(stimpooled_postsynaptic_events[stim,:]) - np.ma.count_masked(stimpooled_postsynaptic_events[stim,:])        
                
                stats_postsynaptic_events[stim,0] = np.mean(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,1] = np.std(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,2] = np.std(stimpooled_postsynaptic_events[stim, :]) / np.sqrt(num_nonmasked_stims)
                stats_postsynaptic_events[stim,3] = success_rate[stim, 0]
                stats_postsynaptic_events[stim,4] = success_rate[stim, 1]
        else:
            num_stims = len(stimpooled_postsynaptic_events)
            stats_postsynaptic_events = np.zeros([num_stims, 3])
            
            for stim in range(num_stims):
                num_nonmasked_stims = len(stimpooled_postsynaptic_events[stim,:]) - np.ma.count_masked(stimpooled_postsynaptic_events[stim,:])        
                
                stats_postsynaptic_events[stim,0] = np.mean(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,1] = np.std(stimpooled_postsynaptic_events[stim, :])
                stats_postsynaptic_events[stim,2] = np.std(stimpooled_postsynaptic_events[stim, :]) / np.sqrt(num_nonmasked_stims)
    
    elif byneuron is True:
        num_neurons = len(postsynaptic_event)
        stats_postsynaptic_events = np.zeros([num_neurons, 3])
        
        for neuron in range(num_neurons):
            stats_postsynaptic_events[neuron,0] = np.mean(postsynaptic_event[neuron][:, :, pooling_index].flatten())
            stats_postsynaptic_events[neuron,1] = np.std(postsynaptic_event[neuron][:, :, pooling_index].flatten())
            stats_postsynaptic_events[neuron,2] = np.median(np.ma.compressed(postsynaptic_event[neuron][:, :, pooling_index].flatten()))
            
            
        
        
    return (stats_postsynaptic_events)
  
def get_median_filtered(signal, threshold=3):
    
    if type(signal) is not np.ma.core.MaskedArray:
        signal = signal.copy()
        difference = np.abs(signal - np.median(signal))
        median_difference = np.median(difference)
        if median_difference == 0:
            s = 0
        else:
            s = difference / float(median_difference)
            
        mask = s > threshold
        mask_2 = signal < 0 
        signal[mask] = 0
        signal[mask_2] = 0
        
    else:
        original_mask = signal.mask
        
        signal = np.array(signal.copy())
        difference = np.abs(signal - np.median(signal))
        median_difference = np.median(difference)
        if median_difference == 0:
            s = 0
        else:
            s = difference / float(median_difference)
        mask = s > threshold
        signal[mask] = np.median(signal)
        
        mask = s > threshold
        mask_2 = signal < 0 
        signal[mask] = 0
        signal[mask_2] = 0
        
        combined_mask_1 = np.ma.mask_or(mask, mask_2)
        combined_mask_2 = np.ma.mask_or(combined_mask_1, original_mask)
        
        signal = np.ma.array(signal, mask = combined_mask_2)
    
    return signal
 
 

    
###----- Advanced plotting tools to compare groups of recordings ----
    
    #plot_events compares two synwrapper attributes (eg group1.height, group2.height)
    
    #plot_statwrapper compares two stat files on attributes. For ex, to plot means +- standard error:
    #stat1 = syn.get_stats(group1.height), stat2 = syn.get_stats(group2.height), syn.plot_statwrapper(stat1, stat2, ind = 0, err_ind = 2)
    
def plot_events(postsynaptic_events_1, postsynaptic_events_2, name1 = 'Group 1', name2 = 'Group 2', 
                          hz = ' ', ylabel = False, ind = 0, err_ind = 1, ylimmin = True,
                          by_neuron = False, pool = False):
                              
    if type(hz) is not str:
        hz = str(hz)
                              
                              
    stats_postsynaptic_events_1 = get_stats(postsynaptic_events_1)
    stats_postsynaptic_events_2 = get_stats(postsynaptic_events_2)
    
#    if pool is False:
    
    x_1 = range(1, len(stats_postsynaptic_events_1) + 1)
    x_2 = range(1, len(stats_postsynaptic_events_2) + 1)
    
        
    if ylabel  is False:
        ylabel = 'Normalized current amplitude'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.hold(True)
    ax.plot(x_1, stats_postsynaptic_events_1[:, ind], color = 'g', linewidth = 2)
    plt.plot(x_2, stats_postsynaptic_events_2[:, ind],color = 'r', linewidth = 2)
    
    ax.fill_between(x_1, stats_postsynaptic_events_1[:, ind] - stats_postsynaptic_events_1[:, err_ind], 
           stats_postsynaptic_events_1[:,ind] + stats_postsynaptic_events_1[:,err_ind], alpha=0.2, facecolor='g', linewidth = 0)
    ax.fill_between(x_2, stats_postsynaptic_events_2[:,ind] - stats_postsynaptic_events_2[:, err_ind], 
           stats_postsynaptic_events_2[:,ind] + stats_postsynaptic_events_2[:, err_ind],alpha=0.2,facecolor='r', linewidth = 0)
    
    ax.set_xlabel('Stimulation number (' + hz + ')')
    ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend([name1, name2],frameon = False , loc = 1)

    ylim_curr = ax.get_ylim()                
    if ylimmin is True:
        ax.set_ylim([0, ylim_curr[1]])
        
    name = 'stimtrain_' + name1 + '_' + name2 + '_' + hz + '_' + ylabel + '.jpg'
    plt.savefig(name, dpi = 800)
    plt.hold(False)


    
def plot_statwrappers(stats_postsynaptic_events_1, stats_postsynaptic_events_2, name1 = 'Group 1', name2 = 'Group 2', 
                          hz = ' ', ylabel = False, xlabel = False, ind = 0, err_ind = 1, ylimmin = False,
                          by_neuron = False, save = True):
                              
    if type(hz) is not str:
        hz = str(hz)
        
    x_1 = range(1, len(stats_postsynaptic_events_1) + 1)
    x_2 = range(1, len(stats_postsynaptic_events_2) + 1)
    
    if ylabel  is False:
        ylabel = 'Normalized current amplitude'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.hold(True)
    ax.plot(x_1, stats_postsynaptic_events_1[:, ind], color = 'g', linewidth = 2)
    plt.plot(x_2, stats_postsynaptic_events_2[:, ind],color = 'r', linewidth = 2)
    
    ax.fill_between(x_1, stats_postsynaptic_events_1[:, ind] - stats_postsynaptic_events_1[:, err_ind], 
           stats_postsynaptic_events_1[:,ind] + stats_postsynaptic_events_1[:,err_ind], alpha=0.2, facecolor='g', linewidth = 0)
    ax.fill_between(x_2, stats_postsynaptic_events_2[:,ind] - stats_postsynaptic_events_2[:, err_ind], 
           stats_postsynaptic_events_2[:,ind] + stats_postsynaptic_events_2[:, err_ind],alpha=0.2,facecolor='r', linewidth = 0)

    if xlabel is False:    
        ax.set_xlabel('Stimulation number (' + hz + ')')
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend([name1, name2],frameon = False , loc = 1)

    ylim_curr = ax.get_ylim()                
    if ylimmin is True:
        ax.set_ylim([0, ylim_curr[1]])
    
    if save is True:    
        name = 'stimtrain_' + name1 + '_' + name2 + '_' + hz + '_' + ylabel + '.jpg'
        plt.savefig(name, dpi = 800)
    plt.hold(False)



