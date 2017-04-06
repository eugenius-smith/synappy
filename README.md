# synappy

SynapPy is a rapid data visualization and quantification tool for single-cell electrophysiologists.

Its main purpose is to read raw data from .abf files, resolve postsynaptic events from noise, and perform quantification on these events.

Postsynaptic events can either be evoked or spontaneous - the module has ways of detecting both kinds. For evoked events (either optically or electrically triggered), the module uses the TTL pulse of a stim channel. For spontaneous or miniature postsynaptic events, the module uses an algorithm based on 1) first-derivative thresholding of rising slope; 2) filtering false positives based on amplitude and decay fits; and 3) tuning of false positive/false negative balance through an intuitive spontaneous event visualization tool based on bokeh plotting, and adjustable detection parameters.

Once Synappy finds the postsynaptic event times, it also conveniently computes variety of statistics on these events:
    .height             (amplitude from baseline)
    .height_norm        (normalized amplitude from baseline)
    .latency             (A variety of methods: max_height, max_slope, 80_20_line (calcuate epsp foot))
    .decay              (monoexponential or biexponential fits)
    .baseline
    .failure rate   

        
A key convenience factor here is the definition of a custom python object class (synwrapper, for synaptic wrapper) which stores the complete set of data associated with a set of files: analog signals, event times, and all the statistical quantification described above. The synwrapper class also provides a number of built-in methods to both visualize and extend the data.

Retrieving data for processing:
event1.height_norm[neuron] gives arrays of normalized heights for each [neuron]).
event.height_norm[neuron][trial, stim] is the total data structure.

    
It also intelligently filters out bad data if events are not above 4*SD(baseline),
or if their decay constant (tau) is nonsensical. These events are masked but kept 
in the underlying data structure.


The main dependencies are: numpy, scipy, matplotlib, bokeh and neo (ver 0.4+ recommended)





---Typical usage---:

    import synappy as syn

    #----Load data----#
    event1 = syn.load(['15d20004.abf', '15d20007.abf', '15d20020.abf'], trials = [[0, 1], [1, 2], [2, 3]], input_channel = [0, 
    0, 0], stim_channel = [2, 2, 2])
     
            #Here, we define a list of files to load. We can also define a particular number of trials for each neuron      
            #(optional), and input channels and stim_channels (also optional, defaulting to first and last respectively).
     
    #----Detect postsynaptic event times----#
    
    #### 1. Add spontaneous events #####
    
    add_events(event1, event_type = 'spontaneous', spont_filtsize = 25, spont_threshampli = 3, spont_threshderiv = -1.2, 
    savgol_polynomial = 3)
    
            #Add spontaneous events to event1, as event1.stim_on.
            #Event detection based on first-derivative thresholding of analog channel, with optional parameters of 
            #spont_filtsize (a savitsky-golay filter applied to analog_signals before derivative is taken); spont_threshampli 
            #(amplitude threshold for events), and spont_threshderiv (derivative threshold for events)
            
            
    #### 2. Add evoked events #####
    
    add_events(event1, event_type = 'stim', stim_thresh = 2)
            
            #Add evoked events to event1, as event1.stim_on.
            #Event detection based on simple thresholding of stim channel. Optional parameter is stim_thresh, which is the 
            #thresholding of the TTL channel (which should either be 0, upon no stim, or 5, upon stim).

    #----Add statistical quantification of postsynaptic events----#
    
    event1.add_all(event_direction = 'down', latency_method = 'max_height')
    
            #Add .height, .baseline, .latency, .height_norm, .decay to event1. Options are event_direction ('up' for IPSCs or 
            #EPSPs; 'down' for EPSCs or IPSPs) and latency_method (more description in script, but this corresponds to the 
            #method for detecting postsynaptic event latency, from the stim_on time - eg for discriminating between 
            #monosynaptic and polysynaptic events.
    
    #----Fetch data----#
    event1.height_norm[neuron]
    
            #fetch normalized height stats for that neuron. dim0 = trials, dim1 = stims.
            #The raw data behind each attribute can be fetched this way.



---Plotting tools---:

    syn.plot_finds(event1, neuron, trial, starttime, endtime)
            #Plots the detected spontaneous events using the bokeh plotting tool, which generates an interactive html-based 
            #plot overlaying analog signals with detected event amplitudes (red dots)

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
    
        #Pools this attribute over all neurons and trials and outputs out[stims, :]
    
    
    syn.get_stats(event_1.attribute, byneuron = False)
    
        #Gets statistics for this attribute over stimtrain (out[stim,:]) or neuron if byneuron is True (out[neuron,:])
        #dim2: 1->mean, 2->std, 3->sterr, 4->success_rate_mean, 5->success_rate_stdev
        #eg if byneuron = True, out[neuron, 3] would give sterr for that neuron, calculated by pooling across all trials/stims.
    

---Useful methods on the synwrapper class---:

    synwrapper.propagate_mask(): #propagate synwrapper.mask through to all other attributes.
    synwrapper.add_heights() #adds .height and .latency
    synwrapper.add_sorting() #adds .mask and sorts [.height, .latency]
    synwrapper.add_invertedsort() #adds .height_fails
    synwrapper.add_normalized() #adds .height_norm
    synwrapper.add_decay() #adds .decay
    synwrapper.remove_unclamped_aps() #identifies events higher than 5x (default) amp.
                                    It updates the .mask with this information, and then .propagate_mask()
    
    synwrapper.add_all() is a general tool to load all stats.
   
   
   
   Data is automatically filtered in two ways:
    1) height for each event must be above 4*std(baseline) for that trial.
    2) decays must not be nonsensical (tau > 0, tau not way larger than rest of set)
        - This decay-based mask update can be turned off during synwrapper.add_decay()
    
    The Success/fail filter is stored in a global synwrapper.mask attribute 
    that can be propagated to all other attributes via the synwrapper.propagate_mask() method.

