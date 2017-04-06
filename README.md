# synappy

SynapPy is a rapid data visualization and quantification tool for single-cell electrophysiologists.


It takes .abf files as inputs and detects poststim synaptic events (based on either a stim-channel for optically or electrically evoked events; or based on spontaneous event detection for mEPSC or mIPSCs).

It also computes variety of statistics on the postsynaptic events:
    .height
    .height_norm
    .latency             [by max_height, max_slope, 80_20_line (calcuate epsp foot)]
    .decay
    .baseline
    .failure rate

        
The raw values are stored as attributes in the synwrapper object
(eg even1.height_norm[neuron] gives arrays of normalized heights for each [neuron])

    
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
   
   
   
   
~~~~~~~Data filtering and manipulation~~~~~~
    
Data is automatically filtered in two ways:
    1) height for each event must be above 4*std(baseline) for that trial.
    2) decays must not be nonsensical (tau > 0, tau not way larger than rest of set)
        - This decay-based mask update can be turned off during synwrapper.add_decay()
    
    The Success/fail filter is stored in a global synwrapper.mask attribute 
    that can be propagated to all other attributes via the synwrapper.propagate_mask() method.

