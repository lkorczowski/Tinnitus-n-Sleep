import matplotlib.pyplot as plt




def create_visual(RAW, detect, duration, interval):
    """ Create two visuals of a burst detection : a simple plot enabling 
    visualization of time repartition and a display over the real signal
    
    Parameters
    ----------
    Raw: mne raw instance of the signal 
    detect: ndarray, shape (n_trials)
    duration : float, duration in seconds of an epoch
    interval : float, interval between 2 epochs (enabling overlap)
    
    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(detect)
    if len(RAW.annotations)>0:
        for i in range(len(RAW.annotations)):
            RAW.annotations.delete(0)
    for i in range(len(detect)):
        if detect[i]:
            RAW.annotations.append([interval * i], [duration], str("brux_burst"))
    RAW.plot(scalings="auto")
    
            
    return
    
    
    
    
    
    
        
    