from tinnsleep.events.burst import burst
from tinnsleep.events.episode import episode
import numpy as np


def classif_to_burst(classif, time_interval=0.25):
    """ Transforms a classification boolean list of results into a
    chronological list of bursts
    
    Parameters
    ----------
    classif : list of booleans,
        output of a classification algorithm that detect non aggregated bursts from a recording
    interval: float,
        time in seconds interval between 2 elements of classif
    Returns
    -------
    burst_list : list of bursts instances
    """

    burst_list = []
    leny = len(classif)
    # test if input is empty
    if leny == 0:
        print("classif empty")
        return []
    # initialization
    flag = classif[0]
    # set a beginning if classif[0]==True
    if flag:
        beg = 0
    i = 1
    # Loop that merges contiguous bursts together
    while i < leny:

        if flag:
            # We continue on the same burst
            if not classif[i]:
                # We close the burst
                flag = False
                burst_list.append(burst(beg * time_interval, (i) * time_interval))

        else:
            if classif[i]:
                # We begin a new burst
                flag = True
                beg = i

        i += 1
    # Deals the case if the last input is True
    if flag:
        burst_list.append(burst(beg * time_interval, leny * time_interval))
    return burst_list


def rearrange_chronological(brux_list):
    """Rearrange a given burst list or episode list in the chronological order according to the beg attribute

    Parameters
    ----------
    burst_list : list of bursts instances
    potentially non-chronologically ordered
    Returns
    -------
    burst_list : list of bursts instances
        same list, chronologically ordered
        """

    brux_list.sort(key=lambda x: x.beg)
    return brux_list


def burst_to_episode(burst_list, delim=3, min_burst_joining=3):
    """ Transforms a chronological list of bursts into a 
    chronological list of episodes
    
    Parameters
    ----------
    burst_list : list of burst instances
    delim: float, (default 3)
        maximal time interval considered eligible between two bursts within a episode
    min_burst_joining: Optional, int, default 3
        Minimum of bursts to join to form an episode.
    Returns
    -------
    ep_list : list of episodes instances 
    """
    leny = len(burst_list)
    # test if input is empty
    if leny == 0:
        #print("burst list empty")
        return []

    # Rearranges the list of burst in chronological order
    burst_list = rearrange_chronological(burst_list)

    # Initialization
    current_epi = episode(burst_list[0])
    ep_list = []
    i = 1
    # Loop that reunites bursts within the same episode
    while (i < leny):
        if burst_list[i].beg - current_epi.end < delim:
            # The burst can be added to the current episode
            current_epi.add_a_burst(burst_list[i])

        else:
            # conclusion of the current_episode
            current_epi.assess_type(min_burst_joining=min_burst_joining)
            # Adding the episode only if valid
            if current_epi.is_valid():
                ep_list.append(current_epi)
            # a new current_episode is initiated
            current_epi = episode(burst_list[i])

        i += 1

    # Deals with the last burst
    current_epi.assess_type(min_burst_joining=min_burst_joining)
    if current_epi.is_valid():
        ep_list.append(current_epi)

    return ep_list

def generate_annotations(li_ep):
    """ Transforms a chronological list of episodes into a
        chronological list of annotations

        Parameters
        ----------
        li_ep : list of episode instances

        Returns
        -------
        Annotations : list of dictionaries
            Chronological list of annotations
        """

    annotations = []
    for ep in li_ep:
        annotations.append(ep.generate_annotation())
    return annotations


def get_event_label(episode):
    """ return the label of the episode

    Parameters
    ----------
    episode : episode instance

    Returns
    -------
    int
        int label of the type of episode
    """

    if episode.is_tonic:
        return 1
    if episode.is_phasic:
        return 2
    if episode.is_mixed:
        return 3


def create_list_events(li_ep, time_interval, time_recording, boolean_output=False):
    """ Creates the list of events, 0 = no event, 1 = tonic episode, 2 = phasic 
    episode, 3 = mixed episode
    
    Parameters
    ----------
    li_ep : list of episode instances
    time_interval: float,
        time interval in seconds between 2 elementary events
    time_recording : float,
        duration of the recording in seconds
    boolean_output: boolean, (default : False)
        ouputs a list of False and True instead of differentiating episodes
    Returns
    -------
    list of integers,
        labels of the events
    """
    # Deals the case the input is empty
    li_events = []
    if len(li_ep) == 0:
        if boolean_output:
            return [False for i in range(int(time_recording / time_interval))]
        else:
            return [0 for i in range(int(time_recording / time_interval))]

    # Initialization
    # Case where the first burst does not begin at time=0
    if li_ep[0].beg / time_interval > 1:
        li_0 = [0 for i in range(int(li_ep[0].beg / time_interval))]
        li_events.extend(li_0)

    # Tagging the first event
    if boolean_output:
        label = 1
    else:
        label = get_event_label(li_ep[0])
    li_events.extend([label for i in range(int((li_ep[0].end - li_ep[0].beg) / time_interval))])

    # Loop of tagging
    if len(li_ep) > 1:
        for i in range(len(li_ep) - 1):
            # Putting zeros between two episodes
            li_events.extend([0 for i in range(int((li_ep[i + 1].beg - li_ep[i].end) / time_interval))])
            # Tagging the next episode
            if boolean_output:
                label = 1
            else:
                label = get_event_label(li_ep[i + 1])
            li_events.extend([label for i in range(int((li_ep[i + 1].end - li_ep[i + 1].beg) / time_interval))])

    # If necessary, adds 0s at the end of li_events until the end of the recording
    if (time_recording - li_ep[-1].end) / time_interval > 1.0:
        li_0 = [0 for i in range(int((time_recording - li_ep[-1].end) / time_interval))]
        li_events.extend(li_0)
    # Conversion to "True" and "False" values
    if boolean_output:
        li_events = [bool(li_events[i]) for i in range(len(li_events))]
    return li_events


def episodes_to_list(list_episodes, time_interval, n_labels):
    """ Creates the list of events based on episode code

    Parameters
    ----------
    li_ep : list of episode instances
    time_interval: float,
        time interval in seconds between 2 elementary events
    n_labels : int
        length of the list dof labels associated with the recording

    Returns
    -------
    labels : ndarray, shape (n_labels, )
        labels of the events
    """
    labels = np.zeros((n_labels,))      # initialized label list
    time_stamps = np.arange(0, n_labels + 1) * time_interval
    for episode_ in list_episodes:
        # condition: complete overlap
        is_episode = np.all(np.c_[episode_.beg <= time_stamps[:-1], time_stamps[1:] <= episode_.end], axis=-1)
        labels[is_episode] = episode_.code  # should be an int code 1 : phasic, 2, tonic, etc.

    return labels
