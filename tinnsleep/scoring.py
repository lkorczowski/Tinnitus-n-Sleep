from tinnsleep.burst import burst
from tinnsleep.episode import episode
from tinnsleep.create_reports import merge_labels_list
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
    current_epi.assess_type()
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
    boolean_output:
        ouputs a list of False and True instead of differentiating episodes
    Returns
    -------
    list of integers,
        labels of the events
    """
    # Deals the case the input is empty
    li_events = []
    if len(li_ep) == 0:
        return []

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

    if boolean_output:
        li_events = [bool(li_events[i]) for i in range(len(li_events))]

    return li_events


def generate_bruxism_report(classif, time_interval, delim):
    """ Generates an automatic clinical bruxism report from a list of events

    Parameters
    ----------
    classif : list of booleans,
        output of a classification algorithm that detect non aggregated bursts from a recording
    time_interval: float,
        time interval in seconds between 2 elementary events
    delim: float,
        maximal time interval considered eligible between two bursts within a episode
    Returns
    -------
    report :  dict
    """
    report = {}
    recording_duration = len(classif) * time_interval
    report["Clean data duration"] = recording_duration
    report["Total burst duration"] = np.sum(classif) * time_interval
    li_burst = classif_to_burst(classif, time_interval)
    nb_burst = len(li_burst)
    report["Total number of burst"] = nb_burst
    report["Number of bursts per hour"] = nb_burst * 3600 / recording_duration
    li_episodes = burst_to_episode(li_burst, delim)
    nb_episodes = len(li_episodes)
    report["Total number of episodes"] = nb_episodes
    if nb_episodes > 0:
        report["Number of bursts per episode"] = nb_burst / nb_episodes
    else:
        report["Number of bursts per episode"] = 0
    report["Number of episodes per hour"] = nb_episodes * 3600 / recording_duration

    # Counting episodes according to types and listing their durations
    counts_type = [0, 0, 0]
    tonic = []
    phasic = []
    mixed = []
    for epi in li_episodes:
        if epi.is_tonic:
            counts_type[0] += 1
            tonic.append(epi.end - epi.beg)
        if epi.is_phasic:
            counts_type[1] += 1
            phasic.append(epi.end - epi.beg)
        if epi.is_mixed:
            counts_type[2] += 1
            mixed.append(epi.end - epi.beg)

    report["Number of tonic episodes per hour"] = counts_type[0] * 3600 / recording_duration
    report["Number of phasic episodes per hour"] = counts_type[1] * time_interval * 3600 / recording_duration
    report["Number of mixed episodes per hour"] = counts_type[2] * time_interval * 3600 / recording_duration

    # Mean durations of episodes per types, "nan" if no episode of a type recorded
    report["Mean duration of tonic episode"] = np.mean(tonic)
    report["Mean duration of phasic episode"] = np.mean(phasic)
    report["Mean duration of mixed episode"] = np.mean(mixed)
    return report


def generate_MEMA_report(classif, time_interval, delim):
    """ Generates an automatic clinical middle ear activition (MEMA) report from a list of events

    Parameters
    ----------
    classif : list of booleans,
        output of a classification algorithm that detect non aggregated bursts from a recording
    interval : float,
        time interval in seconds between 2 elementary events
    delim : float,
        maximal time interval considered eligible between two bursts within a episode
    Returns
    -------
    report : dict
    """
    report = {}
    recording_duration = len(classif) * time_interval
    report["Clean MEMA duration"] = recording_duration
    report["Total MEMA burst duration"] = np.sum(classif) * time_interval
    li_burst = classif_to_burst(classif, time_interval=time_interval)
    nb_burst = len(li_burst)
    report["Total number of MEMA burst"] = nb_burst
    report["Number of MEMA bursts per hour"] = nb_burst * 3600 / recording_duration
    li_episodes = burst_to_episode(li_burst, delim=delim, min_burst_joining=0)
    nb_episodes = len(li_episodes)
    report["Total number of MEMA episodes"] = nb_episodes
    if nb_episodes > 0:
        report["Number of MEMA bursts per episode"] = nb_burst / nb_episodes
    else:
        report["Number of MEMA bursts per episode"] = 0
    report["Number of MEMA episodes per hour"] = nb_episodes * 3600 / recording_duration

    episodes_duration = [(epi.end - epi.beg) for epi in li_episodes]

    report["Mean duration of MEMA episode"] = np.mean(episodes_duration)

    return report


def _cond_subclassif(ep_to_sub, labels_artifacts, labels_condition, time_interval):
    # ---------------Subclassifying MEMA and bruxism events----------------------
    comb_ep = []
    pure_ep = []
    compt_arti = 0
    # merge episodes and compare with condition and artefacts
    for elm in ep_to_sub:
        if np.sum(labels_artifacts[int(elm.beg / time_interval):int(elm.end / time_interval)]) == 0:
            if np.sum(labels_condition[int(elm.beg / time_interval):int(elm.end / time_interval)]) > 0:
                comb_ep.append(elm)
            else:
                pure_ep.append(elm)
        else:
            compt_arti += 1
    # ------------------
    # Pure episodes creation
    li_ep_p = create_list_events(pure_ep, time_interval, time_interval * len(labels_condition), boolean_output=True)
    # Combined episode creation
    li_ep_c = create_list_events(comb_ep, time_interval, time_interval * len(labels_condition), boolean_output=True)

    return li_ep_c, li_ep_p, compt_arti

def _labels_to_ep_and_bursts(labels, time_interval, delim_ep, min_burst_joining=3):
    # ------------grouping episodes and bursts together------------------------------------
    bursts = classif_to_burst(labels, time_interval=time_interval)
    li_ep = burst_to_episode(bursts, delim=delim_ep, min_burst_joining=min_burst_joining)
    events = create_list_events(li_ep, time_interval, len(labels) * time_interval, boolean_output=True)
    ep_and_bursts = np.any(np.c_[labels, events], axis=-1)  # rassembling bruxism bursts and episodes
    return ep_and_bursts, li_ep




def combine_brux_MEMA(labels_brux, labels_artifacts_brux, time_interval_brux, delim_ep_brux, labels_MEMA,
                      labels_artifacts_MEMA, time_interval_MEMA, delim_ep_MEMA,
                      min_burst_joining_brux=3, min_burst_joining_MEMA=0):
    """
    Hypothesis labels_artifacts_brux has the same length as labels_brux idem for MEMA

    """
    # Putting labels inputs on the same sampling and epoching
    if len(labels_brux) != len(labels_MEMA):
        if len(labels_brux) < len(labels_MEMA):
            labels_brux = merge_labels_list ([labels_brux], len(labels_MEMA))
            # adapts and fuses artifacts
            labels_artifacts =  merge_labels_list ([labels_artifacts_brux,labels_artifacts_MEMA], len(labels_MEMA))
            time_interval = time_interval_MEMA
        else:
            labels_MEMA = merge_labels_list([labels_MEMA], len(labels_brux))
            # adapts and fuses artifacts
            labels_artifacts = merge_labels_list([labels_artifacts_brux,labels_artifacts_MEMA], len(labels_brux))
            time_interval = time_interval_brux
    else: #inputs of same length
        # fuses artifacts
        labels_artifacts = merge_labels_list([labels_artifacts_brux, labels_artifacts_MEMA], len(labels_brux))
        time_interval = time_interval_brux

    # Creating lists of episode and bursts for bruxism and MEMA
    brux_burst_ep, li_ep_brux = _labels_to_ep_and_bursts(labels_brux, time_interval, delim_ep_brux,
                                                         min_burst_joining=min_burst_joining_brux)
    MEMA_burst_ep, li_ep_MEMA = _labels_to_ep_and_bursts(labels_MEMA, time_interval, delim_ep_MEMA,
                                                         min_burst_joining=min_burst_joining_MEMA)

    # Conditionnal labelling of events
    MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA= _cond_subclassif(li_ep_MEMA, labels_artifacts,
                                                                 brux_burst_ep, time_interval)
    brux_comb_ep, brux_pure_ep, compt_arti_brux = _cond_subclassif(li_ep_brux, labels_artifacts,
                                                                   MEMA_burst_ep, time_interval)

    return brux_comb_ep, brux_pure_ep, compt_arti_brux, MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA






