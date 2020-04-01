from tinnsleep.burst import burst
from tinnsleep.episode import episode


def classif_to_burst(classif, time_interval=0.25):
    """ Transforms a classification boolean list of results into a 
    chronological list of bursts
    
    Parameters
    ----------
    classif : list of booleans, output of a classification algorithm that 
    detect non aggregated bursts from a recording
    interval: float, time interval between 2 elements of classif

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
        burst_list.append(burst(beg * time_interval, (leny) * time_interval))
    return burst_list



def rearrange_chronological(burst_list):
    """Rearrange a given burst_list in the chronological order according to the beg attribute"""
    burst_list.sort(key=lambda x: x.beg)
    return burst_list



def burst_to_episode(burst_list, delim=3):
    """ Transforms a chronological list of bursts into a 
    chronological list of episodes
    IMPORTANT: the list of bursts MUST be chronological
    
    Parameters
    ----------
    burst_list : list of burst instances
    delim: float, maximal time interval considered eligible between two bursts 
    within a episode
    Returns
    -------
    ep_list : list of episodes instances 
    """
    leny = len(burst_list)
    # test if input is empty
    if leny == 0:
        print("burst list empty")
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
            current_epi.assess_type()
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


def get_event_label(episode):
    """ return the label of the episode"""
    if episode.is_tonic:
        return 1
    if episode.is_phasic:
        return 2
    if episode.is_mixed:
        return 3


def create_list_events(li_ep, interval):
    """ Creates the list of events, 0 = no event, 1 = tonic episode, 2 = phasic 
    episode, 3 = mixed episode
    
    Parameters
    ----------
    li_ep : list of episode instances
    interval: float, interval between 2 elementary events
    Returns
    -------
    list of integers, labels of the events 
    """
    #Deals the case the input is empty
    li_events = []
    if len(li_ep) == 0:
        return []

    # Initialization
    # Case where the first burst does not begin at time=0
    if li_ep[0].beg / interval > 1:
        li_0 = [0 for i in range(int(li_ep[0].beg / interval))]
        li_events.extend(li_0)

    # Tagging the first event
    label = get_event_label(li_ep[0])
    li_events.extend([label for i in range(int((li_ep[0].end - li_ep[0].beg) / interval))])

    # Loop of tagging
    if len(li_ep) > 1:
        for i in range(len(li_ep) - 1):
            # Putting zeros between two episodes
            li_events.extend([0 for i in range(int((li_ep[i + 1].beg - li_ep[i].end) / interval))])
            # Tagging the next episode
            label = get_event_label(li_ep[i + 1])
            li_events.extend([label for i in range(int((li_ep[i + 1].end - li_ep[i + 1].beg) / interval))])

    return li_events
