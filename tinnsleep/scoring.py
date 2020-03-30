from tinnsleep.burst import burst
from tinnsleep.episode import episode

def classif_to_burst(classif, interval=0.25):
    """ Transforms a classification boolean list of results into a 
    chronological list of bursts
    
    Parameters
    ----------
    classif : list of booleans, output of a classification algorithm that 
    detect non agregated bursts from a recording
    interval: float, time interval between 2 elements of classif

    Returns
    -------
    list of bursts instances 
    """

    burst_list=[]
    leny=len(classif)
    #test if input is empty
    if leny==0:
        print("entree classif vide")
        return []
    #initialization
    flag= classif[0]
    #set a beggining if classif[0]==True
    if flag:
        beg=0
    i=1
    #Loop that merges contiguous bursts together
    while(i<leny):
        
        if flag:
            #We continue on the same burst
            if not classif[i]:
                #We close the burst
                flag=False
                burst_list.append(burst(beg*interval,(i)*interval))
                
        else:
            if classif[i]:
                #We begin a new burst
                flag=True
                beg=i
                
        i+=1
    #Deals the case if the last input is True
    if flag:
         burst_list.append(burst(beg*interval,(leny)*interval))
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
    list of episodes instances 
    """
    leny=len(burst_list)
    #test if input is empty
    if leny==0:
        print("entree classif vide")
        return []
    
    #Initialization
    current_epi= episode(burst_list[0])
    ep_list=[]
    i=1
    #Loop that reunites bursts within the same episode
    while(i<leny):
        if burst_list[i].beg-current_epi.end<delim:
            #The burst can be added to the current episode
            current_epi.add_a_burst(burst_list[i])
            
        else:
             #conclusion of the current_episode
             current_epi.assess_type()
             #Adding the episode only if valid
             if current_epi.is_valid():
                 ep_list.append(current_epi)
             #a new current_episode is initiated
             current_epi= episode(burst_list[i])
             
        i+=1
        
    #Deals with the last burst
    current_epi.assess_type()
    if current_epi.is_valid():
        ep_list.append(current_epi)
    
    
    return ep_list
            