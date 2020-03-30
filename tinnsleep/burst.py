class burst():
    """Define a bruxism burst object and caracterize it

    Parameters
    ----------
    beg : time of beggining of the burst
    end : time of end of the burst
    

    Attributes
    ----------
    is_tonic: boolean, caracterizes if a bruxism burst is tonic i.e superior in
    duration to 2 seconds.


    See Also
    --------

    """

    def __init__(self, beg, end):
        self.beg = beg
        self.end = end
        if (self.end - self.beg >= 2):
            self.is_tonic=True
        else:
            self.is_tonic=False
        
    def is_equal(self, burst):
        """ identify if two burst are equal
        
        Parameters
        ----------
        self : burst instance
        burst : burst instance
    
        Returns
        -------
        boolean, True if beg and end are the same
        """
        return self.beg==burst.beg and self.end==burst.end
    
    def is_before(self, burst):
        """ test if the current burst is temporally before another given burst
        Parameters
        ----------
        self : burst instance
        burst : burst instance
    
        Returns
        -------
        boolean, True if the current burst beg is inferior to the other burst 
        beg
        """
        return self.beg<burst.beg
    
    
    def is_overlapping(self, burst):
        """ test if the current burst is overlapping with another given burst
        Parameters
        ----------
        self : burst instance
        burst : burst instance
    
        Returns
        -------
        boolean, True if the bursts are overlapping
        """
        
        is_contained= self.beg>=burst.beg and self.end<=burst.end
        contains= self.beg<=burst.beg and self.end>=burst.end
        ov_left = (self.beg>=burst.beg and self.end>=burst.end) and (burst.end-self.beg>0)
        ov_right=(self.beg<=burst.beg and self.end<=burst.end) and (self.end-burst.beg>0)
        return is_contained or contains or ov_right or ov_left
            
    def merge_if_overlap(self, burst):
        """ test if the current burst is overlapping with another burst and if
        it is merges the two burst together
        Parameters
        ----------
        self : burst instance
        burst : burst instance
    
        Returns
        -------
        boolean, True if the bursts are overlapping
        """
        if burst.is_overlapping(self):
            if self.beg>burst.beg:
                self.beg = burst.beg
            if self.end<burst.end:
                self.end = burst.end
            
            if (self.end - self.beg > 2):
                self.is_tonic=True
            else:
                self.is_tonic=False
            
            #print(self.beg)
            #print(self.end)
            return  True
        else:
            return False
        

           
           