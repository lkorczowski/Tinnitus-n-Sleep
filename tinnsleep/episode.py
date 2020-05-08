class episode():
    """Define a bruxism episode object and characterize it

    Parameters
    ----------
    beg : time of beginning of the episode
    end : time of end of the episode
    

    Attributes
    ----------
    burst_list: list, list of burst(as simple episode objects) contained within
    an episode
    is_tonic: boolean, characterizes if a bruxism episode is tonic i.e superior
    in duration to 2 seconds.
    is_phasic: boolean, characterizes if a bruxism episode is phasic
    is_mixed: boolean, characteerises if a bruxism episode is mixed


    See Also
    --------

    """

    def __init__(self, burst):
        self.beg = burst.beg
        self.end = burst.end
        self.burst_list = [burst]

    def add_a_burst(self, burst):
        """ adds a burst to the current instance of episode
        Parameters
        ----------
        self : episode instance
        burst : burst instance
    
        Returns
        -------
        None
        """
        # Deal if the burst begins before the current episode
        # Case if it is overlapping
        if self.burst_list[0].merge_if_overlap(burst):
            self.beg = min(self.beg, burst.beg)
            # Check if it is now not overlapping with the one after
            if len(self.burst_list) > 1:
                if self.burst_list[0].merge_if_overlap(self.burst_list[1]):
                    self.burst_list.pop(1)
            # checks the other end if the burst goes until there
            self.end = max(self.end, burst.end)

        # Case if it is not overlapping
        elif self.beg >= burst.end:
            self.beg = burst.beg
            self.burst_list.insert(0, burst)

    # Deal if the burst ends after the current episode
        # Case if it is overlapping
        elif self.burst_list[len(self.burst_list) - 1].merge_if_overlap(burst):
            self.end = max(self.end, burst.end)
            # Check if it is now not overlapping with the one before
            if len(self.burst_list) > 1:
                if self.burst_list[-2].merge_if_overlap(self.burst_list[-1]):
                    self.burst_list.pop(-1)
        # Case if it is not overlapping
        elif self.end <= burst.beg:
            self.end = burst.end
            self.burst_list.append(burst)

        # Deal if the burst doesn't affect the episode borders
        else:
            # Check if the episode doesn't contain the burst already
            already_there = False
            for elm in self.burst_list:
                if burst.is_equal(elm):
                    print("the burst is already there")
                    already_there = True
            # If it is a new burst
            if not already_there:
                for i in range(len(self.burst_list)):
                    # Test if the new burst overlaps with one existing
                    flag = self.burst_list[i].merge_if_overlap(burst)
                    if flag:
                        if not i == len(self.burst_list) - 1:  # Si le burst introduit recoupe sur 2 bursts
                            if self.burst_list[i].merge_if_overlap(self.burst_list[i + 1]):
                                self.burst_list.pop(i + 1)
                        break
                    # Deal if it is not
                    else:
                        if burst.is_before(self.burst_list[i]):
                            self.burst_list.insert(i, burst)
                            break

    def set_tonic(self):
        """ set the is_tonic episode attribute"""
        self.is_tonic = False
        for bu in self.burst_list:
            if bu.is_tonic:
                self.is_tonic = True
                self.code += 10

    def set_phasic(self, min_burst_joining=3):
        """ set the is_phasic episode attribute"""
        if len(self.burst_list) >= min_burst_joining:
            self.is_phasic = True
            self.code += 100
        else:
            self.is_phasic = False

    def set_mixed(self):
        """ set the is_mixed episode attribute"""
        if self.is_tonic and self.is_phasic:
            self.is_mixed = True
            self.is_tonic = False
            self.is_phasic = False
        else:
            self.is_mixed = False

    def assess_type(self, min_burst_joining=3):
        """ characterizes fully the episode"""
        self.code = 1
        self.set_tonic()
        self.set_phasic(min_burst_joining=min_burst_joining)
        self.set_mixed()

    def is_valid(self):
        """ test if the episode is tonic, phasic or mixed and returns True 
        if it is"""
        return self.is_tonic or self.is_phasic or self.is_mixed

    def generate_annotation(self, orig_time = 0.0):
        """ Generate an annotation on the format of a dictionary like:
        annotation = {'onset': 0.5, 'duration': 1.0, 'description': "blink", 'orig_time': 0.0}
        from a given episode
        Parameters
        ----------
        orig_time: float | default 0.0
            sets the time of offset for the annotation

        Returns
        -------
        annotation: dict
            the mne compatible dictionary of annotation
        """
        # First test if it is a valid episode
        if self.is_valid():
        # Chooses the description suited for the episode
            if self.is_tonic:
                descrip="Tonic"
            if self.is_phasic:
                descrip="Phasic"
            if self.is_mixed:
                descrip="Mixed"

            return {'onset': self.beg, 'duration': self.end-self.beg, 'description': descrip, 'orig_time': orig_time}
        # Case it is not a valid episode
        else:
            return {}