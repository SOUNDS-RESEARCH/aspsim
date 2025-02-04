import numpy as np





class PhaseCounter:
    """An index counter to keep track of non-overlapping continous phases. 

    The number is how many samples that each phase should be. The first phase will start at sample 0.

    Example
    -------
    A processor needs the first 2000 samples for an initialization, then must wait 5000 samples before beginning the real processing step.
    The class can be used by defining the phases as
    >>> phase_def = {'init' : 2000, 'wait' : 5000, 'process' : np.inf}

    and then checking if either of the following is true
    >>> if phase_counter.phase == 'init'
    >>> if phase_counter.current_phase_is('init')

    Notes
    -----
    np.inf represents an infinite length
    This should naturally only be used for the last phase
    If all phases has finished, the phase will be None. 

    first_sample will be True on the first sample of each phase,
    allowing running one-time functions in each phase

    Extended implementation to block_size != 1 can be done later

    """
    def __init__(self, phase_lengths, verbose=False):
        assert isinstance(phase_lengths, dict)
        self.phase_lengths = phase_lengths
        self.verbose = verbose
        self.phase = None
        self.first_sample = True
        

        #phase_lengths = {name : length for name, length in self.phase_lengths.items() if length != 0}
        #phase_lengths = {name : length for name, length in self.phase_lengths.items()}
        
        #phase_idxs = [i for i in self.phase_lengths.values() if i != 0]
        self.phase_lengths = {name : i if i >= 0 else np.inf for name, i in self.phase_lengths.items()}
        #assert all([i != 0 for i in p_len])
        self.start_idxs = np.cumsum(list(self.phase_lengths.values())).tolist()
        self.start_idxs = [i if np.isinf(i) else int(i) for i in self.start_idxs]
        self.start_idxs.insert(0,0)

        self.phase_names = list(self.phase_lengths.keys())
        if self.start_idxs[-1] < np.inf:
            self.phase_names.append(None)
        else:
            self.start_idxs.pop()

        self.start_idxs = {phase_name:start_idx for phase_name, start_idx in zip(self.phase_names, self.start_idxs)}

        self._phase_names = [phase_name for phase_name, phase_len in self.phase_lengths.items() if phase_len > 0]
        self._start_idxs = [start_idx for start_idx, phase_len in zip(self.start_idxs.values(), self.phase_lengths.values()) if phase_len > 0]

        self.idx = 0
        self.next_phase()

    def next_phase(self):
        if self.verbose:
            print(f"Changed phase from {self.phase}")
            
        self.phase = self._phase_names.pop(0)
        self._start_idxs.pop(0)
        if len(self._start_idxs) == 0:
            self._start_idxs.append(np.inf)
        self.first_sample = True
        
        if self.verbose:
            print(f"to {self.phase}")

    def progress(self):
        self.idx += 1
        if self.idx >= self._start_idxs[0]:
            self.next_phase()
        else:
            self.first_sample = False

    def current_phase_is(self, phase_name):
        return self.phase == phase_name



class EventCounter:
    """
    An index counter to keep track of events that should 
    only happen every x samples

    event_def is a dictionary with all event
    each entry is event_name : (frequency, offset)

    Example:
    event_counter = EventCounter({'event_1' : (256,0), 'event_2' : (1,0), 'event_3' : (1024,256)})
    event_2 will happen every sample, event_1 every 256 samples
    First at sample 256 all three events will happen simultaneouly. 

    To be used as:
    if event_name in event_counter.event:
        do_thing()

    """
    def __init__(self, event_def):
        self.event_def = event_def
        self.event = []

        self.freq = {name : freq for name, (freq, offset) in event_def.items()}
        self.offset = {name : offset for name, (freq, offset) in event_def.items()}

        self.idx = 0

    def add_event(self, name, freq, offset):
        self.event_def[name] = (freq, offset)

    def _check_events(self):
        self.event = []
        for name, (freq, offset) in self.event_def.items():
            if (self.idx - offset) % freq == 0:
                self.event.append(name)

    def progress(self):
        self.idx += 1 
        self._check_events()