import numpy as np
import matplotlib.pyplot as plt


def karplus_strong_note(sr=44100, note=14, duration=.01, decay=.98):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    decay: float 
        Decay amount (between 0 and 1)

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    N = int(duration*sr)
    y = np.zeros(N)
    ## TODO: Fill this in
    frequency = (sr/100)*2**(note/12)
    T = int(sr/frequency)
    y[0:T] = np.random.rand(T)
    for i in range(T,N):
        y[i] = decay * ((y[i-T]+y[i-T+1])/2)
    return y

def fm_synth_note(sr=44100, note=14, duration=.01, ratio = 2, I = 2, 
                  envelope = lambda N, sr: np.ones(N),
                  amplitude = lambda N, sr: np.ones(N)):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    ratio: float
        Ratio of modulation frequency to carrier frequency
    I: float
        Modulation index (ratio of peak frequency deviation to
        modulation frequency)
    envelope: function (N, sr) -> ndarray(N)
        A function for generating an ADSR profile
    amplitude: function (N, sr) -> ndarray(N)
        A function for generating a time-varying amplitude

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    #y(t) = A(T)cos(2pi(fc)t+I(t)sin(2pi(fm)t))
    N = int(duration*sr)
    y = np.zeros(N)
    
    fc = 440*2**(note/12)
    fm = ratio*fc
    
    t = np.arange(N)/sr
    
    I = I*envelope(N,sr)
    
    y = amplitude(N,sr)*np.cos(2*np.pi*fc*t+I*np.sin(2*np.pi*fm*t))
    
    return y

def exp_env(N, sr,lam=3):
    """
    Make an exponential envelope
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    lam: float
        Exponential decay rate: e^{-lam*t}

    Returns
    -------
    ndarray(N): Envelope samples
    """
    return np.exp(-lam*np.arange(N)/sr)

def drum_like_env(N, sr):
    """
    Make a drum-like envelope, according to Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    y = np.zeros(N)
    t = np.arange(N)/sr
    y = ((t**2)*exp_env(N,sr,45)*(t/250))+.125
    return y

def wood_drum_env(N, sr):
    """
    Make the wood-drum envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    y = np.zeros(N)
    t = np.arange(N)/sr
    for i in range(int(N/8)):
        y[i] = y[i] - y[i+1]
    return y
    
    return np.zeros(N)

def brass_env(N, sr):
    """
    Make the brass ADSR envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
 
    env = np.zeros(N)
    time = N/sr
    
    if time <= 0.3:
        release = np.linspace(.75,0,(int(sr*0.05)))
        decay = np.linspace(1,0.75,(int(sr*0.2)-int(sr*.05)))
        attack = np.linspace(0,1,int(sr*0.05))
        env = np.concatenate((attack,decay,release))
   
    else:
        attack = np.linspace(0, 1, int(sr*0.1))
        decay = np.linspace(1,0.75,int(sr*0.1))
        sustain = np.linspace(0.75,0.7,(int(sr*0.9)-int(sr*0.2)))
        release = np.linspace(0.7,0,int(sr*0.1))
        env = np.concatenate((attack,decay,sustain,release))
        
    return env 


def dirty_bass_env(N, sr):
    """
    Make the "dirty bass" envelope from Attack Magazine
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    envelope = exp_env(N, sr)
    for i in range(0, int(N/2)):
        time = i/N
        envelope[i] = i**-time
        envelope[int(N/2)+i] = 1 - (i**-time)      
    return envelope

def fm_plucked_string_note(sr=44100, note=14, duration=.01, lam = 3):
    """
    Make a plucked string of a particular length
    using FM synthesis
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    lam: float
        The decay rate of the note
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, lam)
    return fm_synth_note(sr, note, duration, \
                ratio = 1, I = 8, envelope = envelope,
                amplitude = envelope)


def fm_brass_note(sr=44100, note=14, duration=.01):
    """
    Make a brass note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr)
    return fm_synth_note(sr, note, duration, \
                ratio = 1, I = 10, envelope = envelope,
                amplitude = envelope)


def fm_bell_note(sr=44100, note=14, duration = .01,lam=.8):
    """
    Make a bell note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    ## TODO: Fill this in
    envelope = lambda N, sr: exp_env(N, sr, lam)
    return fm_synth_note(sr, note, duration, \
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope) 


def fm_drum_sound(sr=4410, note=14, duration=.01, fixed_note = -14):
    """
    Make what Chowning calls a "drum-like sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    ------
    ndarray(N): Audio samples for this drum hit
    """
    ## TODO: Fill this in
    envelope = lambda N, sr: exp_env(N, sr)
    return fm_synth_note(sr, fixed_note, duration, \
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope)

def fm_wood_drum_sound(sr=44100, note=14, duration=.01, fixed_note = -14):
    """
    Make what Chowning calls a "wood drum sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    ## TODO: Fill this in
    envelope = lambda N, sr: exp_env(N, sr)
    return fm_synth_note(sr, fixed_note, duration, \
                ratio = 1.4, I = 10, envelope = envelope,
                amplitude = envelope) 

def fm_dirty_bass_note(sr=44100, note=14, duration=.01):
    """
    Make a "dirty bass" note, based on 
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    ## TODO: Fill this in
    envelope = lambda N, sr: exp_env(N, sr,lam=3)
    return fm_synth_note(sr, note, duration, \
                ratio = 1, I = 18, envelope = envelope,
                amplitude = envelope)  # This is a dummy value

def make_tune(filename, sixteenth_len, sr, note_fn):
    """
    Parameters
    ----------
    filename: string
        Path to file containing the tune.  Consists of
        rows of <note number> <note duration>, where
        the note number 0 is a 440hz concert A, and the
        note duration is in factors of 16th notes
    sixteenth_len: float
        Length of a sixteenth note, in seconds
    sr: int
        Sample rate
    note_fn: function (sr, note, duration) -> ndarray(M)
        A function that generates audio samples for a particular
        note at a given sample rate and duration
    
    Returns
    -------
    ndarray(N): Audio containing the tune
    """
    tune = np.loadtxt(filename)
    notes = tune[:, 0]
    durations = sixteenth_len*tune[:, 1]
    lil_tune = []
    num_notes = len(notes)
    for i in range(num_notes):
        indv_note = note_fn(sr, notes[i], durations[i])
        if np.isnan(notes[i]):
            indv_note = np.zeros(durations[i]*sr)
            lil_tune = np.concatenate((lil_tune,indv_note))
        else:
            lil_tune = np.concatenate((lil_tune,indv_note))
    ## TODO: Fill this in
    return lil_tune # This is a dummy value