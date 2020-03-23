import mne

#Create Raw file
def CreateRaw(data, ch_names):
    """Generate a mne raw structure based on hardcoded info"""
    ch_types = ['eeg'] * len(ch_names)
    sfreq = 200
    montage = 'standard_1020'
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=montage, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw

def RawToEpochs_sliding(raw, duration, interval, picks=None):
    """Generate an Epoch structure from mne raw given the duration and interval by sliding window"""
    events = mne.make_fixed_length_events(raw, id=1, duration=interval)
    epochs = mne.Epochs(raw, events,
                        tmin=0, tmax=duration, picks=picks,
                        reject_by_annotation=True, detrend=0, preload=False,
                        baseline=None,
                        verbose=False)
    return epochs