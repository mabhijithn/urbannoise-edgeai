#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:10:30 2021

@author: abhijith
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:53:20 2021

@author: abhijith
"""
import pyaudio
import wave
from os import path
from collections import deque
import os
import audioop
import math
import time
from datetime import datetime
import glob
import warnings
from pathlib import Path

#WAV_FILE_PATH = path.join(os.getenv('HOME'),'edgeai-wavs')
WAV_FILE_PATH = 'wavfiles'

if path.isdir(WAV_FILE_PATH) is False:
    os.mkdir(WAV_FILE_PATH)
    
# Microphone stream config.
CHUNK = 1024  # CHUNKS of bytes to read each time from mic
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
#th = os.getenv("WAV_REC_THRESHOLD", "2000")  # The threshold intensity that defines silence
                  # and noise signal (an int. lower than THRESHOLD is silence).
                  # based on peak ampltude in each chunk of audio data
SILENCE_LIMIT = 2  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording is saved and evaluated.

PREV_AUDIO = 1  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the sound.
                  
NOISE_MONITOR_DUR = 60*10 # Duration for which to record 'silent' period at night
                    # and calculate noise during day time.

dev_index = 0 # device index found by p.get_device_info_by_index(ii)
MAX_FILE_LENGTH = 4 # Number of seconds until a new file is started while recording
MIN_SAVE_LENGTH = 1
WAV_FILE_LIMIT = 6000000000
    
file_count = 0 # counter for how many files created in this session.


def save_speech(data, p):
    '''
    

    Parameters
    ----------
    data : bytearray
        Audio data recorded using pyaudio stream.
    p : PyAudio instance
        PyAudio instance.

    Returns
    -------
    str
        Name of the saved WAV file.
    timestr : str
        datetime object converted to H:M:S format using strftime.

    '''

    filename = str(int(time.time()))
    now = datetime.now()
    # writes data to WAV file
    data = b''.join(data)
    wf = wave.open(path.join(WAV_FILE_PATH,filename+'.wav'), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    timestr = now.strftime("%H:%M:%S")
    return filename + '.wav', timestr

def find_baseline(duration=60):
    '''
    

    Parameters
    ----------
    duration : float, optional
        The duration (in seconds) for which sound is to be recorded to 
        calculate baseline average RMS value. The default is 60 seconds. This 
        baseline is used against which recorded audio is compared to check for
        noise.

    Returns
    -------
    avg_power : float
        Average RMS value of audio for duration. A measure of power in the
        audio signal.

    '''
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                   input_device_index=dev_index,
                   input=True,
                   frames_per_buffer=CHUNK)
    print('Collecting ambient background...')
    frames = []
    for i in range(0, int(RATE/CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(audioop.rms(data,4))
    avg_power = sum(frames)/len(frames)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print('Average RMS value of ambient backround from ', duration, 's recorded: ',avg_power)
    return avg_power

def listen_for_audio(night_noise,bg_power,threshold):
    '''
    

    Parameters
    ----------
    night_noise : float
        Average RMS value of a 'silent' urban noise period. For example, you 
        can run find_baseline for a duration of 10-15 mins at night and save
        the calculated average RMS value as 'silent' background noise.
        This is used to calculate noise during daytime for every fixed duration 
        (10 or 15 minutes).
    bg_power : float
        Average ambient noise power from audio recorded at the beginning of 
        urban noise monitoring. The urban noise is compared against this ambient
        background noise before saving as WAV file. 
    threshold : float
        Noise in dB. If the recorded audio has a noise level above this threshold 
        (in dB), then the recorded audio is saved.

    Returns
    -------
    response : TYPE
        DESCRIPTION.

    '''
    global file_count, INPUT_INDEX
    #Open stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index=dev_index,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Listening to audio input...")
    audio2send = []
    cur_data = ''  # current chunk  of audio data
    rel = RATE/CHUNK
    slid_win = deque(maxlen=int(SILENCE_LIMIT * rel)+1)
    noise_win = deque(maxlen=int(NOISE_MONITOR_DUR * rel)+1)
    start_time = datetime.now()

    #Prepend audio from 50% duration before noise was detected
    prev_audio = deque(maxlen=int(PREV_AUDIO * rel)+1)

    started = False
    response = []
    file_split = 0
    while 1:
        cur_data = stream.read(CHUNK, exception_on_overflow = False)
        snr = audioop.rms(cur_data,4)
        slid_win.append(snr)
        noise_win.append(snr)
        curr_time = str(datetime.now())
        time_diff = datetime.now()-start_time
        if time_diff.seconds>=NOISE_MONITOR_DUR:
            start_time = datetime.now()
            avg_noise = sum(noise_win)/len(noise_win)
            # Calculate noise-level with respect to the 'silent' background 
            # which was measured (ideally) at night
            noise_level = 20*math.log(avg_noise/night_noise,10)
            curr_time_str = start_time.strftime("%H:%M:%S")
            print(f'Noise Level at {curr_time_str}: {noise_level} dB')
            # Store the noise level every fixed interval into a CSV file
            with open('noise_level.csv','a') as f:
                writestr = f'\n{curr_time},{noise_level}'
                f.write(writestr)
                
        # Calculate noise-level with respect to the current ambient background 
        avg_noise = sum(slid_win)/len(slid_win)
        noise_level = 20*math.log(avg_noise/bg_power,10)

        # Compare the noise in recorded audio against a threshold for recording
        # or discarding
        if(noise_level>=threshold and file_split == 0):
           if(not started):
               print("Starting file recording...")
               started = True
           audio2send.append(cur_data)
           if len(audio2send)/rel > (MAX_FILE_LENGTH - 0.5):
               file_split = 1
        elif (started is True):
            print("Finished recording.")
            # The limit was reached, finish capture
            file_len = round(len(audio2send)/rel)
            avg_noise = sum(slid_win)/len(slid_win)
            noise_level = 20*math.log(avg_noise/bg_power,10)
            if file_len>=MIN_SAVE_LENGTH:
                filename, timestr = save_speech(list(prev_audio) + audio2send, p)
                file_count = file_count + 1
                print('Saving file ',filename, 'of length', file_len, 'sec')
                with open('recorded_files_meta.csv','a') as f:
                    writestr = f'\n{timestr},{filename},{noise_level}'
                    f.write(writestr)
            else:
                print('File not saved. ', 'Duration is only', file_len, 'sec')
            # Reset all
            started = False
            slid_win = deque(maxlen=int(SILENCE_LIMIT * rel)+1)
            prev_audio = deque(maxlen=int(0.5 * rel)+1)
            audio2send = []
            file_split = 0
            if file_count % 10 == 0:
                # every ten files that are created, check wav file space usage
                print("{0} files created so far.".format(str(file_count)))
                root_directory = Path(WAV_FILE_PATH)
                wav_space = sum(f.stat().st_size for f in root_directory.glob('*.wav') if f.is_file())
                if wav_space > WAV_FILE_LIMIT:
                    print("Warning: wav files are utilizing more drive space than the specified limit!")
            print("Listening ...")
        else:
            prev_audio.append(cur_data)
    print("Finished recording.")
    stream.close()
    p.terminate()

    return response

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    noise_floor_file = 'noise_floor_night.csv'
    rms_bg = find_baseline(duration=NOISE_MONITOR_DUR)
    now = datetime.now()
    timestr = now.strftime("%H:%M:%S")
    # Uncomment the following lines to save 'silence' noise level calculated
    # ideally at night. This data will be used later to calculate noise-level
    # during daytime.
    #with open(noise_floor_file,'w') as f:
    #    writestr = f'{timestr},{rms_bg}'
    #    f.write(writestr)
    with open(noise_floor_file,'r') as f:
        content = f.readlines()
    for item in content:
        item = item.strip()
        night_noise = float(item.split(',')[-1])
    listen_for_audio(night_noise,bg_power=rms_bg,threshold=10)