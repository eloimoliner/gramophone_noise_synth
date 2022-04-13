import ast

#import tensorflow as tf
import random
import os
import numpy as np
import soundfile as sf
import math
import pandas as pd
import glob
from tqdm import tqdm
import torch

#generator function. It reads the csv file with pandas and loads the largest audio segments from each recording. If extend=False, it will only read the segments with length>length_seg, trim them and yield them with no further processing. Otherwise, if the segment length is inferior, it will extend the length using concatenative synthesis.

        

class TrainNoiseDataset (torch.utils.data.IterableDataset):
    def __init__(self,   path_noises,  fs=22050, seg_len=34816,seed=42 ):
        super(TrainNoiseDataset).__init__()
        random.seed(seed)
        np.random.seed(seed)
       
        self.seg_len=int(seg_len)
        #print(path_noises)
        noises_info=os.path.join(path_noises[0],"info.csv")
        self.noise_generator=noise_sample_generator(noises_info,fs, self.seg_len,  "train", seed=seed) #this will take care of everything
        self.fs=fs

    def __iter__(self):
        while True:
    
            #scale=np.random.uniform(-3,3)
    
            #load noise signal
            data_noise= next(self.noise_generator)
            if len(data_noise.shape)>1:
                 data_noise=np.mean(data_noise,axis=1)
                
            #normalize
            data_noise=data_noise/np.max(np.abs(data_noise))
            new_noise=data_noise #if more processing needed, add here
            #estimate noise power
            #rms_noise=np.sqrt(np.var(new_noise))
            b=1.4826
            rms_noise=np.sqrt(b**2 * np.median(new_noise**2))
            
            new_noise= (0.1/rms_noise)*new_noise #starting at -10 db
      
            noise_signal=new_noise #not sure if this is correct, maybe revisit later!!
            #noise_signal=10.0**(scale/10.0) *noise_signal #from -14dB to -6dB. Hope it is enough
            noise_signal=noise_signal.astype('float32')
            
            yield noise_signal


def noise_sample_generator(info_file,fs, length_seq, split, seed=43):
    random.seed(seed)
    head=os.path.split(info_file)[0]
    load_data=pd.read_csv(info_file)
    #split= train, validation, test
    #load_data_split=load_data.loc[load_data["split"]==split]
    #load_data_split=load_data_split.reset_index(drop=True)
    while True:
        #r = list(range(len(load_data_split)))
        i=np.random.randint(0,len(load_data))
        #for i in r:
        segments=ast.literal_eval(load_data.loc[i,"segments"])
        num=np.random.randint(0,len(segments))
        loaded_data, Fs=sf.read(os.path.join(head,load_data["recording"].loc[i],segments[num]))
        assert round(fs)==Fs, "wrong sampling rate"
        assert len(loaded_data)>=length_seq, "too small!!"
        #print(len(loaded_data), length_seq)
        idx=np.random.randint(0,len(loaded_data)-length_seq)
        yield loaded_data[idx:idx+length_seq]

def __extend_sample_by_repeating(data, fs,seq_len):        
    rpm=78
    target_samp=seq_len
    large_data=np.zeros(shape=(target_samp,2))
    
    if len(data)>=target_samp:
        large_data=data[0:target_samp]
        return large_data
    
    bls=(1000*44100)/1000 #hardcoded
    
    window=np.stack((np.hanning(bls) ,np.hanning(bls)), axis=1) 
    window_left=window[0:int(bls/2),:]
    window_right=window[int(bls/2)::,:]
    bls=int(bls/2)
    
    rps=rpm/60
    period=1/rps
    
    period_sam=int(period*fs)
    
    overhead=len(data)%period_sam
    
    if(overhead>bls):
        complete_periods=(len(data)//period_sam)*period_sam
    else:
        complete_periods=(len(data)//period_sam -1)*period_sam
    
    
    a=np.multiply(data[0:bls], window_left)
    b=np.multiply(data[complete_periods:complete_periods+bls], window_right)
    c_1=np.concatenate((data[0:complete_periods,:],b))
    c_2=np.concatenate((a,data[bls:complete_periods,:],b))
    c_3=np.concatenate((a,data[bls::,:]))
    
    large_data[0:complete_periods+bls,:]=c_1
    
    
    pointer=complete_periods
    not_finished=True
    while (not_finished):
        if target_samp>pointer+complete_periods+bls:
            large_data[pointer:pointer+complete_periods+bls] +=c_2
            pointer+=complete_periods
        else: 
            large_data[pointer::]+=c_3[0:(target_samp-pointer)]
            #finish
            not_finished=False

    return large_data
