#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:37:08 2022

@author: clement
"""

from pyAudioAnalysis import MidTermFeatures as aFm
from pyAudioAnalysis import audioBasicIO as aIO
import moviepy.editor as mp
import numpy as np
import argparse
import json

import matplotlib.pyplot as plt
import cupy as cp
from scipy.signal import butter, lfilter, freqz

import time

#filtrage numérique

def butter_lowpass(cutoff, fs, order=3):#filtre passe bas
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(cutoff1,cutoff2,fs,order = 3):#filtre passe bande
    nyq = 0.5*fs
    b,a = butter(order,(cutoff1/nyq,cutoff2/nyq),"bandpass")
    return b,a

def butter_bandpass_filter(data,cutoff1,cutoff2,fs,order = 3): #pour appliquer le filtre
    b,a = butter_bandpass(cutoff1,cutoff2,fs,order = order)
    y = lfilter(b,a,data)
    return(y)

def butter_lowpass_filter(data, cutoff, fs, order=5):#pour appliquer le filtrage
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def spectre(signal,fs = 44100):#dessine le spectre
    T = len(signal)/fs
    tfd = np.fft.fft(signal)
    N=len(signal)
    spectre = np.absolute(tfd)*2/N
    freq=np.arange(N)*1.0/T
    plt.plot(freq,spectre)
    plt.axis([-0.1,fs/2,0,spectre.max()])

def get_index(list_dict, vid_name):
    """helper to read the json file."""
    for i in range(len(list_dict)):
        if list_dict['videos'][i]['name'] == vid_name:
            return i


def extract_time_start(video_path, bip_ref_path="ref_bip_isolated.wav", references_path="ref_features_bip.npy"):
    mt = np.load(references_path)
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5

    # extraction on the long signal
    my_clip1 = mp.VideoFileClip(video_path)
    fs = 44100
    s_long = my_clip1.audio.to_soundarray(fps=fs)
    s_long = s_long[:, 0]
    duration_long = len(s_long) / float(fs)

    # extract short-term features using a 50msec non-overlapping windows
    win, step = 0.05, 0.05
    win_mid, step_mid = 0.4, 0.05
    mt_long, st_long, mt_n_long = aFm.mid_feature_extraction(s_long, fs, win_mid * fs, step_mid * fs,
                                                             win * fs, step * fs)

    # normalization
    mt_long = mt_long.T
    for i in range(len(mt_long)):
        mt_long[i] = mt_long[i] / np.linalg.norm(mt_long[i])

    temps_possible = []

    for i in range(len(mt)):
        arg_min_dist = 0
        min_dist = 1000
        for j in range(len(mt_long)):
            if np.linalg.norm(mt[i] - mt_long[j]) < min_dist:
                arg_min_dist = j
                min_dist = np.linalg.norm(mt[i] - mt_long[j])
        temps_possible.append(arg_min_dist * duration_long / mt_long.shape[0])

    median_time = np.median(temps_possible)
    temps_possible_non_aberrant = []
    aberration = 0.5
    for i in range(len(temps_possible)):
        if median_time - aberration <= temps_possible[i]:
            if temps_possible[i] <= median_time + aberration:
                temps_possible_non_aberrant.append(temps_possible[i])

    if temps_possible_non_aberrant != []:
        # 0.11s de silence avant le bip dans les sons de références
        start = np.median(temps_possible_non_aberrant) + 0.11

    else:
        # Erreur
        start = -1
    return start


def continuous_normalisation(signal,windows):
    kernel = cp.ones(windows)/windows
    mean = cp.convolve(signal,kernel,"same")
    carre = (signal - mean)**2

    std = cp.sqrt(cp.convolve(carre,kernel,"same"))
    std[std==0] = 1
    rep = signal/std
    
    return(rep)

def extract_time_start_intercorr(video_path, bip_ref_path="ref_bip_isolated.wav"):
    
    fs, s_ref = aIO.read_audio_file(bip_ref_path)

    s_ref = s_ref[4851:-1]#on enleve le bruit blanc
    s_ref = butter_bandpass_filter(s_ref,870, 1100, fs, order=3)
    audioclip = mp.AudioFileClip(video_path)
    
    signal = audioclip.to_soundarray(fps = fs)[:,0]
    #plt.plot(np.arange(0,len(signal))/fs,signal)
    #plt.title('signal')
    
    #filtrage du signal
    signal = butter_bandpass_filter(signal,870, 1100, fs, order=3)
    #plt.plot(np.arange(0,len(signal))/fs,signal)
    
    #passage sur carte graphique
    
    signal= cp.array(signal)

    #signal =  continuous_normalisation(signal,3*len(s_ref))
    #plt.figure()
    #plt.plot(np.arange(0,len(signal))/fs,signal.get())
    #plt.title("apres normalisation continue")


    intercorr = cp.correlate(signal,cp.array(s_ref))
    #plt.figure()
    #plt.plot(np.arange(0,len(intercorr))/fs,intercorr.get())
    time_start = cp.argmax(intercorr,axis=0)/fs #on ajoute les 0.11 seconde de bkanc. On pourrait être plus précis au regard de la précision de l'intercorr
    
    return(float(time_start)) #la récupération de time_start en float ou en numpy.ndarray est extrèmement lente. Je ne comprends pas

def extract_time_start_naif(video_path,bip_ref_path="ref_bip_isolated.wav"):
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    s_ref = s_ref[4851:-1]#on enleve le bruit blanc
    audioclip = mp.AudioFileClip(video_path)
    fs = 44100
    signal = audioclip.to_soundarray(fps = fs)[:,0]
    
    signal =  continuous_normalisation(cp.array(signal),3*len(s_ref)).get()
    
    signal = butter_bandpass_filter(signal,870, 1100, fs, order=3)
    
    
    return(np.argmax(np.absolute(signal))/fs)
    
def synchro_videos(video_path1,video_path2):
    fs = 44100
    
    audioclip1 = mp.AudioFileClip(video_path1)
    signal1 = audioclip1.to_soundarray(fps = fs)[:,0]
    
    audioclip2 = mp.AudioFileClip(video_path2)
    signal2 = audioclip2.to_soundarray(fps = fs)[:,0]
    
    if len(signal1) >= len(signal2):
        return(float(cp.argmax(cp.correlate(cp.array(signal1),cp.array(signal2)))/fs))
    else:
        return(float(cp.argmax(cp.correlate(cp.array(signal2),cp.array(signal1)))/fs))
if __name__ == "__main__":
    """
    video_path ='./data/2021_GT_Marseille/2021_Marseille_dos_dames_50_serie3/2021_Marseille_dos_dames_50_serie3_fixeDroite.mp4'
    bip_ref_path="ref_bip_isolated.wav"
    print(extract_time_start_naif(video_path))
    """
    video_path1 ='./data/2021_GT_Marseille/2021_Marseille_freestyle_hommes_50_serie5/2021_Marseille_freestyle_hommes_50_serie5_fixeDroite.mp4'
    video_path2 ='./data/2021_GT_Marseille/2021_Marseille_freestyle_hommes_50_serie5/2021_Marseille_freestyle_hommes_50_serie5_fixeGauche.mp4'
    print(synchro_videos(video_path2,video_path1))