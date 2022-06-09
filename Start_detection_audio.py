#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:37:08 2022

@author: clement
"""

from pyAudioAnalysis import MidTermFeatures as aFm
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioTrainTest as aT
import moviepy.editor as mp
import numpy as np
import argparse
import json

import matplotlib.pyplot as plt
import cupy as cp
from scipy.signal import butter, lfilter, freqz, find_peaks

import time

import pyAnalysismodifie as pam

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


def extract_time_start(video, bip_ref_path="ref_bip_isolated.wav", references_path="ref_features_bip.npy"):
    mt = np.load(references_path)
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5

    # extraction on the long signal
    if type(video)== str:
        #récupération du signal
        audioclip = mp.AudioFileClip(video)
        s_long = audioclip.to_soundarray(fps = fs)[:,0]
        del audioclip #pas besoin de garder la vidéo en mémoire
    else:
        s_long = video

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

def extract_time_start_intercorr(video, bip_ref_path="ref_bip_isolated.wav"):
    
    fs, s_ref = aIO.read_audio_file(bip_ref_path)

    s_ref = s_ref[4851:-1]#on enleve le bruit blanc
    s_ref = butter_bandpass_filter(s_ref,870, 1100, fs, order=3)
    
    
    if type(video)== str:
        #récupération du signal
        audioclip = mp.AudioFileClip(video)
        signal = audioclip.to_soundarray(fps = fs)[:,0]
        del audioclip #pas besoin de garder la vidéo en mémoire
    else:
        signal = video
    
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

def extract_time_start_naif(video,bip_ref_path="ref_bip_isolated.wav"):
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    s_ref = s_ref[4851:-1]#on enleve le bruit blanc
    
    if type(video)== str:
        #récupération du signal
        audioclip = mp.AudioFileClip(video)
        signal = audioclip.to_soundarray(fps = fs)[:,0]
        del audioclip #pas besoin de garder la vidéo en mémoire
    else:
        signal = video

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
    

def fenetres(signali, s_ref, filtrage = True,fs = 44100,plot_peaks = False):
    
    if filtrage:
        s_ref = butter_bandpass_filter(s_ref,870, 1100, fs, order=3)#filtrage du bruit de référence. On le cherche dans le signal filtré
        signal = butter_bandpass_filter(signali,870, 1100, fs, order=3)#filtrage du signal
        

    signal= cp.array(signal)#passage sur carte graphique

    #traitement
    intercorr = cp.correlate(signal,cp.array(s_ref))
    intercorr = intercorr/cp.std(intercorr)#on normalise pour pouvoir définir un seuil qui fait sens
    intercorr = intercorr.get()
    peaks,_ = find_peaks(intercorr,threshold = 0.01,distance = len(s_ref))#c'est le coeur de l'algorithme, il faut choisir les bons paramètres pour prendre les bons pics, pour l'instant on pren dune distance de la longueur de s_ref, car on ne veut pas tous les pics mais le plus gros qui correspond au signal du départ
    #Il vaut mieux prendre bcp de pics que pas assez. Le départ doit toujours être pris par la détection de pics. L'algorithme de classification fera le tri plus tard
    if plot_peaks:
    
        plt.figure()
        plt.plot(np.arange(0,len(intercorr))/fs,intercorr)
        plt.plot(peaks/fs,intercorr[peaks],"xr")
        
    
    return(peaks)

def extract_time_start2(video,model_type,bip_ref_path="ref_bip_isolated.wav",filtrage = True,plot_peaks = False,plot_scores =False):
        
    #récupération du beep de référence
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    s_ref = s_ref[4851:-1]#on enleve le bruit blanc au début
    
    
    if type(video)== str:
        #récupération du signal
        audioclip = mp.AudioFileClip(video)
        signal = audioclip.to_soundarray(fps = fs)[:,0]
        del audioclip #pas besoin de garder la vidéo en mémoire
    else:
        signal = video

    peaks = fenetres(signal,s_ref,fs,plot_peaks = plot_peaks)
    temp = []
    
    for peak in peaks:
        temp.append(pam.file_classification(fs,signal[peak-100:peak + int(0.3*fs)], model_type+"StartDetector",model_type)[1][0])
    start_time = peaks[np.argmax(temp)]/fs
    
    if plot_scores:

        plt.figure()
        plt.plot(peaks/fs,np.log(temp),'x')
    
    return(start_time)
    

def genererModele(list_classes,model_type,folder):
    aT.extract_features_and_train(list_classes, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep,model_type,folder + '/'+model_type+"StartDetector", False)
    
    

if __name__ == "__main__":
    
    video_path ='./data_videos/2021_CF_Montpellier/2021_CF_Montpellier_freestyle_hommes_50_FinaleA/2021_CF_Montpellier_freestyle_hommes_50_FinaleA_fixeDroite.mp4'
    bip_ref_path="ref_bip_isolated.wav"
    print(extract_time_start2(video_path,'svm',plot_peaks = False,plot_scores = False))
    

    #pour entrainer les modèles de classification
    """
    list_classes = ['./data_audio/entrainement/beeps','./data_audio/entrainement/non-beeps']
    aT.extract_features_and_train(list_classes, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "temp", False)
    """
    