#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:04:38 2022

@author: clement
"""

from Start_detection_audio import *

import os
import json
import cupy as cp
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def dicotiser(f):
    with open(f, encoding = 'utf-8') as json_data:
        return(json.load(json_data))
    
def normaliser(signal):#signal en cupy array, renvoie une cupy array
    return(signal/cp.std(signal))

def traitement(s_beep_numpy,s_ref_44100,s_ref_22050,fs,fs_beep):
    
        if len(np.shape(s_beep_numpy)) == 2:
            s_beep_numpy = s_beep_numpy[:,0]
        s_beep = normaliser(cp.array(s_beep_numpy))
        if fs_beep == 44100:
           max_intercorr = cp.max(cp.correlate(s_beep,s_ref_44100)).get()*len(s_ref_44100)/min(len(s_beep_numpy),len(s_ref_44100))
        elif fs_beep == 22050:
            max_intercorr = cp.max(cp.correlate(s_beep,s_ref_22050)).get()*len(s_ref_44100)/min(len(s_beep_numpy),len(s_ref_22050))#on normalise pour comparer
        else:
            print("Fréquence particuliere:",fs_beep)
            ss_ref = scipy.signal.decimate(s_ref_44100_numpy,fs_beep/fs)
            max_intercorr = cp.max(cp.correlate(s_beep,normaliser(cp.array(ss_ref)))).get()*len(ss_44100)/min(len(s_beep_numpy),len(ss_ref))
        
        return(max_intercorr)
    


if __name__ == "__main__":
    
    
    folder = "./data_audio"
    
    # récupération du buzzer de référence
    bip_ref_path="ref_bip_isolated.wav"
    fs, s_ref_numpy = aIO.read_audio_file(bip_ref_path)
    s_ref_44100_numpy = s_ref_numpy[4851:-1]#on enleve le bruit blanc
    s_ref_22050_numpy = scipy.signal.decimate(s_ref_44100_numpy,2)

    s_ref_44100 = normaliser(cp.array(s_ref_44100_numpy))
    s_ref_22050 = normaliser(cp.array(s_ref_22050_numpy))
    
    #récupération des fichiers de donnée
    
    
    walker = os.walk(folder)
    
    beeps = []
    non_beeps = []
    beeps_created = []
    
    for dirpath,dirnames,filenames in walker:
        dirname = dirpath.split('/')[-1]
        if dirname == "beeps":
            for filename in filenames:
                beeps.append(dirpath+ '/'+ filename)
        if dirname == "beeps_created":
            for filename in filenames:
                beeps_created.append(dirpath+ '/'+ filename)

        if dirname == "non-beeps":
            for filename in filenames:
                non_beeps.append(dirpath+ '/'+ filename)


    rep =""
    
    #Traitement des données
    
    print("Traitement des beeps")
    beeps_vide =[]
    
    rep_beeps = []
    lens_beeps = []
    for beep in beeps:
        fs_beep, s_beep_numpy = aIO.read_audio_file(beep)
        lens_beeps.append(len(s_beep_numpy))
        
        if np.any(s_beep_numpy):
        
            max_intercorr = traitement(s_beep_numpy,s_ref_44100,s_ref_22050,fs,fs_beep)
            
            rep += 'beep'+','+beep.split('/')[-1] + ',' + str(max_intercorr)+'\n'
            rep_beeps.append(max_intercorr)
        else:
            beeps_vide.append(beep)
            
     
    
     
    print("Traitement des beeps_created")
    rep_beeps_created = []
    lens_beeps_created = []
    for beep in beeps_created:
        fs_beep, s_beep_numpy = aIO.read_audio_file(beep)
        lens_beeps_created.append(len(s_beep_numpy))
        
        if np.any(s_beep_numpy):
               
            max_intercorr = traitement(s_beep_numpy,s_ref_44100,s_ref_22050,fs,fs_beep)
            rep += 'beep_created'+','+beep.split('/')[-1] + ',' + str(max_intercorr)+'\n'
            rep_beeps_created.append(max_intercorr) 
            if np.isnan(max_intercorr):
                print(beep)
        else:
            beeps_vide.append(beep)
            
        
    print("Traitement des non-beeps")
    rep_non_beeps = []
    lens_non_beeps = []
    for beep in non_beeps:
        fs_beep, s_beep_numpy = aIO.read_audio_file(beep)
        lens_non_beeps.append(len(s_beep_numpy))
        
        if np.any(s_beep_numpy):
        
            max_intercorr = traitement(s_beep_numpy,s_ref_44100,s_ref_22050,fs,fs_beep)
            
            rep += 'non_beep'+','+beep.split('/')[-1] + ',' + str(max_intercorr)+'\n'
            rep_non_beeps.append(max_intercorr)
        else:
            beeps_vide.append(beep)
            
   
    print('=================')
    print()     
    print('Beeps vide:')
    print('-----------------')
    for beep in beeps_vide:
        print(beep)
    print('-----------------')   
   
    
    # plot:
        
        
    pos_data = [1,2,3]
    color_data = ['b','g','r']
    labels = ['Beeps', 'Beeps created','Non beeps']
    
    data = [np.log(rep_beeps),np.log(rep_beeps_created),np.log(rep_non_beeps)]
    fig, ax = plt.subplots()

    vp = ax.violinplot(data,pos_data,
                       showmeans=False, showmedians=False, showextrema=False)
    
    for i,v in enumerate(vp['bodies']):
        v.set_facecolor(color_data[i])
    
    for i in range(3):
        ax.scatter(np.ones_like(data[i])*pos_data[i],data[i],c= color_data[i],marker = '.',linewidths = 0.01)
        
    ax.set_xticks(pos_data)
    ax.set_xticklabels(labels)
    
    ax.set_ylabel('Log max cross correlation')
    
    ax.set_title("Etude du jeu de données")
    
    plt.show()
    
    
 

