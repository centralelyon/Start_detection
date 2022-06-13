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
    


def traitement2(s_beep_numpy,ts_ref_44100,ts_ref_22050,fs,fs_beep):
    
        if len(np.shape(s_beep_numpy)) == 2:
            s_beep_numpy = s_beep_numpy[:,0]
        s_beep = normaliser(cp.array(s_beep_numpy))
        if fs_beep == 44100:
            temp = []
            for beep_ref in ts_ref_44100:
                temp.append(cp.max(cp.correlate(s_beep,beep_ref)).get()*len(beep_ref)/min(len(s_beep_numpy),len(beep_ref)))
            max_intercorr = np.median(temp)
        elif fs_beep == 22050:
            temp = []
            for beep_ref in ts_ref_22050:
                temp.append(cp.max(cp.correlate(s_beep,beep_ref)).get()*len(beep_ref)/min(len(s_beep_numpy),len(beep_ref)))
            max_intercorr = c=np.median(temp)
        else:
            print("Fréquence particuliere:",fs_beep)
            ss_ref = scipy.signal.decimate(s_ref_44100_numpy,fs_beep/fs)
            max_intercorr = cp.max(cp.correlate(s_beep,normaliser(cp.array(ss_ref)))).get()*len(ss_44100)/min(len(s_beep_numpy),len(ss_ref))
        
        return(max_intercorr)



def trouver_JSON(filenames):
    rep = []
    for filename in filenames:
        if '.json' in filename:
            rep.append(filename)
    return(rep)



if __name__ == "__main__":
    
    print('Récupération des url des courses...')

    ap = ffmpegProcessor()
    
    base_url = "https://dataroom.liris.cnrs.fr/vizvid_json/pipeline-tracking/"
    # compet = "2021_CF_Montpellier"
    compet = "2022_CF_Limoges"

    runs = getruns4compet(compet)
    # print(runs)
    # run = "2022_CF_Limoges_papillon_hommes_50_finaleA"
    # sec = make_sections()
    # lon = make_longueurs()
    ap = ffmpegProcessor()

    # -----------------------------------
    courseslinks = []
    flashStarts = []
    for run in runs:
        course_url = base_url + compet + "/" + run + "/"
        data = read_json_dataroom(course_url + run + ".json")
        meta_right = [d for d in data['videos'] if "Droite" in d["name"]][0]
        meta_left = [d for d in data['videos'] if "Gauche" in d["name"]][0]
        
        meta = meta_right if "start_flash" in meta_right else meta_left
        
        if 'start_flash' in meta and meta['start_flash'] != 0:
            courseslinks.append(course_url+meta["name"])
            flashStarts.append(meta['start_flash'])

    
    n_vid = len(courseslinks)
    print(f'Benchmarking {n_vid} videos:')
    
    model_types = ["flash","naif","intercorr","implemente","knn","svm","svm_rbf","extratrees","gradientboosting","randomforest"]


    rep = ","
    for nom in model_types:
        rep += nom +','
    rep = rep[0:-1]
    rep += '\n'
    
    for i,video_url in enumerate(courseslinks):
        video_name = video_url.split('/')[-1]
        print('---------------------------------- ' + str(i) + ' i.e: ' + str(100*i/n_vid) + '%')
        print(video_name )
        rep += video_name + ','
        print('Downloading....')
        signal = ap.extract_audio(video_url)
        print('Computing...')
        for j,model in enumerate(model_types):
            print(j)
            if model == "flash":
                time = flashStarts[i]
            elif model == "naif":
                time = extract_time_start_naif(signal)
            elif model == "intercorr":
                time = extract_time_start_intercorr(signal)
            elif model == "implemente":
                time = extract_time_start(signal)
            else:
                time = extract_time_start2(signal,model)
            
            rep += str(time)
            if j != len(model_types)-1:
                 rep += ','
            else:
                rep += '\n'

        

    print(rep)
    
    Bench = open('Bench','a')
    Bench.write(rep)
    Bench.close()
    
