#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:01:54 2022

@author: clement
"""





import os
import json
import cupy as cp
import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt
import ffmpeg
from ffmpeg import Error
import requests


def dicotiser(f):
    with open(f, encoding = 'utf-8') as json_data:
        return(json.load(json_data))

def read_json_dataroom(url):
    r = requests.get(url)
    return r.json()


def getruns4compet(compet):
    base = "https://dataroom.liris.cnrs.fr/vizvid_json/pipeline-tracking/"
    data = read_json_dataroom(base + compet)

    return [d["name"] for d in data if d['type'] == "directory" and d["name"][:3] == "202"]


def trouver_JSON(filenames):
    rep = []
    for filename in filenames:
        if '.json' in filename:
            rep.append(filename)
    return(rep)

class ffmpegProcessor:
    def __init__(self):
        self.cmd = '/usr/bin/ffmpeg'

    def extract_audio(self, filename):
        try:
            probe = ffmpeg.probe(filename)
            out, err = (
                ffmpeg
                    .input(filename)
                    .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar='44100')
                    .run(cmd=self.cmd, capture_stdout=True, capture_stderr=True)
            )
        except Error as err:
            print(err.stderr)
            raise

        return probe,np.frombuffer(out, np.float32)   


if __name__ == "__main__":
    destination = './Pistes'
    
    print('Récupération des url des courses...')

    
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
    flashStarts = {}
    for i,run in enumerate(runs):
        print(f'{100*i/len(runs)} %')
        course_url = base_url + compet + "/" + run + "/"
        data = read_json_dataroom(course_url + run + ".json")
        meta_right = [d for d in data['videos'] if "Droite" in d["name"]][0]
        meta_left = [d for d in data['videos'] if "Gauche" in d["name"]][0]
        
        meta = meta_right if "start_flash" in meta_right else meta_left
        
        if 'start_flash' in meta and meta['start_flash'] != 0:
            probe,signal = ap.extract_audio(course_url+meta["name"])
            fs = int(probe['streams'][1]['sample_rate'])
            scipy.io.wavfile.write(os.path.join(destination,run+".wav"),fs,signal)
            flashStarts[run+".wav"] = meta['start_flash']
        
    
    with open(os.path.join(destination,'Starts.json'), 'w', encoding='utf-8') as f:
        json.dump(flashStarts, f, ensure_ascii=False, indent=4)
    
            


