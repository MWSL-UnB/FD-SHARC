# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:22:02 2018

@author: Calil
"""

import os
from configparser import ConfigParser

frequencies = ['50',
               '40']

directions = ['DL', 
              'UL']

distances = [0,
             100,
             200,
             400,
             600,
             800,
             1000]

temp_file = 'temp.ini'
try: os.remove(temp_file)
except OSError: pass

# Distance from cluster center to edge
c_to_e_dist = 791

for freq in frequencies:
    for dire in directions:
        for dist in distances:
            
            if dist == 0 : dist_str = 'center'
            else: dist_str = str(dist)
            
            old_file_name = freq + '_' + dire + '_' + dist_str + '_parameters.ini'
            new_file_name = freq + '_' + dire + '_' + str(dist) + '_parameters.ini'
            
            os.rename(old_file_name,temp_file)
            
            config = ConfigParser()
            config.read(temp_file)
            
            config.set('FSS_ES','x',str(c_to_e_dist + dist))
            
            with open(new_file_name, 'w') as configfile:
                config.write(configfile)
                
            os.remove('temp.ini')
            