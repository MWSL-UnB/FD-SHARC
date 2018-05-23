# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:34:47 2018

@author: Calil
"""

import os
import glob

directories = [x[0] for x in os.walk('.')]

for dr in directories:
    if 'output_2018-05-21' in dr:
        ini_files =  glob.glob(os.path.join(dr,"*.ini"))
        
        file = ini_files[0]
        file  = os.path.basename(file)
        file = file.replace('_parameters.ini','')
        
        os.rename(dr,file)