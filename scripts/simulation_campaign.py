# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 19:03:36 2018

@author: Calil
"""

import os
import glob
from traceback import print_tb
from sharc.main_cli import main
from sys import stdout

def log_message(message: str):
    print(message)
    stdout.flush()
    with open('simulation_campaign.log','a') as clog:
        clog.write(message)
        clog.flush()

log_message("_____SIMULATION SCRIPT_____\n")
log_message("Setting up..\n")

# Setup paths
cases_folder = os.path.join('..','cases')
subfolders = [f.path for f in os.scandir(cases_folder) if f.is_dir()]

log_message("Beginning simulation cases...")

for k, folder in enumerate(subfolders):
    case = os.path.basename(os.path.normpath(folder))
    log_message("\n\nCURRENT CASE: " + case)
    # Get ini file
    files = glob.glob(os.path.join(folder,'*.ini'))
    if len(files) == 0:
        log_message('\nWarning: no configuration file in case ' + case + 
                   '. Going to next folder')
        continue
    elif len(files) > 1: 
        log_message('\nWarning: more than one configuration file in case ' + case + 
                       '. Using file: ' + os.path.basename(os.path.normpath(files[0])))
    
    # Run simulation
    file = files[0]
    try:
        main(['-p',file,'-o',folder])
    except Exception as e:
        log_message(str(e) + "\nTraceback: ")
        print_tb(e.__traceback__)
        log_message("\n Moving on...")
            
    log_message("\n" + case + " case finished." + str(len(subfolders) - k - 1)\
               + " cases to go.\n")
    
log_message('DONE SIMULATING')
