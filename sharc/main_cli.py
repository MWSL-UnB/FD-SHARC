# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:17:14 2017

@author: edgar
"""

import sys, getopt
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from sharc.model import Model
from sharc.gui.view_cli import ViewCli
from sharc.controller import Controller
from sharc.support.logging import Logging


def main(argv):
    print("Welcome to SHARC!\n")
    
    param_file = ''

    try:
        opts, args = getopt.getopt(argv, "h:p:o:")
    except getopt.GetoptError:
        print("usage: main_cli.py -p <param_file> -o <output_directory>")
        sys.exit(2)

    param_file = os.path.join(os.getcwd(), "parameters", "parameters.ini")
    out_dir = os.path.join(os.getcwd(),"output")
    if opts:
        for opt, arg in opts:
            if opt == "-h":
                print("usage: main_cli.py -p <param_file> -o <output_directory>")
                sys.exit()
            elif opt == "-p":
                param_file = os.path.join(os.getcwd(), arg)
            elif opt == "-o":
                out_dir = os.path.join(os.getcwd(), arg)
            
    Logging.setup_logging()
    
    model = Model()
    view_cli = ViewCli()
    controller = Controller()
    
    view_cli.set_controller(controller)
    controller.set_model(model)
    model.add_observer(view_cli)
    
    view_cli.initialize(param_file,out_dir)



if __name__ == "__main__":
    main(sys.argv[1:])