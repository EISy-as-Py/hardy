# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:03:40 2020
@author: hurtd


The zeroth-level functions, related to importing and setting up data.
    The twin goals are to be fast, but also as broad and thoughtful about
    the data we may import as possible.
 We may crib from work done in eisy/data_managment.py since we contributed
     to both project.

 __Note:__ Some of this may have Options to User-Interface,
         which is great but should DEFAULT be off, so we don't deal with it
         during Automated Processes!
         (Similarly, travis doesn't like User-Interfacing...)

 __Timeline + Milestones__:
  * 2020-04-21: Basic 2D, 3D importer that works with
                  the simple files we're using
  * 2020-04-28: Importer that works with each of the Raw Data types
                  that we're using as part of our part-1 program. __HAND OFF!__
  * 2020-05-12: Debugging and Tests that cover some basic and logical
                  use-cases (header, Too-Many-Columns, etc)
  * 2020-06-09: "Smart-ish" handling of the use cases mentioned. This should
                  be the final __HAND OFF__

 __Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work.

 __Module List__:
 #### import_2d():
  * Basic file to import a csv that has two columns
      (with or without column names?) - Essentially pd.read_csv,
      but with error handling to work better in our program and deal with index
  * INPUT: filepath/name, error-handling instructions
              (interact=False, return2=True, etc)
  * OUTPUT: x_data, y_data  <-- Two (Tuples? Series? Discuss) 1D datasets
  * ACTIONS: Checks that the file is actualy 2D data -
              if not, either asks for guidance, errors, or
              returns the first two columns depending on instructions

 #### import_3d():
  * Basic file to import a csv that has three columns
              (with or without column names?) - Essentially pd.read_csv,
              but with error handling to work better in our program
  * INPUT: filepath/name, error-handling instructions
              (interact=False, return3=True)
  * OUTPUT: x_data, y_data, z_data  <-- Three
              (Tuples? Series? Discuss) 1D datasets
  * ACTIONS: Checks that the file is actualy 3D data - if not,
              either asks for guidance, errors, or returns the
              first three columns depending on instructions

 #### get_classification_files():
  * Our project will expect a specific though simple folder structure:
              main __"/train/"__ data folder will have (2? more?)
              classification folders with the classifier names as the
              folder names - inside will be all the raw data in (csv?) format.
  * INPUT: Base folder containing classified data folders
  * OUTPUT: Dictionary, where keys are the classifications (folder names)
              and each have a (tuple/list) of file names to load
  * ACTIONS: Checks the file and data format. May try and load one/more files?
  * NOTE: Should we estimate size of all files and eventually try and
              estimate program-time? that may be useful...
"""


def import_2d():
    '''
  * Basic file to import a csv that has two columns
      (with or without column names?) - Essentially pd.read_csv,
      but with error handling to work better in our program and deal with index
  * INPUT: filepath/name, error-handling instructions
              (interact=False, return2=True, etc)
  * OUTPUT: x_data, y_data  <-- Two (Tuples? Series? Discuss) 1D datasets
  * ACTIONS: Checks that the file is actualy 2D data -
              if not, either asks for guidance, errors, or
              returns the first two columns depending on instructions
     '''


def import_3d():
    """
  * Basic file to import a csv that has three columns
              (with or without column names?) - Essentially pd.read_csv,
              but with error handling to work better in our program
  * INPUT: filepath/name, error-handling instructions
              (interact=False, return3=True)
  * OUTPUT: x_data, y_data, z_data  <-- Three
              (Tuples? Series? Discuss) 1D datasets
  * ACTIONS: Checks that the file is actualy 3D data - if not,
              either asks for guidance, errors, or returns the
              first three columns depending on instructions
    """


def get_classification_files():
    """
  * Our project will expect a specific though simple folder structure:
              main __"/train/"__ data folder will have (2? more?)
              classification folders with the classifier names as the
              folder names - inside will be all the raw data in (csv?) format.
  * INPUT: Base folder containing classified data folders
  * OUTPUT: Dictionary, where keys are the classifications (folder names)
              and each have a (tuple/list) of file names to load
  * ACTIONS: Checks the file and data format. May try and load one/more files?
  * NOTE: Should we estimate size of all files and eventually try and
              estimate program-time? that may be useful...
    """
