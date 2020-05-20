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

         ALSO: Many functions will be copied over from project EISy, which we
         wrote in March of 2020 and is similarly useful here.

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




"""
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
---SECTION 1 : Files to Import List   -----------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
"""


def get_file_list(dir_path='../local_data', str_has=['.'], str_inc=['.'],
                  ftype='csv', interact=False):
    """
    Get a list of file paths to open, which fulfill certain criteria.
    (Alternative to single/multiselect File Dialog, or hard-coded file names
        INPUTS:
            * Path to check initially.
                (hard-coded default for now. can we globally config?)
            * Basic filter parameters, defaults just for testing now.
                str_has: AND filter (Must contain all in list)
                str_inc: OR  filter (Must contain at least one)
            * File type to check for. Default csv
            * whether to use file dialogs if the path fails, or just error out.
        OUTPUTS:
            * TUPLE of file names that pass the tests (built as list)
            * Final (successful) directory path used. (from cwd, or from base)

    """
    import os.path
    from os import listdir
    import tkinter
    from tkinter.filedialog import askdirectory
    if interact:
        root = tkinter.Tk()  # this will help control file dialog boxes!

    # Check if the path specified includes at least 1 file of the file type
    success, err_msg = check_dir_path(dir_path, [ftype], 1, False)

    while not success and len(dir_path):
        # If the "default" directory failed the check,
        # Either raise that error, or ask for a different directory!
        # (Will exit If you X out of the dialog, to avoid getting stuck)
        print("Bad Folder: <" + dir_path + ">   -   " +
              'Choose a new one, then Update your Config File!')
        if interact:
            root.lift()
            root.focus_force()
            dir_path = askdirectory(parent=root, title=err_msg,
                                    initialdir=dir_path)
            success, err_msg = check_dir_path(dir_path, [ftype], 1, False)
        else:

            raise AssertionError(err_msg)
    else:
        if interact:
            root.destroy
        print("You found a good folder at: <" + dir_path + ">")

    if not len(dir_path):
        if interact:
            root.destroy()
        raise AssertionError("You Closed the Dialog Window Without a Folder!")
    if interact:
        root.destroy()

    """
    If we've gotten this far, we found files!
    So now, we will filter the list based on the parameters given, and return
    the result as a file list to open.
    """
    # NOTE: Add File Type to File_has, so we only select that type of file
    str_has.append(ftype)

    full_dir = listdir(dir_path)
    files_wanted = []
    for file in full_dir:
        # For each file, decide if it passes
        for str_AND in str_has:
            # IF any of these fail, ignore the file.
            if str_AND in file:
                pass
            else:
                break
        else:
            # Only does this if all "Required" strings pass
            for str_OR in str_inc:
                # If ANY string is found in the file,
                # Add it to the list and then go to next file
                if str_OR in file:
                    files_wanted.append(os.path.join(dir_path, file))
                    break
                else:
                    pass
    return tuple(files_wanted), dir_path


def check_dir_path(dir_path, files_contain=['.csv'], n_required=1,
                   raise_err=False):
    """
    Check if a directory contains the files you want:
        INPUTS:
            * Path to check (required)
            * LIST of strings to check. Files must contain ALL strings to pass.
                (Default is set to look for .csv)
            * Number of successful files required to pass the test
                (Default is 1 file)
            * Failure Handling. whether to Return Failure or raise an Error.
                (Default is FALSE, which will not raise errors.)
        OUTPUTS:
            * BOOLEAN (T/F), did we find all the required files?
            * Error Message, to use in selecting a folder if we failed.
    """
    import os.path
    from os import listdir
    if os.path.isdir(dir_path):
        # First confirm that it's a directory, otherwise fail
        file_list = listdir(dir_path)
        if len(file_list) == 0:
            if raise_err:
                raise AssertionError("That Directory is Empty")
            else:
                return False, "That Directory is Empty!"
        else:
            files_found = 0
        for file in file_list:
            # Search each file name
            for str_required in files_contain:
                # To succeed, must have ALL strings in the list
                if str_required in file:
                    # If this string is in the name
                    pass  # Check the next string required
                else:
                    break  # Break out of this loop (try next file?)
            else:
                # This FOR-ELSE means the file name passed the test!
                files_found += 1  # Add 1 to found_files
                if files_found >= n_required:
                    return True, "At least "+str(n_required)+" files passed!!"
        else:
            # This FOR-ELSE means that no files passed!
            if raise_err:
                raise AssertionError(str(files_found) +
                                     " Files Passed. Needed " +
                                     str(n_required) + ".")
            else:
                return False, str(str(files_found) + " Files Passed. Needed " +
                                  str(n_required) + ".")
    else:
        if raise_err:
            raise AssertionError("This is not a Directory!")
        return False, "This Is Not a Directory"
    return False, "Something Else went Wrong? Debug..."


def ask_file_list():
    """
    Alternative to get_file_list, just makes a tkinter window and asks the user
    to select the files. Written easiy so we don't have to remember tkinter
    """
    import os.path
    import tkinter
    from tkinter.filedialog import askopenfilenames
    root = tkinter.Tk()
    files_list = askopenfilenames(multiple=True)
    root.destroy()

    return files_list, os.path.abspath(files_list[0])
