import os.path
import tkinter
import pickle

import pandas as pd

from os import listdir
from tkinter.filedialog import askdirectory


def get_file_list(dir_path='../local_data', str_has=['.'], str_inc=['.'],
                  ftype='csv', interact=False):
    """
    Get a list of file paths to open, which fulfill certain criteria.
    (Alternative to single/multiselect File Dialog, or hard-coded file names

    Parameters
    ----------
    dir_path: str
              Path to check initially.
              (hard-coded default for now. can we globally config?)
    str_has: numpy.array
             Basic filter parameters, defaults just for testing now.
             str_has: AND filter (Must contain all in list)
    str_inc: numpy.array
             OR  filter (Must contain at least one)
    ftype: str
           File type to check for. Default csv
    interact: bool
              whether to use file dialogs if the path fails, or just error out.

    Returns
    -------
    files_wanted: tuple
                  TUPLE of file names that pass the tests (built as list)
    dir_path: str
              Final (successful) directory path used. (from cwd, or from base)

    """

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

    Parameters
    ----------
    dir_path: str
        Path to check (required)
    files_contain: list
        list of strings to check. Files must contain ALL strings to pass.
                (Default is set to look for .csv)
    n_required: int
        Number of successful files required to pass the test
                (Default is 1 file)
    raise_err: bool
        Failure Handling. whether to Return Failure or raise an Error.
                (Default is FALSE, which will not raise errors.)
    Returns
    -------
    BOOLEAN (T/F), did we find all the required files?
    Error Message, to use in selecting a folder if we failed.
    """
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
    from tkinter.filedialog import askopenfilenames
    root = tkinter.Tk()
    files_list = askopenfilenames(multiple=True)
    root.destroy()
    the_dir, the_file = os.path.split(files_list[0])
    return files_list, the_dir


def classes_from_fnames(file_list=None, path=None, expect=2, print_ok=True,
                        from_serials=False):
    """
    Given a list of file names, determine if there are classifying endings
    that split the data into "expect" (default 2) Groups.

    Paramters
    ---------
    file_list: list
        list containing the raw data files
    path: str
        path to the folder containg the raw data files
    expect: int
        number representing how many classes the data is expected to be
        classified into
    print_ok: bool
        if True, returns print statements indicating complition of steps
        within the function
    from_serials: bool
        if True uses the serial IDs instead of file names

    Returns
    -------
    classification_list: list
        list containing the classes/labels to separate the data in.
    """
    classification_list = []
    populations = {}  # Dictionary

    # Get file list either passed, or from path
    if not file_list and not path:
        raise AssertionError("Need either File List OR Path")
    elif file_list and path:
        print("Given List of Fnames, will Ignore Path")
    elif path:
        file_list = os.listdir(path)
    else:
        pass
    n_files = len(file_list)

    # Downselect for CSVs
    csv_files = []
    for file in file_list:
        if file.endswith('.csv'):
            csv_files.append(file)
        else:
            pass
    if print_ok:
        print("From {} Files:".format(n_files))
        print("Found {} CSVs...".format(len(csv_files)))

    if from_serials:
        # Workaround for using serial IDs instead of file names
        if print_ok:
            print("Using Serial IDs Instead!")
        csv_files = file_list

    for file in csv_files:
        str_end = len(file)
        str_start = 0
        i = len(file)

        while str_start == 0 and i >= 0:

            # Loop until you find the string starting underscore,
            # Or until you are at the beginning (which would be a fail)
            i -= 1  # Step back into filename
            if file[i] == '.':  # if you find the file extension dot
                str_end = i
            elif file[i] == '_':  # first underscore you find
                str_start = i+1  # String starts AFTER the underscore

        if i == 0:  # If you reached the start of the file with no "_"
            if print_ok:
                print("File {} \t has no underscore - cannot classify")
            else:
                raise AssertionError(
                    "File {} \t has no underscore - cannot classify")
            # FOR NOW, Allow and ignore??
        elif str_start == 0 or str_end == 0:
            if print_ok:
                print("File {} \t has no label - cannot classify")
            else:
                raise AssertionError(
                    "File {} \t has no label - cannot classify")
        else:
            # IF file name found a class in format
            label = file[str_start:str_end]
            if label not in classification_list:
                classification_list.append(label)
                populations[label] = 1
            else:
                populations[label] += 1

    # Now we have a list of classifications, and population of each.
    # print that for now. (Allow print to be turned off...))
    n_found = len(classification_list)
    if n_found == expect:
        # Perfect! we found the classifications expected.
        if print_ok:
            print("Success! Found {} Classifications:".format(expect))
            for label in classification_list:
                print("\t{} Files of Label : {}".format(
                    populations[label], label))

    elif n_found <= expect:
        # What if we found FEWER than expected?
        print(classification_list)
        raise AssertionError("Found {} Labels, Expected at least {}...".format(
            n_found, expect))

    elif n_found >= expect:
        # Important! What if we found MORE than expected?
        # Well... Either error out, or move on with the most populous N!
        print("Found {} Labels, Only Expected {}...".format(n_found, expect))

        n_to_pass = int(len(csv_files)/expect * 0.9)
        # To pass, must have at least 1/nth of the total files
        # (With a 10% margin for error?)
        new_classification_list = []
        for label in classification_list:
            print("\t{} Files of Label : {}".format(
                populations[label], label))
            if populations[label] >= n_to_pass:
                new_classification_list.append(label)
            else:
                pass

        if len(new_classification_list) == expect:
            # Now that we've downsorted, I think there MUST be equal or less.
            #   Less may happen if data is a 75%/25% split for instance...
            # For now, throw our hands up in the air, and only say OK if
            #   we successfully fixed the situation.
            # Maybe that 10% buffer should be bigger or smaller (~line 333)
            classification_list = new_classification_list
            # Over-write the old list with the new.
        elif expect == 2:
            # IF expect is 2, use the most populous and all others are just
            # "NOT" that one...
            maxpop = 0
            mainlabel = ''
            for label in classification_list:
                if populations[label] > maxpop:
                    # If this is the highest population,
                    # Set this to be the biggest label
                    maxpop = populations[label]
                    mainlabel = label
                else:
                    pass
            classification_list = [mainlabel, "not_{}".format(mainlabel)]

    return classification_list


def read_csv(full_fname, skiprows=0, last_skiprows=None,
             maxskip=100):
    """
    Function to loop through pandas read_csv, checking the data
        and trying again if it's bad.
    Note:
        Will Return ONLY columns which are interger or floats.
            No lists, no strings, no silly things.

    Parameters
    ----------
    full_fname: str
        joined path and file name so that we can load the file

    try_skiprows: int
        this replaces the hard "skiprows" in the old functions.
        It'll be the first we try.

    last_skiprows: int (optional)
        Function Output of the successful skiprows #.
        To be re-fed into the function on the next loop
        occurance to speed up.
    max_skip: int
               loop size. Will error if you skip this many rows.

    Returns
    -------

    fdata : Pandas DataFrame
            The dataframe obtained from teh csv file
    last_skiprows : int
            the value fo the last row skipped
    """
    fdata = pd.read_csv(full_fname, skiprows=skiprows)
    # ^ We will use this to Track whether we did a successful load.
    #       Turn it TRUE if a load does not error, but Turn if FALSE
    #       if a successful load does not pass the data tests.
    # try:
    #     fdata = pd.read_csv(full_fname, skiprows=try_skiprows)
    #     load_success = _test_df(fdata)
    #     last_skiprows = try_skiprows
    # except pd.errors.ParserError:  # Error if file changes width
    #     pass
    #
    # # Second Try: "Last successful", if given.
    # if last_skiprows and not load_success:
    #     try:
    #         fdata = pd.read_csv(full_fname, skiprows=last_skiprows)
    #         load_success = _test_df(fdata)
    #     except pd.errors.ParserError:
    #         pass
    # else:
    #     pass
    #
    # # Finally, loop from n = 0 to maxrows until something passes!
    # n = 0
    # while not load_success and n <= maxskip:
    #     try:
    #         fdata = pd.read_csv(full_fname, skiprows=n)
    #         load_success = _test_df(fdata)
    #         last_skiprows = n
    #     except pd.errors.ParserError:
    #         pass
    #     n += 1  # Increment skiprows every time we fail.
    last_skiprows = 0
    assert len(list(fdata)) != 0, 'the csv was not correctly loaded'
    if 'Unnamed: 0' in list(fdata):
        fdata.pop('Unnamed: 0')
    return fdata, last_skiprows


def pickled_data_loader(raw_datapath, run_name):
    '''
    Loads the pickled data

    Parameters
    ----------
    raw_datapath: str
                  location of raw data
    run_name: str
              transformation name from the dictionary

    Output
    ------
    data_tuples: tuple
                 list of tuples containing file name
                 transformed data, label
    '''

    pickled_file_name = raw_datapath + run_name + '.pkl'
    imported_data = open(pickled_file_name, 'rb')
    image_data = pickle.load(imported_data)
    imported_data.close()
    os.remove(pickled_file_name)

    return image_data


def _test_df(fdata, columns_to_pass=2):
    """
    Parameters
    ----------
    fdata : dataframe
        Dataframe loaded by pd.read_csv()
    columns_to_pass: int
        number of columns containing the raw data.

    Returns
    -------
    load_success : bool
        If true, it means the data was correctly loaded
    """
    assert type(fdata) is pd.DataFrame, "Not Dataframe"

    if isinstance(fdata.iloc[0][0], str):
        return False
    else:
        column_types = fdata.dtypes
        good_columns = 0
        for dtype in column_types:
            if dtype == 'float':
                good_columns += 1
            elif dtype == 'float64' or 'float32':
                good_columns += 1
            elif dtype == 'int':
                good_columns += 1
            else:
                pass

        if good_columns >= columns_to_pass:
            return True  # GOOD Test!
        else:
            return False  # BAD Test! Don't error, just continue!
