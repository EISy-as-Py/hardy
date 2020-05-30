import numpy as np
import os
# import pandas as pd
import pickle
import time

# import hardy.handling.visualization as vis
# import hardy.handling as handling
import visualization as vis
import handling


def save_load_data(filename, data=None, save=None, load=None,
                   file_extension='.sav', location='./'):
    """Function to save and load model

    Function that can save or load model depending on given parameters.

    Parameters
    ----------
    filename : str
               string indicating the filename for saving or loading dataset.
    data : list
           dataset that is to be saved or loaded.
    save : bool
           boolean value if true saves the compressed dataset.
    load : bool
           boolean value if true loads the compressed dataset.
    file_extension : str
                     String containing the file extension to use
    location :  str
                string containing the path to the folder to save the
                pickled file in

    Returns
    -------
    loaded_data : list
                  dataset that is loaded from the specified location
    """
    if save:
        pickle.dump(data, open(location + filename + file_extension, 'wb'))
        return print('Successfully Pickled')
    elif load:
        loaded_data = pickle.load(open(location + filename + file_extension,
                                  'rb'))
        return loaded_data


def _data_tuples_from_fnames(input_path='./', skiprows=6, classes=None):
    """
    Setting up the Data_tuples list, from ONE FOLDER with all of the data
        (OF different classes) inside of it.
    For each file, do a "smart-load" of the data, remove bad columns,
        and determine the classification from the file name.
    Then Return that line of data_tuples in the format of:
        (SERIAL, DataFrame, LABEL)
    """
    # Get list of classes for later
    list_of_tuples = []
    if classes is None:
        # This tells us to find the categories on our own.
        #    See "Handling" package for these methods.
        classes = handling.cats_from_fnames(os.listdir(input_path))
    else:
        pass

    # Now loop through each item in list, load the dataframe, and append
    # the SERIAL, Dataframe (fixed), and Class to the list of tuples!
    fread_timer = time.perf_counter()
    n_total = len(os.listdir(input_path))
    n_trigger = int(n_total/10)
    n_counter = 0
    n_reset = 0

    last_skiprows = None
    for entry in os.listdir(input_path):
        n_counter += 1
        n_reset += 1
        if n_reset >= n_trigger:
            fread_rate = int(n_trigger / (time.perf_counter() - fread_timer))
            # Rate in Files per Second.
            print('\rLoaded\t{} of {}\tFiles'.format(n_counter, n_total) +
                  '\t at rate of {} Files per Second'.format(fread_rate),
                  end='')
            fread_timer = time.perf_counter()
            n_reset = 0
        else:
            pass

        if entry.endswith('.csv'):
            # Read data into pandas dataframe
            fdata, last_skiprows = \
                handling._smart_read_csv(input_path+entry,
                                         try_skiprows=skiprows,
                                         last_skiprows=last_skiprows)
            # Now remove any columns with bad data types
            # (Strings, objects, etc)
            for column in fdata.columns:
                if fdata[column].dtypes is float:
                    pass
                elif fdata[column].dtypes is np.dtype('float64'):
                    pass
                elif fdata[column].dtypes is int:
                    pass
                else:
                    # If type is not int, float, or numpy special float...
                    # It's either string, object, or something else bad..
                    fdata = fdata.drop(columns=column)

            label = None
            for each_label in classes:
                # Find the first label that matches.
                if not label and each_label in entry:
                    label = each_label
                else:
                    pass
            if not label:
                # If none of the labels fit, make new "not" first label
                label = "not_" + classes[0]
            else:
                pass

            list_of_tuples.append((entry.rstrip(entry[-4:]),
                                  fdata, label))
        else:
            # If File is not csv, ignore
            pass
    t_mins = round(n_total/fread_rate/60,2)
    print("\n\t Success!\t About {} Minutes...".format(t_mins))
    # (Because timer has no Newline Character!)
    return list_of_tuples


def rgb_list(data_tuples, plot_format='RgBrGb', column_names=None,
             combine_method='add'):
    '''
        Input a path of csv files (with some guidance),
        Plot them RGB-wise into images
        return a list of tuples as to be fed into the keras PreProcess f(n)

        INPUTS:
            data_tuples :   list of tuples
                            following the convention:
                            (SERIAL, DataFrame, LABEL)
                            (see below...)
            plot_format :   string
                            to pass into rgb_visualize
                                "single", "else", or some "RGBrgb"...
                                DEFAULT: "RgBrGb"? Discuss with group!!!
            combine_method :string
                            to pass into rgb_visualize

            column names :  list of strings (Optional)
                            IF given, will drop all columns not in the
                                list given. (If no colums match, will ERROR.)
        RETURNS
            list_of_rgb_tuples  :   list of tuples
                                    following the format: (SERIAL, IMG, LABEL)
            SERIAL      :   File name with the extension taken off
                                (We should parse with . not just [-4])...
            IMG         :   ndarray of NxNx3

            LABEL       :   Classification label, either from the passed list
                                or from the last part of the serial/filename:
                                "123847_afsukjeh_*LABEL*.csv""

    '''

    print("Making rgb Images from Data...", end='\t')
    t = time.perf_counter()
    list_of_rgb_tuples = []
    for data_tuple in data_tuples:
        # For each dataframe given
        fdata = data_tuple[1]

        rgb_image = rgb_visualize(fdata, plot_format, combine_method,
                                  column_names)
        # Need some check that the visualization worked?

        rgb_tuple = (data_tuple[0], rgb_image, data_tuple[2])
        list_of_rgb_tuples.append(rgb_tuple)

    t_sec = round(time.perf_counter()-t, 2)
    print("Success in {}seconds!".format(t_sec))
    return list_of_rgb_tuples


def rgb_visualize(fdata, plot_format='RGBrgb', combine_method='add',
                  column_names=None):
    '''
        Input a list of dataframes (already read and/or processed),
        Plot them RGB-wise into images
        return a list of tuples as to be fed into the keras PreProcess f(n)

        INPUTS:
            plot_format :   EITHER 'single' (bodge, depreciate later)
                            OR some combination of "RGBrgb", which will be
                            the order of columns plotted:
                            R = red   X-axis      r = red   Y-axis
                            G = green X-axis      g = green Y-axis
                            B = blue  X-axis      b = blue   Y-axis
                             * X = do not plot (skip column)
                             ** If RGBrgb letters are missing, simply pass
                                to the plotting function as "None"

                            The to-be-depreciated 'single' is thus:
                                "RB"
                            The As-written "else" is thus:
                                "Rb"

            combine_method: "add" or "mlt" - which visualization fn to use

        RETURNS

        list_of_tuples  :   list of tuples, following: (SERIAL, IMG, LABEL)

            SERIAL      :   File name with the extension taken off
                                (We should parse with . not just [-4])

            IMG         :   ndarray of NxNx3

    '''
    if not column_names:
        column_names = list(fdata.columns)

    if plot_format == 'single':
        rgb_image = vis.rgb_plot(red_array=fdata[column_names[0]],
                                 blue_array=fdata[column_names[1]],
                                 plot=False)
    elif plot_format == "else":
        rgb_image_x = vis.rgb_plot(red_array=fdata[column_names[0]],
                                   plot=False)
        rgb_image_y = vis.rgb_plot(blue_array=fdata[column_names[1]],
                                   plot=False)
        rgb_image = vis.orthogonal_images_add(rgb_image_x, rgb_image_y,
                                              plot=False)
    else:
        # Writing new Decision-matrix to organize with the input-string
        # Loop through the string, and if you see an "RGB,rgb",
        #   then that column is the one which will go there!
        R = None
        G = None
        B = None
        r = None
        g = None
        b = None
        for i in range(len(plot_format)):
            # Loop through the string. react to FIRST encounter of str
            if R is None and plot_format[i] == "R":
                R = fdata[column_names[i]]
            if G is None and plot_format[i] == "G":
                G = fdata[column_names[i]]
            if B is None and plot_format[i] == "B":
                B = fdata[column_names[i]]
            if r is None and plot_format[i] == "r":
                r = fdata[column_names[i]]
            if g is None and plot_format[i] == "g":
                g = fdata[column_names[i]]
            if b is None and plot_format[i] == "b":
                b = fdata[column_names[i]]
        rgb_image_x = vis.rgb_plot(red_array=R, green_array=G,
                                   blue_array=B, plot=False)
        rgb_image_y = vis.rgb_plot(red_array=r, green_array=g,
                                   blue_array=b, plot=False)

        # Default to "Add", but check for the option of using the mlt fn.
        if combine_method == "mlt":
            rgb_image = vis.orthogonal_images_mlt(rgb_image_x,
                                                  rgb_image_y,
                                                  plot=False)
        else:
            rgb_image = vis.orthogonal_images_add(rgb_image_x,
                                                  rgb_image_y,
                                                  plot=False)
    return rgb_image


def rgb_list_to_DirFlow(rgb_tuples, basepath, newfolder="rbg_for_keras",
                        delete_existing=True):
    """
    Takes the list of tuples (as made in "rgb_list") and creates the exact
        file structure of saved PNG images that will be used in the
        "KERAS FLOW FROM DIRECTORY" method.

    Also will save a log file in the base path (csv? or look for log?)
        describing the
    """
    classes = []
    for each_image in rgb_tuples:
        if each_image[2] not in classes:
            # Make a unique list of the classes...
            classes.append(each_image[2])
        else:
            pass

    newfolder_path = os.path.join(basepath, newfolder)
    # Make new folder full path... But what if it exists?
    if os.path.isdir(newfolder_path):
        # Well, if "Delete Existing" is true, delete that folder.
        if delete_existing:
            # Loop through and delete all... DANGEROUS
            # So instead wrote a safety loop.
            _safe_clear_dirflow(newfolder_path)
        else:
            raise AssertionError("Directory Full! Pass new 'newfolder'\n" +
                                 "\t Or use 'delete_existing'=True")
    else:
        pass

    # Now, make folders and fill them with the images!
    os.makedirs(newfolder_path)
    for each_class in classes:
        class_folder = os.path.join(newfolder_path, each_class)
        os.makedirs(class_folder)
        for each_image in rgb_tuples:
            if each_image[2] == each_class:
                # If this image is in the class, save it in this folder!
                save_png = os.path.join(class_folder, each_image + '.png')



    return basepath, newfolder


def _safe_clear_dirflow(path):
    """
    Safely check that the path contains ONLY folders of png files.
        if any other structure, will simply ERROR out.
    (for now, doesn't fix errors, just raises them)
    """
    print("Clearing {}...".format(path))
    assert os.isdir(path), "Didn't pass a folder to be cleaned"
    for folder in os.listdir(path):
        cat_folder = os.path.join(path, folder)
        assert os.path(os.isdir(cat_folder)), \
            "Dir contains Non-Folder File!"
        for file in os.listdir(cat_folder):
            # For every file, confirm is PNG or error.
            # DONT DELETE YET, IN CASE OF ERRORS!
            assert ".png" in file, "Folder has Non PNG Contents!"
    # If we got though that with no error, then now we can delete!
    for folder in os.listdir(path):
        cat_folder = os.path.join(path, folder)
        for file in os.listdir(cat_folder):
            os.remove(os.path.join(cat_folder, file))
        os.rmdir(cat_folder)
    os.rmdir(path)
    return True



# Testing Zone:
Path_0 = "C:/Users/hurtd/Py/hardy/hardy/local_data/"
EIS_fname_data = Path_0 + "200504_csv_EIS_simulaiton/"
Simple_dir_data = Path_0 + "2020-4-24_0001/"

EIS_data_tuples = _data_tuples_from_fnames(EIS_fname_data)
EIS_rgb_tuples = rgb_list(EIS_data_tuples)
EIS_folder_to_keras = rgb_list_to_DirFlow(EIS_rgb_tuples,
                                          basepath = EIS_fname_data,
                                          )