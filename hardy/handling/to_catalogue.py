# import numpy as np
import pandas as pd
import pickle
import os

# from visualization import (rgb_plot, orthogonal_images_add)
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


def rgb_list(input_path='./', classes=['noisy', 'not_noisy'],
             plot_format='single', skiprows=6, column_names=None):
    '''
        Input a path of csv files (with some guidance),
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
                             ** If string skips letters, totally ok.

                            The to-be-depreciated 'single' is thus:
                                "RB"
                            The As-written "else" is thus:
                                "Rb"

        RETURNS

        list_of_tuples  :   list of tuples, following: (SERIAL, IMG, LABEL)

            SERIAL      :   File name with the extension taken off
                                (We should parse with . not just [-4])

            IMG         :   ndarray of NxNx3

    '''
    list_of_tuples = []
    if not classes:
        classes = handling.cats_from_fnames(os.listdir(input_path))

    for entry in os.listdir(input_path):
        if entry.endswith('.csv'):
            # Read data into pandas dataframe
            fdata = pd.read_csv(input_path+entry, skiprows=skiprows)

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
                    if not R and plot_format[i] == "R":
                        R = fdata[column_names[i]]
                    if not G and plot_format[i] == "G":
                        G = fdata[column_names[i]]
                    if not B and plot_format[i] == "B":
                        B = fdata[column_names[i]]
                    if not r and plot_format[i] == "r":
                        r = fdata[column_names[i]]
                    if not g and plot_format[i] == "g":
                        g = fdata[column_names[i]]
                    if not b and plot_format[i] == "b":
                        b = fdata[column_names[i]]
                rgb_image_x = vis.rgb_plot(red_array=R, green_array=G,
                                           blue_array=B, plot=False)
                rgb_image_y = vis.rgb_plot(red_array=r, green_array=g,
                                           blue_array=b, plot=False)
                rgb_image = vis.orthogonal_images_add(rgb_image_x,
                                                      rgb_image_y,
                                                      plot=False)

#  The labelling of the data is somewhat hardcoded in this funciton right now.
#  Consider improving it. We can now call cats_from_fnames for the full list.
#            if classes[0] in entry:
#                label = classes[0]
#            else:
#                label = classes[1]

            label = None
            for each_label in classes:
                # Find the first label that matches.
                if not label and each_label in entry:
                    label = each_label
                else:
                    pass
            if not label:
                # If none of the labels fit, make new "not" label
                label = "not_" + classes[0]
            else:
                pass
            list_of_tuples.append((entry.rstrip(entry[-4:]),
                                   rgb_image, label))

    return list_of_tuples


def rgb_list_from_df(serial_list, df_list, classes=None,
             plot_format='RGBrgb', column_names=None,
             combine_method='add'):
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
    list_of_tuples = []
    if not classes:
        classes = handling.cats_from_fnames(serial_list, from_serials=True)

    # MAIN DIFFERENCE BETWEEN THIS AND THE FROM_FILE:
    #   Already have a list of DFs and the corresponding list of serials,
    #   But we need to confirm they're correct
    assert len(serial_list) == len(df_list), "InputError: DF and " +\
        "Serial Lists Don't Match!"

    for f_number, fdata in enumerate(df_list):
        # Read data into pandas dataframe
        serial = serial_list[f_number]

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
                if not R and plot_format[i] == "R":
                    R = fdata[column_names[i]]
                if not G and plot_format[i] == "G":
                    G = fdata[column_names[i]]
                if not B and plot_format[i] == "B":
                    B = fdata[column_names[i]]
                if not r and plot_format[i] == "r":
                    r = fdata[column_names[i]]
                if not g and plot_format[i] == "g":
                    g = fdata[column_names[i]]
                if not b and plot_format[i] == "b":
                    b = fdata[column_names[i]]
            rgb_image_x = vis.rgb_plot(red_array=R, green_array=G,
                                       blue_array=B, plot=False)
            rgb_image_y = vis.rgb_plot(red_array=r, green_array=g,
                                       blue_array=b, plot=False)

            # Default to "Add", but check for the option of using the mlt fn.
            if combine_method is "mlt":
                rgb_image = vis.orthogonal_images_mlt(rgb_image_x,
                                                      rgb_image_y,
                                                      plot=False)
            else:
                rgb_image = vis.orthogonal_images_add(rgb_image_x,
                                                      rgb_image_y,
                                                      plot=False)

#  The labelling of the data is somewhat hardcoded in this funciton right now.
#  Consider improving it. We can now call cats_from_fnames for the full list.
#            if classes[0] in entry:
#                label = classes[0]
#            else:
#                label = classes[1]

        label = None
        for each_label in classes:
            # Find the first label that matches.
            if not label and each_label in serial:
                label = each_label
            else:
                pass
        if not label:
            # If none of the labels fit, make new "not" label
            label = "not_" + classes[0]
        else:
            pass
        list_of_tuples.append(serial, rgb_image, label)

    return list_of_tuples
