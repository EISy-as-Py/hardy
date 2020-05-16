# import numpy as np
import pandas as pd
import pickle
import os
from hardy.handling.visualization import (rgb_plot, orthogonal_images_add)


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
             plot_type='single', skiprows=6, column_names=None):
    '''

    '''
    list_of_tuples = []
    for entry in os.listdir(input_path):
        if entry.endswith('.csv'):
            fdata = pd.read_csv(input_path+entry, skiprows=skiprows)
            if plot_type == 'single':
                rgb_image = rgb_plot(red_array=fdata[column_names[0]],
                                     blue_array=fdata[column_names[1]],
                                     plot=False)
            else:
                rgb_image_x = rgb_plot(red_array=fdata[column_names[0]],
                                       plot=False)
                rgb_image_y = rgb_plot(blue_array=fdata[column_names[1]],
                                       plot=False)
                rgb_image = orthogonal_images_add(rgb_image_x, rgb_image_y,
                                                  plot=False)
#  The labelling of the data is somewhat hardcoded in this funciton right now.
#  Consider improving it
            if classes[0] in entry:
                label = classes[0]
            else:
                label = classes[1]
            list_of_tuples.append((entry.rstrip(entry[-4:]),
                                   rgb_image, label))

    return list_of_tuples
