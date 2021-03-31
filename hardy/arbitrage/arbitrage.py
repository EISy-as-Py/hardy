import copy  # Used for Deepcopy, to never edit the raw data frame
import yaml

import numpy as np
import pandas as pd

import hardy.arbitrage.transformations as transform


def import_tform_config(tform_config_path='./tform_config.yaml', raw_df=None):
    """Function that imports the transformations from configuration

    Parameters
    ----------
    tform_config_path : Str, optional
        Path of transform configuration file to apply to the data.
    raw_df: pd.DataFrame
        Dataframe of raw data to use for assrting that the configuration
        file is correctly calling on the data


    Returns
    -------
    tform_command_list : list of str
         Ordered list of transform commands to use.
         Differs from the dict.keys() because this is ordered!

    tform_command_dict :  dict of List-of-Transform-tuples
         Each key will return a list of transforms to do on this data loop.
         Each "List of Transforms" as stated elsewhere contain:
              (Index=0, transform, source),
              (Index=1, transform, source),
              (Index=2, transform, source),
        where:
              "Index" is the output column destination,
              "transform" is command in transform.list_1d1d, and
              "source" is the raw data column to be used in the tform
    """

    # CHECKS
    # ------
    # All Index are in 0-5 (for RGBrgb-style plotting)
    # All transform commands are in transform.list_1d1d
    # All source # are below the length of the raw-data list

    if raw_df is not None:
        df_cols = len(raw_df.columns)
    else:
        df_cols = 100  # IF no Dataframe given, allow up to 100 columns!

    # Given an "Example" Dataframe (usually the first tuple in list?)
    # Get the number of columns available, to check that the commands
    #   are in the right range.

    with open(tform_config_path, 'r') as file:
        loading = yaml.load(file, Loader=yaml.FullLoader)
        tform_command_dict = loading['tform_command_dict']
        tform_command_list = loading['tform_command_list']
        file.close()

    # Checking type, format, and range of all commands
    assert type(tform_command_list) is list, "Config List must be List!"
    assert type(tform_command_dict) is dict, "Config Dict must be Dict?"

    for command in tform_command_list:
        assert type(tform_command_dict[command]) is list, \
            "Config Dict must only contian Lists!"
        for each_tform in tform_command_dict[command]:
            assert type(each_tform) is list, \
                "Command '{}' has Non-List object".format(command)
            assert len(each_tform) == 3, \
                "Command '{}' has Wrong List format.".format(command)
            assert each_tform[0] >= 0 and each_tform[0] < 6, \
                "INDEX for Tform '{}' Does Not fit".format(command) +\
                " in RGBrgb (must be 0 to 5)"
            assert getattr(transform, each_tform[1]), \
                "Transform '{}' Not Available ".format(each_tform[1]) +\
                "from Tform_1d1d."
            if type(each_tform[2]) == 'int':
                assert each_tform[2] >= 0 and each_tform[2] < df_cols, \
                    "Source Column given in '{}' out of Range" \
                    .format(command) + " Of Raw DataFrame."
            if type(each_tform[2]) == 'tuple':
                assert 4 >= len(each_tform[2]), 'too many arguments' +\
                    ' provided. Maximum of 4 allow at the moment.'
    print("Successfully Loaded {} Transforms to Try!".format(
        len(tform_command_list)))
    return tform_command_list, tform_command_dict


def apply_tform(raw_df, tform_commands, rgb_col_number=6):
    """ Function that applies transformations

    Parameters
    ----------
    raw_df : pd.DataFrame of raw data from list_of_tuples
             Un-Transformed data to apply transform to.
             This will be a call of list_of_tuples[#][1], because as
             defined elsewhere, each raw data has one tuple in list,
             contains (Filename, DataFrame, classifier)

    tform_commands : List of Tform Commands
                     This will be a call of tform_command_dict
                     [tform_command_list[#]],
                     Thus it will contain a list of tform commands:
                     (Index=0, transform, source),
                     (Index=1, transform, source),
                     (Index=2, transform, source),
                     As explained elsewhere

    Returns
    -------
    tform_df: pd.DataFrame
              Each column is placed in "Index", and gets its name from "source"
              (New name from SourceColumnName__tform__TformName)
              Each column's data is ouput of the tform_1d1d function called
              and the remainder are passed as zero (with # as col name?)

    """
    # First get new column names:
    old_names = list(raw_df.columns)
    new_names = list(range(rgb_col_number))
    for command in tform_commands:
        if type(command[2]) == 'int':
            new_names[command[0]] = old_names[command[2]] + '__tform__' +\
                command[1]
        elif type(command[2]) == 'tuple' and command[1] == 'power':
            if command[2][1] == 'None':
                new_names[command[0]] = old_names[command[2][0]] +\
                    '__tform__' + command[1]
            else:
                new_names[command[0]] = str(old_names[command[2][0]] + '*' +
                                            old_names[command[2][1]]) +\
                                          '__tform__' + command[1]

    # Now initialize output df with zeros from length of first df column
    df_len = len(raw_df[old_names[0]])
    zero_arr = np.zeros([df_len, rgb_col_number])
    tform_df = pd.DataFrame(data=zero_arr, columns=new_names)

    # And now, apply each transform and assign the output to the
    #   Column as instructed in that command
    for command in tform_commands:
        if type(command[2]) == 'int':
            target_raw = raw_df[old_names[command[2]]]
            # Get raw data (series?) from source
            # Perform the tform
            transform_function = getattr(transform, command[1])
            tform_data = transform_function(target_raw)
            # Save in output df
            tform_df[new_names[command[0]]] = tform_data
        if type(command[2]) == 'list':
            # if command[1] == 'power':
            # the power trasnformation is in the form of x^(n)y^(m).
            # the arguments should be inputted as (x, y, n, m)
            data_series_1 = raw_df[old_names[command[2][0]]]
            data_series_2 = raw_df[old_names[command[2][1]]]
            if len(command[2]) == 2:
                meta_data = None
            else:
                meta_data = command[2][2:]
            transform_function = getattr(transform, command[1])
            tform_data = transform_function(
                    data_series_1, data_series_2, meta_data)
    return tform_df


def tform_tuples(list_of_tuples, tform_commands, rgb_format="RGBrgb"):
    """
    Wrapping function to apply a list of transform commands to each
    dataframe in the list_of_tuples, and replace it with a same-format
    list_of_tuples containing transformed data.

    Parameters
    ----------
    list_of_tuples : List of Tuples
        Described in depth elsewhere. Standardized list for each raw file,
        tuple in format (filename_str, DataFrame, label)
    tform_commands : List of List(3)
        Described in depth elsewhere
    rgb_format : str, optional
        String of how we will parse the output files.
        Input here to get the output dataframe size.
        The default is "RGBrgb".

    Returns
    -------
    transformed_tuples : List of Tuples
        Formatted the same as the input list, but each DataFrame is
        replaced with the Transformed DF.

    """
    rgb_n = len(rgb_format)
    transformed_tuples = []
    for raw_data in list_of_tuples:
        fname = raw_data[0]
        raw_df = copy.deepcopy(raw_data[1])
        # ^DeepCopy will FORCE writing new data to avoid messing with RAW.
        label = raw_data[2]
        tform_df = apply_tform(raw_df, tform_commands, rgb_n)
        tform_tup = (fname, tform_df, label)
        transformed_tuples.append(tform_tup)

    return transformed_tuples
