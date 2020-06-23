# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:04:32 2020
@author: hurtd

WRAPPING PACKAGE to Combine "Hanlding" and "Transformations" to produce
Data To Plot on Demand and deliver to the Image Processing f(n)s


 This package will contain several 'sections':

  * __Perform Transformation__: Wrapper function(s) to actually /do/ the
          transformations. This MAY be smart-ish and perform only specified
          ones, perhaps from a CONFIG or LIST? (That List and Config
          is an idea we'll talk about in __yNot__ probably!)

  * __Association Functions:__ This is optional, but once you've done
          the transformations (or before?) you can check how well the data
          CORRELATES/ASSOCIATES, both to itself and to a "Standard" dataset
          AKA starting with linear data and transforming it!
          This may give us "SCORES" for each transformation,
          which we can use to prioritize!


 __Timeline + Milestones__:
  * 2020-04-21: List of the high-priority functions and
                  Simple-Transformations, with progress and
                  timeline to get them all done soon.
  * 2020-04-28: Passing Tests and can __HAND OFF__ to the classifier - a
                  DataFrame of "all" the transformed data columns.
                  Recieve Handoff from handling, and begin to Integrate.
  * 2020-05-12: Complex Transforms - consider what other things we may want,
                  and discuss feedback with group
  * 2020-06-09: Make Decision on Association functions and __HAND OFF__
                  if so. Otherwise, simply focus on new group priorities
  * 2020-06-23: IF yNot function is doing Configuration ideas, make
                  Stretch-Goal learning gameplan... TBD...

 __Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work. Configuration?


 __Module List__:

### *SECTION: Perform Transformation*
#### get_xy():
  * Uses handling.py to load a file, check that we have xy data,
          and do a quick analysis on it!
  * INPUT: file name
  * OUTPUT: each of 1D arrays X and Y, plus messages OR list of
          "Approved" transformations??

#### perform_transform():
  * Wrapping function, to take some input data and return a
          dataframe with every transform that we want to use:
  * INPUT: 1D arrays X, Y, some sort of list or guiance for what
          transforms to do
  * OUTPUT: Pandas dataframe of X and Y with all of their transforms
          as requested.
  * note: There might be a better way to do this -

#### generate_linear_transforms():
  * Creates a "sample" linear 2D dataset, possibly following
          range instructions, and performs all(?) transforms on the data
  * INPUT: [ALL OPTIONAL? Default 0-1 and X=Y], [List of transforms to do??
          default True to perform "ALL"]
  * OUTPUT: pandas dataframe with all X-transformationa and all
          Y-transformations - (Maybe in Standard names? Maybe Not?)
  * note: This could get messy - And maybe use the same wrapping function
          above to perform the listed tranforms?

### *SECTION: Association Functions*
  * note: Not really planning these yet, that will be scoped or
          descoped based on Team Update by __2020-04-28__

#### setup_correlation_matrix():
  * Sets up the 2D matrix of "scores" to judge the correlations

#### correlate_to_transforms():
  * For a given transform, determine (?) how correlated the
          data is to all other columns in the dataframe

#### correlate_to_linear():
  * Compares the given transform to the linear_transform()
          function transforms. Any that correlate are probably good ideas?!?

#### correlate_to_null():
  * Maybe compares the given transform to a flat line of low but
          nonzero values (all value = 0.1)?
  * Not sure what's the best way to do this...
  * What I'm TRYING to do is to identify/flag "BORING" data, which are
          probably BAD transforms to use?

 #### grade_all_transforms():
  * Wrapping function. Given a "fully" transformed dataset
          (or generate it here?), run all correlations and use some
          fancy math or grading (we generate?) to give each column a "SCORE"
  * The __SCORE__ should reflect how "Interesting" we think the data
          is (which is a topic for discussion, but all zeros is
          not interesting)
  * INPUT: dataframe ready to be "graded", OR give an XY dataset and
          we'll call the transform functions on it
  * OUTPUT: dictionary of Key,Values where each Key is a transform
          (or column key), and each Value is a "grade" to estimate
          how interesting we think the data may be
  * note: This is SUPER arbitrary and is the 'creative' part of the
          STRETCH-GOALS of the project.s!

 #### grade_all_files():
  * Load all the files in a list, perform transformations, and grade.
          Hopefully this is a fast function so you can do a large
          list of files.
  * Then combine all the grades to get an average idea of what
          transforms we consider "good"
  * INPUT: list of files
  * OUTPUT: dictionary of key, values as before, where results
          are averaged across dataset
  * ACTIONS: Optionally, save results as a report csv (or append to
          existing csv report??), to track grades over time

 #### load_transform_results():
  * IF we've run this program before, we should have a file that has
          previous "grades", this time based on the model training
  * If one transform shows up in a lot of the best-performing models,
          we should bump it to the top of the Transform To-Do List!

 #### combine_grades_results():
  * Somehow we should combine the 'grades' with the 'prior results'
          to get our new guess for what are the 'best' transformations to try
  * this will allow our model to do the highest-profile transfomations
          first and hopefully get good results in fewer attempts.

"""
# import os.path
import copy  # Used for Deepcopy, to never edit the raw data frame
# import time
import yaml

import numpy as np
import pandas as pd

# Imports must be changed depending on Package vs GIT build....
import hardy.arbitrage.transformations as transform
# import hardy.arbitrage.transformations as transform


"""
note :  Almost All of these Functions are partly Obsolete because of the
        to_config.py change in default data methods. So instead I'm updating
        to conform to the new standards. Some logic can be re-used.

        Most importantly, we are working with data in the "List_of_Tuples"
        Format, where each data item is passed as a list of the following:
            (ID_String, DataFrame, Classify_string(or int?))
        Later this data is transformed into image list_of_rgb:
            (ID_String, RGB-3D-Array, Classify_string(or int?)
        These ARBITRAGE functions are designed to "intercept" the DataFrame
            List_of_Tuples and return an EQUALLY FORMATTED one where each
            df is replaced with the one created by the transform list passed.

        WE ARE NO LONGER READING FILES IN THIS PACKAGE.
        This will ALL be imported BY the preprocessing.py package, which
        will contain all loops to Read, Arbitrate, and Create CNN-readable
        Data list.
"""

tform_1d1d = transform.list_1d1d  # Import dict of Transform functions
tform_keys = list(tform_1d1d.keys())  # This is the list of f(n) keys


def import_tform_config(tform_config_path='.\tform_config.yaml', raw_df=None):
    """Function that imports the transformations from configuration

    Parameters
    ----------
    tform_config_path : Str, optional
                        Path of transform configuration file to
                        DESCRIPTION. The default is '.\tform_config.yaml'.

    CHECKS
    ------
    All Index are in 0-5 (for RGBrgb-style plotting)
    All transform commands are in transform.list_1d1d
    All source # are below the length of the raw-data list

    Returns
    -------
    tform_command_list : list of str
                         Ordered list of transform commands to use.
                         Differs from the dict.keys() because this is ordered!
                         (May save Report with this string as the key,
                         to be looked up?)

    tform_command_dict :  dict of List-of-Transform-tuples
                          Each key will return a list of transforms
                          to do on this data loop.
                          Each "List of Transforms" as stated elsewhere
                          contain:
                          (Index=0, transform, source),
                          (Index=1, transform, source),
                          (Index=2, transform, source),
                          etc. where:
                          "Index" is the output column destination,
                          "transform" is command in transform.list_1d1d, and
                          "source" is the raw data column to be used in the
                          tform
    """
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
            assert each_tform[1] in tform_keys, \
                "Transform '{}' Not Available ".format(each_tform[1]) +\
                "from Tform_1d1d."
            assert each_tform[2] >= 0 and each_tform[2] < df_cols, \
                "Source Column given in '{}' out of Range".format(command) +\
                " Of Raw DataFrame."

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
        new_names[command[0]] = old_names[command[2]] + '__tform__' +\
            command[1]

    # Now initialize output df with zeros from length of first df column
    df_len = len(raw_df[old_names[0]])
    zero_arr = np.zeros([df_len, rgb_col_number])
    tform_df = pd.DataFrame(data=zero_arr, columns=new_names)

    # And now, apply each transform and assign the output to the
    #   Column as instructed in that command
    for command in tform_commands:
        target_raw = raw_df[old_names[command[2]]]
        # Get raw data (series?) from source
        tform_data = tform_1d1d[command[1]](target_raw)  # Perform the tform
        tform_df[new_names[command[0]]] = tform_data  # Save in output df

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


#


# =============================================================================
# """
# TESTING ZONE
# """
#
# a=time.perf_counter()
# test_dir = '../local_data/2020-4-21_0000'
# test_tform_list = (
#      (0, "1d_exp", 0),
#      (1, "1d_none", 1),
#      (2, "1d_cumsum", 1),
#      )
#
# transformed_data, save_path = setup_tform_files(test_dir)
# test_tform_data = load_and_transform_data(transformed_data, test_tform_list,
#                                           save_path=save_path)
#
# print("Time was : {} seconds".format(round(time.perf_counter()-a,2)))
# """
# To-Do now:
#    -make some sort of iterable to scan the data and determine GLOBAL features
#    (like max and min, to deterimine what transforms are OK for each dataset)
#    -Use this function to create a LIST of Transform_list possibilities, and
#    iterate over that list (as we will be able to direct)
#
# """
# =============================================================================
