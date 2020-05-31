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
  * NOTE: There might be a better way to do this -

#### generate_linear_transforms():
  * Creates a "sample" linear 2D dataset, possibly following
          range instructions, and performs all(?) transforms on the data
  * INPUT: [ALL OPTIONAL? Default 0-1 and X=Y], [List of transforms to do??
          default True to perform "ALL"]
  * OUTPUT: pandas dataframe with all X-transformationa and all
          Y-transformations - (Maybe in Standard names? Maybe Not?)
  * NOTE: This could get messy - And maybe use the same wrapping function
          above to perform the listed tranforms?

### *SECTION: Association Functions*
  * NOTE: Not really planning these yet, that will be scoped or
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
  * NOTE: This is SUPER arbitrary and is the 'creative' part of the
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
import os.path
import copy  # Used for Deepcopy, to never edit the raw data frame
import time

import numpy as np
import pandas as pd

# import transformations as transform
import hardy.arbitrage.transformations as transform
tform_1d1d = transform.list_1d1d

# This is a test of the imported transforms functions,
#    so I can make sure it's working
a = np.linspace(0.1, 10, 1000)
b = transform.transform_1d_cumsum(a)


def setup_tform_files(input_path='../local_data'):
    """
    Takes in a directory

    Takes in a list of transforms,
    """
    classes = []
    for item in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, item)):
            # If it's a folder, check how many files are inside
            if len(os.listdir(os.path.join(input_path, item))) > 10:
                # If at least n files in the folder, it may be a classifier
                classes.append(item)
    assert len(classes) >= 2, "Must have found at least 2 classes to run a CNN"

    transformed_data = {}
    for each_class in classes:
        n_files = 0
        size_files = 0
        raw_files = []
        transform_files = []
        for fname in os.listdir(os.path.join(input_path, each_class)):
            if '.csv' in fname:
                # Make a list of all the csvs in that folder
                fullname = os.path.join(input_path, each_class, fname)
                raw_files.append(fullname)
                savename = os.path.join(input_path, 'transform',
                                        each_class, fname)
                transform_files.append(savename)
                n_files += 1
                size_files += os.path.getsize(fullname)
            else:
                pass

        # Now, for each classification, make a dataframe that will (for now)
        #   only have the filenames in it. Then we can loop through and process
        #   files one-by-one
        transformed_data[each_class] = pd.DataFrame(data=raw_files,
                                                    columns=['raw_path'])
        transformed_data[each_class]['tform_path'] = transform_files

        # save_path will be the equivalent masterfolder which will have the
        # two classification subfolders in it, JUST like the RAW Data Folders
        save_path = os.path.join(input_path, 'transform')
        os.makedirs(os.path.join(save_path, each_class), exist_ok=True)

        print("{} Classifier has {} files, taking {} MB of data".format(
                each_class, n_files, size_files/1000000))

    """
    Now, to load all the data into that pd dataframe
    """
    for each_class in classes:
        fread_data = []
        i_interval = int(len(transformed_data[each_class])/10)
        i_target = copy.deepcopy(i_interval)
        print("\n{}:".format(each_class))
        timer = time.perf_counter()
        for i, file_row in transformed_data[each_class].iterrows():
            if i >= i_target:
                dt = time.perf_counter() - timer  # Total time in Seconds
                timer = time.perf_counter()
                rate = int(i_interval / dt)  # Files per Second
                print("\r{}\tout of".format(i) +
                      "\t{}\t files...".format(len(transformed_data[
                                               each_class])) +
                      "\t{}\tFiles/second".format(rate),
                      end="")  # THIS IS A FLAKE8 DISASTER
                i_target += i_interval
            n = 0
            while n < 100:
                try:
                    # fread = pd.read_csv(file_row['raw_path'])
                    fread = pd.read_csv(file_row['raw_path'], skiprows=n)
                    if np.float not in fread.dtypes.values:
                        # This should not happen!
                        n += 1
                    else:
                        # for column in fread.columns:
                        #    if not np.float in fread[column].dtypes.values:
                        #        fread.drop(column)
                        #    else:
                        #        pass
                        fread_data.append(fread)
                        # print(n)
                        n = 1000
                except pd.errors.ParserError:
                    n += 1
                if n > 75 and n <= 100:
                    raise AssertionError("Failed to Import: {}".format(
                        file_row['raw_path']))
        transformed_data[each_class]['raw_data'] = fread_data
    return transformed_data, save_path


def load_and_transform_data(transformed_data, tform_list=None,
                            save_path=False):

    if tform_list:
        """
        To Perform Transformations, we need to be given a tuple/list of
            commands. This will appear in the following method:
        (
        (Index=0, transform, source),
        (Index=1, transform, source),
        (Index=2, transform, source),
        (Index=3, transform, source),
        (Index=4, transform, source),
        (Index=5, transform, source),
        )

        Wherein Index will be a number from 0 to 5 representing RGB in X and Y,
            respecitively.
        Each "transform" will be a function in one of the 1D lists (for now)
            which should be CALLABLE via the tform_1d1d function.
        each "source" will be the naturally occuring array number
            (OR COLUMN NAME?) that will be transformed upon.

        tform_data=[]
        tform_data[Index] = tform_1d1d[transform](source)

        """
        # Setup Column Names for the Transform:
        tform_columns = []
        n_cols = len(tform_list)
        for tform in tform_list:
            col_name = "{}_on_col_{}".format(tform[1], tform[2])
            tform_columns.append(col_name)

        for each_class in transformed_data:
            # Loop through both classifications
            # Option: Redefine names here?
            i_interval = int(len(transformed_data[each_class])/10)
            i_target = copy.deepcopy(i_interval)
            print("\n{}:".format(each_class))
            all_transforms = []  # Initialize storage dataframe(list)
            timer = time.perf_counter()
            for i, file_row in transformed_data[each_class].iterrows():
                if i >= i_target:
                    dt = time.perf_counter() - timer  # Total time in Seconds
                    timer = time.perf_counter()
                    rate = int(i_interval / dt)  # Files per Second
                    print("\r{}\tout of\t{}\t files...".format(i,
                          len(transformed_data[each_class]),) +
                          "\t{}\tFiles/second".format(rate), end="")
                    i_target += i_interval

                # Loop through each file's raw data
                raw_data = copy.deepcopy(file_row['raw_data'])
                tform_data = np.ones([len(raw_data), n_cols])
                # Should we initialize as n_cols, or  as always 6??
                for tform in tform_list:
                    col_source = raw_data.columns[tform[2]]
                    pass_array = raw_data[col_source]
                    # print(pass_array)  # Debugging
                    tform_data[:, tform[0]] = tform_1d1d[tform[1]](pass_array)
                tform_df = pd.DataFrame(tform_data, columns=tform_columns)
                all_transforms.append(tform_df)
            transformed_data[each_class]['tform_data'] = all_transforms

    """
    Now, the Save_Path part of the module. IF you gave us a save path
    (NOTE: save_path can be copied from the setup function)
    then we will loop through all of the files, and save them as their newly
    transformed pandas CSV files
    """
    if save_path:
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        for each_class in transformed_data:
            folder = os.path.join(save_path, each_class)
            if os.path.exists(folder):
                pass
            else:
                os.makedirs(folder)

            for i, file_row in transformed_data[each_class].iterrows():
                tform_data = file_row["tform_data"]
                tform_data.to_csv(file_row['tform_path'], index=False)

    return transformed_data


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
