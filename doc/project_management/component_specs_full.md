# Component Specification:
List of MODULES, each containing Component Spec "Cards". 
If there's a clever way of formatting this, we should consider it!

Each "Module" contains:

__module.py__
 * Purpose and Description
 * Timeline + Milestones (?)
 * Current Status
 * LIST of CARDS / FUNCTIONS


Each "Card" contains:

__func_name():__
 * Purpose and general description
 * INPUT: description of imputs- types and sources
 * OUTPUT: description of returns
 * ACTIONS: Any non-obvious actions the function can take other than IN/OUT (example - option to saving files)
 * NOTES:
 -------------------------------------------------------------------------------
 
 ## handling.py
 The zeroth-level functions, related to importing and setting up data. The twin goals are to be fast, but also as broad and thoughtful about the data we may import as possible. 
 We may crib from work done in eisy/data_managment.py since we contributed to both project.
 __Note:__ Some of this may have Options to User-Interface, which is great but should DEFAULT be off, so we don't deal with it during Automated Processes!
 (Similarly, travis doesn't like User-Interfacing...)
 __Timeline + Milestones__:
  * 2020-04-21: Basic 2D, 3D importer that works with the simple files we're using 
  * 2020-04-28: Importer that works with each of the Raw Data types that we're using as part of our part-1 program. __HAND OFF!__
  * 2020-05-12: Debugging and Tests that cover some basic and logical use-cases (header, Too-Many-Columns, etc)
  * 2020-06-09: "Smart-ish" handling of the use cases mentioned. This should be the final __HAND OFF__
  
 __Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work.
  
 __Module List__:
 #### import_2d():
  * Basic file to import a csv that has two columns (with or without column names?) - Essentially pd.read_csv, but with error handling to work better in our program
  * INPUT: filepath/name, error-handling instructions (interact=False, return2=True)
  * OUTPUT: x_data, y_data  <-- Two (Tuples? Series? Discuss) 1D datasets 
  * ACTIONS: Checks that the file is actualy 2D data - if not, either asks for guidance, errors, or returns the first two columns depending on instructions
 
 #### import_3d():
  * Basic file to import a csv that has three columns (with or without column names?) - Essentially pd.read_csv, but with error handling to work better in our program
  * INPUT: filepath/name, error-handling instructions (interact=False, return3=True)
  * OUTPUT: x_data, y_data, z_data  <-- Three (Tuples? Series? Discuss) 1D datasets 
  * ACTIONS: Checks that the file is actualy 3D data - if not, either asks for guidance, errors, or returns the first two columns depending on instructions
  
 #### get_classification_files():
  * Our project will expect a specific though simple folder structure: main __"/train/"__ data folder will have (2? more?) classification folders with the classifier names as the folder names - inside will be all the raw data in (csv?) format. 
  * INPUT: Base folder containing classified data folders
  * OUTPUT: Dictionary, where keys are the classifications (folder names) and each have a (tuple/list) of file names to load
  * ACTIONS: Checks the file and data format. May try and load one/more files? 
  * NOTE: Should we estimate size of all files and eventually try and estimate program-time? that may be useful...
  
 
 ## arbitrage.py
 The first-level functions, which will take input data (either 2D, 3D, or eventually nD...), and perform transformations to generate the full set of data-columns that we will test against!
 This package will contain several 'sections':
  * __Transformation Functions:__ This is the mathematical side, and to start out we will be able to perform a variety of 1D or 2D transformations such as Log, Inverse, accumulate, Integrate, derrive, etc.
  * __Complex Transforms__: Some data transformations are combinations of the ones above (you can integrate AFTER you log-ify, for instance)
  * __Perform Transformation__: Wrapper function(s) to actually /do/ the transformations. This MAY be smart-ish and perform only specified ones, perhaps from a CONFIG or LIST? (That List and Config is an idea we'll talk about in __yNot__ probably!)
  * __Association Functions:__ This is optional, but once you've done the transformations (or before?) you can check how well the data CORRELATES/ASSOCIATES, both to itself and to a "Standard" dataset AKA starting with linear data and transforming it! This may give us "SCORES" for each transformation, which we can use to prioritize!
 
 __Note (again):__ Some of this may have Options to User-Interface, which is great but should DEFAULT be off, so we don't deal with it during Automated Processes!
 (Similarly, travis doesn't like User-Interfacing...)
 
 __Timeline + Milestones__:
  * 2020-04-21: List of the high-priority functions and Simple-Transformations, with progress and timeline to get them all done soon.
  * 2020-04-28: Passing Tests and can __HAND OFF__ to the classifier - a DataFrame of "all" the transformed data columns. Recieve Handoff from handling, and begin to Integrate
  * 2020-05-12: Complex Transforms - consider what other things we may want, and discuss feedback with group
  * 2020-06-09: Make Decision on Association functions and __HAND OFF__ if so. Otherwise, simply focus on new group priorities
  * 2020-06-23: IF yNot function is doing Configuration ideas, make Stretch-Goal learning gameplan... TBD...
 
 __Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work. Configuration?
 
 __Module List__:
 
 ### *SECTION: Basic 1D Transformations*
 #### transform_log():
  * INPUT: 1D data array with NO NEGATIVE VALUES
  * OUTPUT: Logrythmic transform of that data 
  * Note: Consider shifting or abs() for negative data? no? simply don't call log transform for negative data?
 #### transform_reciprocal():
  * INPUT: 1D data array - Limits tbd?
  * OUTPUT: all values inverted (1/x)
  * Note: twice should return itself!
 #### transform_cumsum():
  * INPUT: 1D data array - 
  * OUTPUT: cumulative sum of data (aka integrated with unit-steps)
  * Note: Not in high priority list?
 #### transform_1d_derrivative():
  * INPUT: 1D data array
  * OUTPUT: the step-by-step delta (Note: copy last delta to retain length?)
  * Note: Also not in high-priority list? Should also be able to complete the loop w/ cumsum.
 #### transform_exp():
  * INPUT: 1D data array - Limits?
  * OUTPUT: e^x of each datapoint. 
  * Note: may be redundant in general with log? should be able to complete that loop!
 #### transform_0to1():
  * INPUT: 1D data array, data handling case instructions
  * OUTPUT: that array shifted and scaled to the 0-to-1 basis (by FIRST shifting to min=0, THEN scaling to max=1)
  * Note: option to leave data alone if min is already 0-to-1, or if max After Shift is already 0-to-1 (case: data begins 0.2-0.4, can either scale 0-1 or leave as is!)

 ### *SECTION: Basic 2D Transformations*
 #### transform_2D_int():
  * INPUT: 2 equal size 1D arrays Y, X, to be integrated (Y)dx - [Optional offset value? to use as the Plus-C]
  * OUTPUT: The integral of Y dx (BOX? Trapz?) - offset if instructed to.
  * Note: Error handling? What if not Sorted/Linear in X? Should we sort by X first? (Or, what if reciprocal data ie CV Sweeps?)
 #### transform_2D_der():
  * INPUT: 2 equal size 1D arrays Y, X - Sorted? in either? 
  * OUTPUT: the Single-point derivatives dY/dX, also the offset so you /could/ integrate it back again!
  * Note: could use some sort of average or smoothing to reduce noise? However that would be LOSSY DATA PRACTICE!
 #### transform_prod():
  * INPUT: 2 equal size 1D arrays X, Y , [Optional Power arguments? or do those in the 1D cases and use as inputs?]
  * OUTPUT: Product of each x*y, maybe with power-math included (options)
  
### *SECTION: More Complex Transformations*
 #### transform_fourier_wavelets():
  * Ok so this is the only High-Priority one that I'm genuinely concerned with... while you "CAN" try to do a transform on a whole dataset, that gets noisy and lossy. 
    What I want to investigate is "Wavelet Filtering" Fourier transform, which we learned about at a Data Sci seminar last quarter?
    (Or otherwise, there's a whole realm of Signal-transforming science, I can research that...)
  * INPUT:  2 equal size 1D arrays X, Y - Sorted in X?? (Frequency range parameters? or is that the X-size?)
  * OUTPUT: 2D? output matrix or Meshgrid - in X-Freq space (for each wavelet size, return the match(-1 to 1?)*amplitude at each X?)
  * NOTE: This will have to be a group discussion- we need TEST DATA that should work in this space, and then we can report that back!
 #### multi_transform():
  * Wrapping function, to perform multiple transformations all together... Not sure which of these may be useful but I can see possible value in knowing the integral of a log function, for example. 
  * INPUT: X, [Y if 2D], Multiple transforms to perform... Is this what classes are for??
  * OUTPUT: Data output from the final transform listed. 
  * NOTE: This is low-priority, and should only be done if we convince ourselves that it's useful... RELATED, if we get the "Smart" learning functionality, maybe we can combine things this way
 #### transform___():
  *
  * INPUT: 
  * OUTPUT:
  
### *SECTION: Perform Transformation*
 #### get_xy():
  * Uses handling.py to load a file, check that we have xy data, and do a quick analysis on it! 
  * INPUT: file name
  * OUTPUT: each of 1D arrays X and Y, plus messages OR list of "Approved" transformations??
 #### perform_transform():
  * Wrapping function, to take some input data and return a dataframe with every transform that we want to use:
  * INPUT: 1D arrays X, Y, some sort of list or guiance for what transforms to do
  * OUTPUT: Pandas dataframe of X and Y with all of their transforms as requested.
  * NOTE: There might be a better way to do this - 
 #### generate_linear_transforms():
  * Creates a "sample" linear 2D dataset, possibly following range instructions, and performs all(?) transforms on the data
  * INPUT: [ALL OPTIONAL? Default 0-1 and X=Y], [List of transforms to do?? default True to perform "ALL"]
  * OUTPUT: pandas dataframe with all X-transformationa and all Y-transformations - (Maybe in Standard names? Maybe Not?)
  * NOTE: This could get messy - And maybe use the same wrapping function above to perform the listed tranforms?
  
 ### *SECTION: Association Functions*
  * NOTE: Not really planning these yet, that will be scoped or descoped based on Team Update by __2020-04-28__
 #### setup_correlation_matrix():
  * Sets up the 2D matrix of "scores" to judge the correlations 
 #### correlate_to_transforms():
  * For a given transform, determine (?) how correlated the data is to all other columns in the dataframe
 #### correlate_to_linear():
  * Compares the given transform to the linear_transform() function transforms. Any that correlate are probably good ideas?!?
 #### correlate_to_null():
  * Maybe compares the given transform to a flat line of low but nonzero values (all value = 0.1)? 
  * Not sure what's the best way to do this... 
  * What I'm TRYING to do is to identify/flag "BORING" data, which are probably BAD transforms to use?
 #### grade_all_transforms():
  * Wrapping function. Given a "fully" transformed dataset (or generate it here?), run all correlations and use some fancy math or grading (we generate?) to give each column a __SCORE__
  * The __SCORE__ should reflect how "Interesting" we think the data is (which is a topic for discussion, but all zeros is not interesting)
  * INPUT: dataframe ready to be "graded", OR give an XY dataset and we'll call the transform functions on it
  * OUTPUT: dictionary of Key,Values where each Key is a transform (or column key), and each Value is a "grade" to estimate how interesting we think the data may be
  * NOTE: This is SUPER arbitrary and is the 'creative' part of the STRETCH-GOALS of the project.s!
 #### grade_all_files(): 
  * Load all the files in a list, perform transformations, and grade. Hopefully this is a fast function so you can do a large list of files.
  * Then combine all the grades to get an average idea of what transforms we consider "good" 
  * INPUT: list of files
  * OUTPUT: dictionary of key, values as before, where results are averaged across dataset
  * ACTIONS: Optionally, save results as a report csv (or append to existing csv report??), to track grades over time
 #### load_transform_results():
  * IF we've run this program before, we should have a file that has previous "grades", this time based on the model training
  * If one transform shows up in a lot of the best-performing models, we should bump it to the top of the Transform To-Do List!
 #### combine_grades_results():
  * Somehow we should combine the 'grades' with the 'prior results' to get our new guess for what are the 'best' transformations to try
  * this will allow our model to do the highest-profile transfomations first and hopefully get good results in fewer attempts.
  
## recognition.py
This is the CNN, Baby! Finally we get to have the Machine do the Learning, instead of doing it ourselves!
Setup and Configure a CNN model, run optimization, and stash the result! 
Since we're stepping through this document and the package in a rough workflow order, this may also call functions from the previous two functions. 
It occurs to me that we haven't turned things into images yet? That can go wherever we want, but it may as well go here because the Image size and data might be an important factor in running the CNN that we need control over!

 __Timeline + Milestones__:
  * 2020-04-21: Initialize CNN, possibly with setup from preveious project EISy... Also set up some sort of image creator to feed it. __NEED: Raw Data input to use__
  * 2020-04-28: Manually able to generate an image set and run the CNN with a given list of metrics. __DEMONSTRATE For Group__
  * 2020-05-12: Integrate with hand-offs from above, given files and list of Transforms, can loop through multiple models and generate Report.
  * 2020-06-09: Improve "strategy" for looping, and discuss calculation time, duration, and next-steps. __HAND OFF__ to data_reporting.py

__Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work.

__Module List__:
 #### func_name():
 * Purpose and general description
 * INPUT: description of imputs- types and sources
 * OUTPUT: description of returns
 * ACTIONS: Any non-obvious actions the function can take other than IN/OUT (example - option to saving files)
 * NOTES:
 
 
## data_reporting.py
  This will handle the INPUT/OUTPUT and Reporting for the results of the model... Group vision here is TBD, but I have a sketch of the various outputs that we might want to report:
   * __Run_list.csv:__ A record with all of the manual parameters of each time we run the program. Should record meta info about how the run was set up, how many/how large files, and how long the run took (useful so we can get a feel for the computing power needed). Of course, also report some sort of 'success' metrics like the best performing result (and settings?) or the performance curve (how good were the best ones all together?). Append every new run with the run ID to point to it's result file...
   * __Report_yymmdd_ID.csv:__ A report generated for each run. */maybe/* only generate if we pass a certian performance standard? (aka the run "passes"). Maybe a large CSV including all of the parameters of the best 10-or-so model runs, (or all above a certain %-spec?)
   * __Best_Transformations.csv:__ A Meta-report that keeps records of any Transformations that occur often across the best performing models for the given dataset. Maybe simply append list with every run (or every *good* run), or perhaps keep this as a "LeaderBoard", which can be Replaced as new tests change the algorythms' opinion of the "best" transformations...
   
 __Timeline + Milestones__:
  * 2020-04-21: 
  * 2020-04-28: 
  * 2020-05-12: 
  * 2020-06-09: 

__Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work.

__Module List__:
 #### func_name():
 * Purpose and general description
 * INPUT: description of imputs- types and sources
 * OUTPUT: description of returns
 * ACTIONS: Any non-obvious actions the function can take other than IN/OUT (example - option to saving files)
 * NOTES:
 
 
 ## y_not.py *...or yNot.py ?*
  Mostly TBD, we included this as a catchall or pun, but have no specific plans.:
  
 __Timeline + Milestones__:
  * 2020-04-21: 
  * 2020-04-28: 
  * 2020-05-12: 
  * 2020-06-09: 

__Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work.

__Module List__:
 #### func_name():
 * Purpose and general description
 * INPUT: description of imputs- types and sources
 * OUTPUT: description of returns
 * ACTIONS: Any non-obvious actions the function can take other than IN/OUT (example - option to saving files)
 * NOTES:
 
  ## hardy.py *...or yNot.py ?*
  Core Python Notebook? Do we need/want a Core notebook to wrap all of the functions in?
  
 __Timeline + Milestones__:
  * 2020-04-21: 
  * 2020-04-28: 
  * 2020-05-12: 
  * 2020-06-09: 

__Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work.

__Module List__:
 #### func_name():
 * Purpose and general description
 * INPUT: description of imputs- types and sources
 * OUTPUT: description of returns
 * ACTIONS: Any non-obvious actions the function can take other than IN/OUT (example - option to saving files)
 * NOTES:
 
