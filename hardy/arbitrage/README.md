## Instructions for using the tform_config file

An example transformation configuration file is shown below:

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart_TransformConfig.PNG" width=700 />

The transformation configuration relies on list of transformation a user intends to perform. The name of each transformation follows the variable naming rules for python. The names for transformation must be listed under <code>tform_command_list</code>.

The rules for plotting must be defined in <code>tform_command_dict</code>. The header of each entry in the dictionary corresponds to the transformation name which should be same as entered in the <code>tform_command_list</code>. The operations performed on the data are defined as arrays under this entry.

As many as <b>six</b> definitions can be entered under transformation command of dictionary. Each command follows the structure of \[column_number, mathematical operation, plotting_value].

The <code>column_number</code> corresponds to the column number according to the data in csv file. <code>Mathematical operation</code> is the operation that needs to be performed on this column and <code>plotting_value</code> corresponds to the color and orientation of the plot in final image that is to be read by machine learning algorithm.

The scheme for <code>plotting_values</code> is as follow:

```
- 0: Red on x-axis
- 1: Green on x-axis
- 2: Blue on x-axis
- 3: Red on y-axis
- 4: Green on y-axis
- 5: Blue on y-axis
```

Currently supported mathematical operations are as follows:

```
- raw: returns raw data without performing any operation
- exp: exponential
- nlog: natural log
- log10: logarithm tranformation with base 10
- reciprocal: reciprocal
- cumsum: cumulative sum
- derivative_1d: Differential with respect to 1 dimension
- derivative_2d: 2-D differentiation
- power: can be used for array multiplication or to take user defined power for array 
```
