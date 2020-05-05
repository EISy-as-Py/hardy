### Project HARDy: Main Package
Congrats, you're in the codebase for project HARDy. 
You can see that so far (edited 2020-05-05), we don't have a "Main" function for the package, everything is contianed in these submodule folders.


### Module Descriptions:
In general, the folders here are submodules that each have their own __init__.py file and can be called independantly, but they also follow a workflow into eachother!

The __Handling__ module should call no others, but is available to be called on as needed to load, save, and pre-process data.

The __Arbitrage__ module has the arbitrage.py function in it (which is mostly the class wrapper function to store and execute data transformations, and then to produce images as needed!
 * Inside this module is also transformations.py, which contains plans and code detailing all of the 1D, 2D, and more complex transformations that the arbitrage.py class will call!
 
The __classifier__ module (To be renamed __Recognition???__) has all of our CNN construction code using the Keras package, as well as wrapping functions to call the above modules and execute data operations

