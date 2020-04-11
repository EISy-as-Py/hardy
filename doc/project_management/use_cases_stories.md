# Functional Specifications
# And User-Case Stories
### __Updated 2020-03-11__
--------------------------------------------------------

## Meet the Team:
* __Alice (she/her)__, a Data Scientist post-doc who has years of data and research. She wants to set up the project, but then focus on other work like writing papers and grants. That's what PostDocs do, right?

* __Bob (he/him)__, a Grad Student Researcher designing high-throughput experiments. He generates way too much data to look at each test result, much less understand it all. He is setting up a program to make sure the work gets done as efficiently as possible. 

* __Chalie (they/them)__, a ChemE Undergrad working in the lab three days a week. Pretty competent, but doesn't have a background in either programming or Electrochemistry, and has never really seen an EIS before. They doesn't want to screw up, because he needs Grad School recs from Alice and Dave, and is afraid of Bob.

* __Dave (Your Highness)__, a stern old Professor who's lab they all work in. Your Highness is not a micro-manager, but simply expects results to be delivered on time. Dave is skeptical of the error and quality of high-throughput data, so needs to be convinced that all the results have well defined error explainations. The last programming Dave did was in Visual Basic in 1998.

* __Ernest (Important to Be)__, A Test-Engineer at a mid-stage startup who has to do all four of their jobs all at once. He doesn't complain about grant proposals or publish-or-perish, it's called do-your-job. Ernie works from 9 to 5 and goes home to his kids as soon as the day is over. And he makes what they all do combined. 

---------------------------------------------------------

## USER STORIES:

### Alice:
 * She will use her old data and __Import & Classify__ tools to make a classifyer and help categorize incoming data.
 * She can use her understanding of EIS and our __Simulation__ tools to make sure the classification set is large and comprehensive - Making up for any gaps in her existing database such as Fuel-Cell, Simple Capacitor, and Short-Circuit data sets which she may not have.
 * She can then use our __CNN Modeling__ tools to bootstrap a Classifyer model, to confidently identify EIS simulated information and her backlog of real data at the same time.
 * She will hand off this model to the rest of the lab, and get back to papers and research. If she gets her hands on more data, the model will already be set up to accept it and improve.

### Bob: 
 * He will set up his lab experiment to export data to a single folder on his computer or a server he has access to. [Future Work: Including binary _Export Tools_ to read directly from the machine's format such as biologic .mpt, etc]
 * Using a few files as examples, he can use __SQL Configuration__ tools to set up a SQL database for his project, and also save __Import Configuration__ files so that data importing is always as easy and reliable as possible!
 * On a daily basis, he can use our __Import Tools__ to grab all related files and process their raw data (F-series, not yet T-series).
 * The __SQL Export__ tools will put all of the raw data, experimental metadata, and processed results to the Lab's SQL Database. 
 * He will use our __Image Export__ tools to save images to be processed, AND/OR he can directly send these images (perhaps without actually making the image files???) to the __Classification Model__.
 * The __Classification Model__ will make several decisions, and report back each of the following statistics AS WELL AS the Confidence and Model-Training statistics that accompany the decision!
	* __Smooth, Noisy-But-Ok, Noisy-And-Fail__: Bob needs to know whether there was something wrong in the data, because some experiments may need to be repeated if the data is bad. There may even be important trends in why some data is noisier than others.
	* __Frequency Settings (Max too low? Min too High? Min too low?)__: Depending on the set-up, Bob maybe needs to change the settings on his equipment to make sure the results are reasonable. 
	* __Dominant Circuit Behavior__: What type of circuit-fit does this data represent? 
	* __MANY OTHER POSISBILITIES__: Bob will work with Alice to continue training the model, so that eventually it can do almost all the processing for them, and they can focus the NEXT Project: What to do with all that Wonderful SQL Meta-Data! But that's a project for another Semester...

### Charlie: 
 * They will be the most hands-on with the equiment and data collection, so need to have a reliable and user friendly interface to make sure that everything is running smoothly. Our __HTML User Interface__ will eventually be JUST what they need!
 * When they get to the lab, Bob will hand them a list of what should be run today. 
 * First, they will run iniital checks on the equipment using two or three well known test circuits (an RC, an LRC, and a Randle's setup). When they put those data sets into the __User Interface__, it will use the __Classification Model__ behind the scenes to confirm that these are confidently the correct results, that noise is low, and that no frequency parameters are unusual. 
 * They will save these Calibration results in a separate "Calibration Tracking" __SQL Database__ using the __GUI__, and then switch the program to load/store data in the main __Experiment SQL Database__. 
 * They will run experiments as normal, and do their homework like a good little student... If it's not automated yet, they will export the data and feed it into the __GUI__ every 30 minutes or so, and it will quickly give feedback on whether things are going according to plan. If not, they can make adjustments as recommended and/or text Bob for advice. 

### Dave: 
 * Dave's new best friend is the __HTML User Interface__, because unlike Grad-Students and Post-Docs it doesn't worry about his feelings and so he can see for himself all of the data with proper, accurate confidence intervals. 
 * Of course, Dave doesn't actually do that very often unless there is a paper coming soon. Dave really uses the __GUI Exporter__ to make beautiful plots and to check the results of Dave's own hypotheses about the experiments. 
 * This way, when that crazy black-box-machine-learning-voodoo-witchcraft tells Bob or Alice that they have discovered a fantastic breakthrough, Dave can explain that he knew that would happen all along.

### Ernest: 
 * Nobody at the company understands why it's so important to do EIS work and track this data over time, but Ernest does it anyway and nobody complains because he processes all of their other data for them and so who cares what other side-projects he is working on.
 * However, Ernest has integrated all of the above tools with his existing __Battery Lifetime Processing__ scripts. Now his __SQL Database__ reflects how the EIS changes as a function of battery lifetime. 
 * Ernest realizes that all of the batteries which start out as healthy RC circuits instead of Randles' circuits perform well up to 300 cycles, but then suddenly die! He doesn't know why that is (what is he, a chemist?) but he alerts the management and the Serial number trace allows them to design a new Quality-Control EIS test at the factory, improving their average lifetime performance by >200%
 * Ernest is promoted to head of R&D, where he can finally use his techno-sorcery to guide the chemists and researchers using the power of Data, and the company becomes the largest battery startup in the world. 
