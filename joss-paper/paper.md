---
title: 'HARDy: Handling Arbitrary Recognition of Data in Python'
tags:
    - Feature Engineering
    - Kernel methods
    - Machine Learning
    - Python
authors:
    - name: Maria Politi
      email: politim@uw.edu
      orcid: 0000-0002-5815-3371
      affiliations: 1
    - name: Abdul Moeez
      email: amoeez@uw.edu
      orcid: 0000-0002-9582-0372
      affiliations: 2
    - name: David Beck
      email: dacb@uw.edu
      orcid: 0000-0002-5371-7035
      affiliations: 1,3
    - name: Stuart Adler
      email: stuadler@uw.edu
      orcid: 0000-0003-3472-0199
      affiliations: 1
    - name: Lilo Pozzo
      email: dpozzo@uw.edu
      orcid: 0000-0001-7104-9061
      affiliations: 1
affiliations:
    - name: University of Washington, Department of Chemical Engineering, Seattle, WA, USA
      index: 1
    - name: University of Washington, Department of Materials Science and Engineering, Seattle, WA, USA
      index: 2
    - name: eScience Institute, University of Washington, Seattle, WA, USA
      index: 3

date: 10 April December 2021
bibliography: paper.bib

---
`HARDy` is a python package that helps evaluate differences in data through feature engineering coupled with kernel methods. The package provides an extension to machine learning by adding layers of feature transformation and representation. The workflow of the package is as follows:

- _Configuration_: Setting attributes for user-defined transformations, machine learning hyperparameters or hyperparameter space
- _Handling_: Import data from `.csv` files and loads into the catalogue
- _Arbitrage_: Applies user defined numerical and visual transformations to the data
- _Recognition_: Machine Learning module that applies user defined hyper-parameter search space for training and evaluation of model
- _Data-Reporting_: Import results of machine learning models and reports it into dataframes and plots

# Background

High Throughput Experimentation (HTE) and High Throughput Testing (HTT) has increased the volume of experimental data available to scientists. One of the major bottlenecks in implementation of HTE and HTT is the data analysis. The need for automatic binning and classification has seen increase employment of machine learning approaches to analyze data for discovery of catalysts, process parameters, and classification of experimental data [@williams2019enabling; @becker2019low]. However, these solutions rely on specific sets of hyperparameters for their machine learning models to achieve the desired purpose. Furthermore, numerical data from experimental characterization of materials carries diversity in both features and magnitude. These features are traditionally extracted using deterministic models based on empirical relationships between variables of the process under investigation. As an example, X-ray diffraction (XRD) data is easier to characterize in linear form as compared to small angle X-ray scattering data, which requires transforming the axis to log-log scale. Moreover, the data may also require further processing or visualization through multiple transformations.

# Statement of Need

One of the most widely applied strategy to enhance the performance of machine learning model is Combined Automatic Machine Learning (AutoML) for CASH [@hutter2019automated]. However, these packages are only limited to hyper-parameter tuning and data features remain untouched. To improve the performance of machine learning model, binning, binarization, normalization, Box-Cox Transformations are popular feature engineering strategies for simple numeric data like ratings vs. reviews [@zheng2018feature]. For the data in text form, as in Natural Language Processing (NLP), flattening, chunking, and filtering are widely used methods [@zheng2018feature]. Based on these methods, high level packages like GeoDeepDive, Automan, and Brainwash have been implemented to explore, evaluate and extract features from text data [@anderson2013brainwash].
For other kinds of numerical data, Deep Feature Synthesis has shown promising results. Here features are generated from relational databases by performing multi-layer mathematical transformation operations [@kanter2015deep]. Another package employs transformation of data in Quantile Sketch Array (QSA) to improve the effectiveness of machine learning model [@nargesian2017learning]. Another aspect often ignored during evaluation of machine learning model is kernel methods based on feature engineering. Herein, in `HARDy` we present an infrastructure which aids in the identification of the best combination of numerical and visual transformations to improve data classification through convolutional neural networks. `HARDy` exploits the difference between human-readable images of experimental data (i.e. Cartesian representation) and computer-readable plots, which should maximize the data density presented to an algorithm and reduce superfluous information. `HARDy` makes use of a configuration file, fed to the open-source package `Keras-tuner`, removing the need for the user to manually generate unique parameters combinations for each neural network model to be investigated.


# Description and Use Case

The python-based package `HARDy` is a modularly structured package which classifies data using 2D convolutional neural networks. A schematic for the package can be found in figure 1.

![Information flow for HARDy](./images/HARDy_diagram.png)

The package was tested on a set of simulated small angle scattering (SAS) data to be classified into four particle different models: spherical, ellipsoidal, cylindrical and core-shell spherical. A total of ten thousand files were generated for each model. The dataset was generated using _sasmodels_,a submodule of _SasView_. The geometrical and physical parameters used to obtain each spectrum were taken from a published work by the Oak Ridge National Lab, discussing a similar classification[@ArchibaldRichardK2020Caas]. The four SAS models were selected as they present similar parameters and data features, which at times make it challenging to distinguish between them.

First, the pre-labelled data is loaded. A subset of the files, approximately three thousand files, is identified as the testing set, which will be composed of the same files for each ML model initialized in the same code run. A user-provided list of transformations, inputted through a configuration file, is applied to the data. Different trials can be specified, so that multiple sets of transformations can be investigated. For each trial, the data can be represented using two sets of images, a traditional Cartesian representation or our proposed method for increasing the information density contained in each graph. The latter visualization option was obtained by encoding the data into the pixel values of each channel composing a color image, for a total of six-channels available. Figure 2 shows an example of numerical and visual transformation applied to a spherical model data.

![Comparison between numerical and visual transformations on the same dataset. Panel (a) show the data visualized in cartesian coordinates in a lin-lin manned. Panel (b) represents the same data after both the scattering length q and the intensity I(q) have been logarithmically transformed. Panel (c) show the log-log data plotted in the blue and green channel of an RGB image](./images/panelplot.png)

The data is then fed into a convolutional neural network, whose hyperparameters and structure are defined using another configuration file. Alternatively, it is also possible to train multiple classifiers for a single transformation trial through the use of a tuner, by instead providing a hyperparameter space and a search method. The name of each SAS model was used as label for the data, which allows for further validation of the test set. The classification results are saved in `.csv` files, organized by each transformation run initially provided, as well as the best performing trained neural network. The package also allows to visually compare the performance of each transformation and, in the case of a tuning session, which hyperparameter combination yielded the higher accuracy through parallel coordinates plots, figure 3.  

![Parallel Coordinate Plot for the classification of SAS data into four models: spherical, cylindrical, ellipsoidal and core-shell sphere. The name of each run corresponds to the transformation applied to the scattering length q and intensity I. These results visualized the data in RGB plot. Only a limited number of transformation tested is shown here (see documentation)](./images/parallel_coordinate_rgb.png)

Both visualization techniques and the same combinations of numerical transformations were tested on the SAS data and the results were compared (see documentation). It is to note that all the CNN generated are validated using the same testing set. It can be noticed that using cartesian coordinates yields a higher number of instances in which the accuracy of the machine learning model trained was 25%, which corresponds to random chance in a four-class classification task (see documentation). On the other hand, the RGB plots show on average higher accuracy for the same combination of numerical transformations. To further validate the results, the test set was fitted using the label provided by the ML model. Additionally, for the case in which the outputted probability was below 70%, the data was also fitted using the second highest possible SAS model.

![Comparison of spherical fit (a) and core-shell sphere fit (b) on a file generated using the latter model. The machine learning model incorrectly labeled the data with a spherical model, however both model seem to fit the data similarly well. ](./images/panel_for_comparison.png)

The average chi-square parameter of the fitted data was determined to be 7.5. Approximately 11 \% of the data had a probability lower than 70\%. In all cases, if the neural network was not able to correctly label the data, the second highest probability was the correct one.

In conclusion, we believe we have demonstrated reliability in the classification task provide by `HARDy`. The code can also be use to guide modeling of data and can help identified the most probable model(s) to use, decreasing significantly the time spent on data analysis. Finally, the minimal user interaction required by the package allows for deployment of the task on a supercomputing cluster system, possibly removing the limitations given by the high computational power required to run these ML algorithms. All configuration files and scripts used to run the example presented in this paper can be found in the package documentation.


# Acknowledgements
This project was supported by: National Science Foundation through NSF-CBET grant no. 1917340, the Data Intensive Research Enabling Clean Technology (DIRECT) National Science Foundation (NSF) National Research Traineeship (DGE-1633216), the State of Washington through the University of Washington (UW) Clean Energy Institute and the UW eScience Institute

# References
