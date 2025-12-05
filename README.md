# MLKnotsProject

## Introduction

Code stored in the following directory was used to tackle a classification/localisation problem involving a set of knots types. The goal of the project is to develop neural networks able to successfully classify different knot types (eventually with the same topological invariants) and try to predict their position along the string. In order to do so LAMMPS simulation of bead spring coarse-grained knots were exploited to generate a large dataset onto which we could train the model.

Code was developed using tensorflow 2.12 and its dependencies.

## Repository structure

- src: source files for model building/training/testing
  - helpers.py: collection of python functions to define environment parameters in the main function
  - loaders.py: collection of python functions to load input files from target directories
  - models.py: collection of tensorflow models
  - main.py: main executable to run with CLI arguments
- tf2.12.yml: conda environment used for model training/testing on Ubuntu 20.04.6 LTS

## How to run the code

The main.py file can be used for every need, once models and problems are defined. It takes in a set of command line options that allow to run the code accordingly to the preferences. In particular it is possible to define:

- problem (-p): Set of knots to use during training/testing
- datatype (-d): Type of data to use (XYZ, Signed 3D Writhe, ...)
- adjacent (-a): Adjacent data type from XYZ (deprecated)
- normalised (-n): Flag to apply a batch normalisation layer over the input layer to get a distribution with average zero and standard deviation 1
- nbeads (-nb): Number of beads of the bead spring knots simulated
- network (-t): Neural Network to be used
- epochs (-e): Number of epochs of training
- mode (-m): Select between test or train mode
- len_db (-ldb): Length of each knot database
- b_size (-bs): Batch size to be used during training
- master_knots_dir (-mkndir): Master directory containing knots files

Problems as set of knots are defined in the set_knots function in helpers.py file and can be customized by adding/removing knot types.

Networks can be instead added in models.py and then called in the generate_model function in helpers.py once the option is added.

## Data Loading

Data is stored on a server on the University of Edinburgh network within a master directory. Different datafiles for different knots are stored using the following data structure:

MASTER/KnotName/N{NBeads}/lp{lp}/DATATYPE/DATATYPE_Filenames.dat

### Data files structure
Depending on the type of data, data files have many different structures.

- XYZ: Three columns containing xyz coordinates with no headers
- SIGWRITHE: Three columns containing respectively Knot number, bead number and 3D signed writhe value
- SIGWRITHE: Three columns containing respectively Knot number, bead number and 3D unsigned writhe value
- DENSXSIGWRITHE: Five columns containing respectively Knot number, bead number, 3D signed writhe value, density and normalised 3D signed writhe value
- 2DSIGWRITHE: Nbeads columns containing 3D signed writhe value for each pair of beads

## Model saving

Best\*NN_Models\*\***/ are folders that contains weights of the best model per input feature and NN type, according to which \*** experiment was conducted (e.g 0_5 classifaction, SQRGRN8 classification, or 'Localise' Knot Localisation). Within each sub folder there is a file called saved_model.pb which is a type of file that can be read by tensorflow into a python script and already has the model weights adjusted to produce the best performance available per model, when fed a compatible input.

If test is run such directories contains also the confusion matrix for the best model for a test dataset obtained by splitting and shuffling the full dataset with a fixed seed.
