# WiSig_Database 

This package provides several functions to create different datasets based on the WiSig database.  
You can upload the database using this link.
https://cores.ee.ucla.edu/downloads/datasets/wisig/ 

In the RiFyFi framework, we have made particular use of the ManySig database, which you can download from the site.


**SetWiSigcsv**: This function takes as input a structure describing the desired dataset and creates 4 CSV files : training data, test data and the corresponding labels for training and test data.
This function call "create_X_Y_WiSig".

**create_X_Y_WiSig**This function returns the data and the label matrices according to the description given in the function parameters.
Open the file with data and load the data and organise the sequences.

**LoadCSV_WiSig**: This function allows to load the CSV files corresponding to the database that the data structure described.
You can create a database with the previous functions and then use it many times to train different networks without the need to generate the data again.