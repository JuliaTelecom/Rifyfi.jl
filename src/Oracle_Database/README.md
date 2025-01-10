# Oracle Database 

This package provides several functions to create different datasets based on the Oracle database.  
You can upload the database using this link.
https://www.genesys-lab.org/oracle 

In particular:
#Dataset1:  Raw IQ samples of over-the-air transmissions from 16 X310 USRP radios 
https://repository.library.northeastern.edu/files/neu:m044q520q 

There are two different records: Run 1 and Run 2, then different distances are suggested such as 2ft, 8ft .... You can use only one distance, some group of distances or all distances to create your dataset.

**SetOracleCSV**: This function takes as input a structure describing the desired dataset and creates 4 CSV files: training data, test data and the corresponding labels for training and test data.
This function calls the "create_X_Y_Database" function.

**create_X_Y_Oracle**: this function returns the data and the label matrix according to the description given in the function parameters.
Open the file with the data and load the data and organise the different sequences.

**LoadCSV_Oracle**: This function allows to load the CSV files corresponding to the database that the data structure described.
You can create a database with the previous functions and then use it many times to train different networks without the need to generate the data again.