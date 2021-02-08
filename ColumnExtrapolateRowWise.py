##################################################
# Column Extrapolation -- Row Wise
# Julie Butler Hartley
# Version 1.0.0
# Date Created: February 2, 2021
# Last Modified: February 8, 2020
#
# Extends the number of columns in a matrix using extrapolation with
# regression algorithms and sequential data formatting.
##################################################

##############################
# IMPORTS
##############################
# THIRD-PARTY IMPORTS
# for arrays and error analysis
import numpy as np 
# for plotting
import matplotlib.pyplot as plt
# for importing files
import csv

# LOCAL IMPORTS 
# Support methods for regression algorithms
from RegressionSupport import *
# Linear regression code
from LinearRegression import LinearRegressionAnalysis
# Ridge regression code
from RidgeRegression import RidgeRegressionAnalysis
# Kernel ridge regression code
from KernelRidgeRegression import KernelRidgeRegressionAnalysis

##############################
# FORMAT DATA
##############################
def formatData (filename, delimiter):
    """
        Inputs:
            filename (a string): a file name representing the file where
                the data is stored.  The import code expects the file to 
                contain only the matrix elements with columns separated by
                a tab and rows separated by a new line.
        Returns:
            formattedData (a 2D numpy array): a 2D array storing the matrix. 
                The columns of the matrix can be accessed via formattedData[:,i],
                and the rows of the matrix can be accessed via formattedData[i].
        Imports data from a file and formats it to be used in the matrix extrapolation
        codes.  Note, this does not format the data in an unusual way.  Many 
        matrices in Python are formatted using this format.
    """
    formattedData = []
    val = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter=delimiter)
        for row in reader:
            val.append(row) 
    for row in range(len(val)):
        formattedData.append([float(i) for i in val[row]])
    return np.asarray(formattedData)


##############################
# COLUMN EXTRAPOLATE
##############################
def columnExtrapolate (R, formattedData, num_new_cols, params, 
        isTuning, tuning_params):
    """
        Inputs:
            R (a class instance): An instance of one of the regression classes
                for sequential data extrapolation (i.e. an instance of 
                LinearRegressionAnalysis, RidgeRegressionAnalysis, 
                KernelRidgeRegressionAnalysis, or LassoRegressionAnalysis (to be 
                implemented soon)).
            formattedData (a 2D numpy array): the matrix to be used as the training
                data (its columns are what is to be extrapolated).
            num_new_cols (an int): the number of columns needed in the final,
                extrapolated matrix.
            params (a list): the list of parameters for the regression algorithm. 
                See README.md for an explanation of the paramaters and the correct 
                order.
            isTuning (a boolean): True means that hyperparameter tuning is performed
                on the first row of the matrix for find the optimal set of hyperparameters.
            tuning_params (a 2D list): The list of hyperparameters to be cycled through
                with hyperparameter tuning.  Pass an empty list if hyperparameter
                tuning will not occur.
        Returns:
            extrapolated_data (a 2D numpy array): the matrix with the correct number
                of columns, generated through sequential regression extrapolation.
        Performs sequential regression analysis on each row of a given matrix to 
        create a matrix with the desired number of columns.
    """
    # If hyperparameter tuning is to occur
    if isTuning:
        # Format the first row of the matrix to be used as training data
        X_train, y_train = time_series_data(formattedData[0])
        # Perform the hyperparameter tuning with the given list of parameters
        # Return the set of parameters that yields the lowest extrapolated MSE
        # score
        params = R.tune_serial_seq (tuning_params, X_train, y_train,
            len(formattedData[0])-4, formattedData[0], True, False)

    # Create a 2D array of zeros to hold the new, extrapolated matrix   
    extrapolated_data = np.zeros((len(formattedData), num_new_cols))
    # Iterate through each row of the training matrix
    for i in range(len(formattedData)):
        # Extract the current row and format it to be used as training data
        row = formattedData[i]
        X_train, y_train = time_series_data(row)
        # Using the current row as training data, extrapolate until the correct
        # length is reached (given by num_new_cols)
        new_row = R.unknown_data_seq(X_train, y_train, row, num_new_cols,
            len(row), params, False, 0.0)
        # Place the extrapolated row in the correct spot in the new matrix
        extrapolated_data[i] = new_row
    # Return the extrapolated matrix    
    return extrapolated_data

##############################
# ERROR ANALYSIS
##############################
def error_analysis (extrapolated_data, true_data):
    """
        Inputs:
            extrapolated_data (a 2D Numpy array): the matrix created with the method columnExtrapolate
            true_data (a 2D Numpy array): the known matrix that corresponds to what is predicted
                with the extrapolated_data matrix
        Returns:
            None
        Compares the matrix generated with the function columnExtrapolate to the known corresponding
        matrix.  Prints the mean squared error score to the console and produces a matrix plot representing
        the difference between the true result and the extrapolated result.
    """
    # Calculate and print the MSE score
    print ("MSE between true matrix and extrapolated matrix:")
    print(np.mean((true_data.flatten() - extrapolated_data.flatten())**2))
    # Produce the matrix plot showing the difference.  Add a color bar to show scale.
    plt.matshow(true_data - extrapolated_data)
    plt.colorbar()
    plt.show()

##############################
# PLOT MATRIX SAME XDATA
##############################
def plot_matrix_same_xdata (matrix, xdata, labels, graph_start = 0):
    for i in range(graph_start, len(labels)):
        plt.plot(xdata, matrix[:,i], label=labels[i])
    plt.legend()
    plt.show()

##############################
# PLOT MATRIX MATSHOW
##############################    
def plot_matrix_matshow (matrix):
    plt.matshow(matrix)
    plt.colorbar()
    plt.show()

##############################
# PLOT GROUND STATE COMPARISON
##############################
def plot_ground_state_comparison (xdata, num_total_cols, extrapolated_data, true_data):
    extrapolated_ground_states = []
    true_ground_states = []
    for i in range(num_total_cols):
        extrapolated_ground_states.append(np.amin(extrapolated_data[:,i]))
        true_ground_states.append(np.amin(true_data[:,i]))
    print("MSE between extrapolated and true ground states:")
    print(mse(np.asarray(extrapolated_ground_states), np.asarray(true_ground_states)))
    plt.scatter(xdata, extrapolated_ground_states, label='Extrapolated')
    plt.scatter(xdata, true_ground_states, label='True')
    plt.axvspan(0, 14, alpha=0.25)
    plt.legend()
    plt.show()

##############################
# MAIN PROGRAM (TO BE DELETED AFTER TESTING PHASE)
##############################
filename = '4He_E_Nmax_UVeff.csv'
data = formatData(filename, ',')
data_train = data[:,0:7]
data_predict = data
LR = LinearRegressionAnalysis()
data_estimate = columnExtrapolate(LR, data_train, 9, [True, True], False, [])
error_analysis(data_estimate, data_predict)
xdata = [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
labels = ['Nmax 4','Nmax 6','Nmax 8','Nmax 10','Nmax 12','Nmax 14','Nmax 16','Nmax 18','Nmax 20']
plot_matrix_same_xdata(data_estimate, xdata, labels)
plot_matrix_matshow(data_estimate)
xdata = [4, 6, 8, 10, 12, 14, 16, 18, 20]
plot_ground_state_comparison(xdata, 9, data_estimate, data_predict)


# Create training and test data matrices
#X = np.arange(0, 4, 0.1)
#Y = np.arange(0, 4, 0.25)
#Y_long = np.arange(0, 10, 0.25)
# Training Matrix
#Z = np.zeros((len(X), len(Y)))
#for i in range (len(X)):
#   for j in range(len(Y)):
#       Z[i][j] = X[i]**2 + Y[j]**2 
# Test Matrix        
#Z_long = np.zeros((len(X), len(Y_long)))        
#for i in range (len(X)):
#   for j in range(len(Y_long)):
#       Z_long[i][j] = X[i]**2 + Y_long[j]**2   
# Print the shapes of the two matrices to terminal        
#print("Size of training matrix: ", Z.shape)
#print("Size of test matrix: ", Z_long.shape)
# Make the regression instance, in this case an instance of the linear regressop class
#LR = LinearRegressionAnalysis()
# Generate the extrapolated matrix and compare it to the true matrix
#Z_predict = columnExtrapolate(LR, Z, len(Z_long), [True, True], False, [])
#error_analysis(Z_predict, Z_long)