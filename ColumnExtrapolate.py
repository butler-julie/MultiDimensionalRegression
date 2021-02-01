##################################################
# Column Extrapolation
# Julie Butler Hartley
# Version 0.0.0
# Date Created: January 31, 2021
# Last Modified: January 31, 2021
#
# A program created to increase the number of columns in a matrix
# via sequential data formatting and regression algorithms.
##################################################

##############################
#IMPORTS
##############################
import numpy as np
import matplotlib.pyplot as plt

##############################
#FormatIMSRGData
##############################
def formatIMSRGdata (filename):
  "do nothing now"
##############################
#FormatFunctionData
##############################
def formatFunctionData (func, X, Y):
  return func(X, Y)
##############################
#LinearRegression
##############################
def LinearRegression(data):
  rows, cols = data.shape()
  design_matrix = np.zeros((cols-2, 2, rows))
  y_data = np.zeros((cols-2, m))
  
  for i in range (0, n-1):
    design_matrix[i][0] = data[:,i]
    desig_matrix[i][1] = data[:,i+1]
  
##############################
#RidgeRegression
##############################
