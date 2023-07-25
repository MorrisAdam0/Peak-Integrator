from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.integrate import fixed_quad
import plotly.graph_objects as go
import csv
import os


#=======================================================================================================================
# FILENAME
#=======================================================================================================================

filename = '../data/4_eald_cycle(S-0.75_Cd-0.55)_edited.csv'


#=======================================================================================================================
data = pd.read_csv(filename)
#remove all NAN values
data = data.dropna()

# adding column names to the data
data.columns = ['Potential (E/V)', 'Current (I/uA)']
param1 = data['Current (I/uA)']
param2 = data['Potential (E/V)']
x_label = 'Potential (E/V)'
y_label = 'Current (I/uA)'

# getting the values of each row
param_1 = param1.values
param_2 = param2.values

# creating the plotly figure
fig = go.Figure()
# adding a trace of the current vs voltage
fig.add_trace(go.Scatter(x=param_2, y=param_1, mode='lines', name='data', line=dict(color='blue', width=1)))

####################################################################################
# Fitting a polynomial regression to the data
####################################################################################

# creating a dataframe to store the results
results = pd.DataFrame(columns=['degree order', 'R^2 score'])
volts = param_2.reshape(-1, 1)  # reshaping voltage data from 1D array to 2D array

scaler = StandardScaler().fit(
    volts)  # scales the data so it has a mean of 0 and a sd of 1. the fit method calculates the mean and sd
volts = scaler.transform(
    volts)  # applies the scaling to the volts data. trasnofrm standardises the data by substracting the mean of the data and dividing by the sd
for degree in range(1, 150):
    poly = PolynomialFeatures(degree=degree)  # degree is the polynomial degree to be calculated.
    X = poly.fit_transform(volts)  # transforms the data into a polynomial of the specified degree.
    # creating a matrix X that includes original volts data as well as all polynomial features up to degree, degree.

    reg = LinearRegression().fit(X,
                                 param_1)  # fit trains the model on the input data X and the output variable current.
    # LinearRegression fits a linear regression model to the data X (the input features) and y (the output features, current).

    r2 = r2_score(param_1, reg.predict(
        X))  # calculates the coefficient of determination (r^2). takes the dependent variable, current,
    # and the predicted values of dependent variable obtained from the regression
    results = results.append({'degree order': degree, 'R^2 score': r2},
                             ignore_index=True)  # appends the results to the dataframe

print(results)

max_r2 = results['R^2 score'].max()

max_index = results['R^2 score'].idxmax()

max_degree = results.loc[max_index, 'degree order']

print(f'The maximum R^2 score is {max_r2} and the degree order is {max_degree}')

best_degree = int(max_degree)
poly = PolynomialFeatures(degree=best_degree)
X = poly.fit_transform(volts)

powers = poly.powers_
coef = np.polyfit(X[:, 0], param_1, best_degree)  # calculates the coefficients of the polynomial of degree best_degree

'''
poly_func = ''
for i, power in enumerate(powers):
    if power[0] == 0:
        poly_func += f'{coef[i]:.2f}'
    else:
        poly_func += f'{coef[i]:.2f}x^{power[0]}+'

print(poly_func)
'''
reg = LinearRegression().fit(X, param_1)
fig.add_trace(go.Scatter(x=param2, y=reg.predict(X), mode='lines', name='regression', line=dict(color='red', width=1)))
fig.update_layout(title='Polynomial Regression of degree {} (R2={:.6f})'.format(best_degree, max_r2),
                  xaxis_title=x_label, yaxis_title=y_label)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.0))
fig.show()

#########################################################################################

value = float(input('Enter an x-coordinate: '))  # user enters x coordinates
value_2 = float(input('Enter an x-coordinate: '))

idx = data.loc[data['Potential (E/V)'] == value].index[0]  # determines the index of the x coordinate in the data
idx_2 = data.loc[data['Potential (E/V)'] == value_2].index[0]

x_0 = param2.iloc[idx]  # extracts the exact x coordinate from the data
y_0 = param1.iloc[idx]

x_1 = param2.iloc[idx_2]
y_1 = param1.iloc[idx_2]

y_1 = reg.predict(poly.fit_transform(scaler.transform(np.array(value).reshape(-1,
                                                                              1))))  # predicting the y value for the user-provided x coordinate using the previously trained model
# firstly reshaped to 2D array (same shape as training data), and then standardised, then transformed into a polynomial feature matrix, then y value predicted.
y_2 = reg.predict(poly.fit_transform(scaler.transform(np.array(value_2).reshape(-1, 1))))

# calculations for a straight line between the two points
x = [x_0, x_1]

m = (y_2 - y_1) / (x_1 - x_0)
b = y_1 - m * x_1
m = np.isscalar(m)
b = np.isscalar(b)

y = [y_1[0], y_2[0]]
'''
y = [m * xi + b for xi in x]
'''

fig.add_trace(
    go.Scatter(x=x, y=y, mode='lines', name='Line between selected Potentials', line=dict(color='green', width=1)))
fig.update_layout(title='Polynomial Regression of degree {} (R2={:.6f})'.format(best_degree, max_r2),
                  xaxis_title=x_label, yaxis_title=y_label)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.0))


fig.show()

y_pred = reg.predict(X)

x = np.linspace(x_0, x_1,
                len(y_pred))  # creates an array, x, containing a sequence of evenly spaced numbers between x_0 and x_1 with the same length as y_pred


def line(x, m, b):
    return [m * i + b for i in
            x]  # defines a function line that takes x, m and b as arguments and returns an array where each element is a value of the equation
    # for each element i in the array x


y_line = line(x, m,
              b)  # calls the line function and assigns the result to y_line. produces an array that represents the line of best fit


# Define the line function
def line_2(x, m, b):
    return m * x + b


# Compute the coefficients of the line of best fit
m = (y_1 - y_2) / (x_1 - x_0)
b = y_1 - m * x_1

# Compute the integral of the line of best fit between x_0 and x_1
area_2 = fixed_quad(lambda x: line_2(x, m, b), x_0, x_1, n=1000)[0]
if area_2 < 0:
    area_2 = area_2 * -1
else:
    pass

# convesrion into arrays
volts = np.array(volts)
y_pred = np.array(y_pred)
y_line = np.array(y_line)

# converts into 1D arrays
volts_flat = volts.flatten()
y_pred_flat = y_pred.flatten()
y_line_flat = y_line.flatten()

# calculate the area
# area = fixed_quad(lambda x: abs(np.interp(x, volts, y_pred) - np.interp(x, volts, y_line)), x_0, x_1, n=1000)
area = fixed_quad(lambda x: abs(np.interp(x.flatten(), param2, y_pred_flat)), x_0, x_1, n=1000)

area_calc = area[0] - (area_2) #for reduction peaks you need to divide the area_2 by -1
print(f'The area of the specified region is {area_calc}')
print(f'Area_0: {area[0]}, Area_2: {area_2}')
area_value = area_calc

charge_C = area_value * 0.050

num_elec = charge_C / 1.602E-19

print(f'The charge (C) is {charge_C} and \n the number of electrons involved is {num_elec}')

results_filename = 'Electrochemical_stripping_data.csv'
if os.path.isfile(results_filename):
    with open(results_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, value, value_2, area_value, charge_C, num_elec])
else:
    with open(results_filename, 'w', newline='') as file_2:
        writer = csv.writer(file_2)
        writer.writerow(['Filename', 'x_coordinate_1', 'x_coordinate_2', 'Area', 'Charge (C)', 'Number of electrons'])
        writer.writerow([filename, value, value_2, area_value, charge_C, num_elec])
