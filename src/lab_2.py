"""
This is a lab for Performing Python Lab 2 for Coursera AI Course
"""

"""
Imports
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline

"""
main block
"""
def main():
    # Main Application Logic
    # Load the data
    url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

    # Understand the data
    # MODEL YEAR, MAKE, MODEL, VEHICLE CLASS, ENGINE SIZE, CYLINDERS, TRANSMISSION, FUEL TYPE
    # FUEL CONSUMPTION IN CITY, FUEL CONSUMPTION in HWY, FUEL CONSUMPTION COMBINED, 
    # FUEL CONSUMPTION COMBINED MPG, CO@ Emissions

    # Load the data
    df = pd.read_csv(url)
    # Verify Sample
    df.sample(5)
    
    # Explore and Select Features
    df.describe()

    # Drop categoricals and useless columns
    df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
    
    # Examine Correlations 
    # Eliminate and strong dependencies
    df.corr()
    # CO2Emissions is above 85% for all features
    # Engine Size and Cylinders are highly correlated, but Engine Size is more correlated to target
    # All fuel are correlated and Fuel Consumption is most correlated with target
    # Drop further uneeded fields
    df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',],axis=1)
    df.head(9)

    # Consider Scatter Plot
    axes = pd.plotting.scatter_matrix(df, alpha=0.2)
    # need to rotate so we can see them
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    
    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()

    # Extract the Input features and labels from the data set
    X = df.iloc[:,[0,1]].to_numpy()
    y = df.iloc[:,[2]].to_numpy()

    # Preprocess selected features
    # Standard way is to subtract the mean and divie by the standard deviation
    from sklearn import preprocessing

    std_scalar = preprocessing.StandardScaler()
    X_std = std_scalar.fit_transform(X)

    pd.DataFrame(X_std).describe().round(2)

    # Create Train and Test Datasets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

    # Begin muiltiple linear regression model
    from sklearn import linear_model

    # Create a model object
    regressor = linear_model.LinearRegression()

    # Train the model in the training data
    regressor.fit(X_train, y_train)

    # Print the coefficients
    coef_ = regressor.coef_
    intercept_ = regressor.intercept_

    print('Coefficients: ',coef_)
    print('Intercept: ',intercept_)

    # Transform from abstract standard space back to model space
    # Get standard scaler's mean and standard deviation
    means_ = std_scalar.mean_
    std_devs_ = np.sqrt(std_scalar.var_)
    
    # the least sqaures parameters can be calculated
    coef_original = coef_ / std_devs_
    intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

    print('Coefficients: ', coef_original)
    print('Intercept: ', intercept_original)

    # Ensure X1, X2, and y_test have compatible shapes for 3D plotting
    X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
    X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

    # Create a mesh grid for plotting the regression plane
    x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), 
                                                                  np.linspace(X2.min(), X2.max(), 100))

    y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf

    # Predict y values using trained regression model to compare with actual y_test for above/below plane colors
    y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
    above_plane = y_test >= y_pred
    below_plane = y_test < y_pred
    above_plane = above_plane[:,0]
    below_plane = below_plane[:,0]

    # Plotting
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points above and below the plane in different colors
    ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
    ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

    # Plot the regression plane
    ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

    # Set view and labels
    ax.view_init(elev=10)

    ax.legend(fontsize='x-large',loc='upper center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect(None, zoom=0.75)
    ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
    ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
    ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
    ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
    plt.tight_layout()
    plt.show()

    # 3D Slices Instead
    plt.scatter(X_train[:,0], y_train,  color='blue')
    plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    plt.scatter(X_train[:,1], y_train,  color='blue')
    plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
    plt.xlabel("FUELCONSUMPTION_COMB_MPG")
    plt.ylabel("Emission")
    plt.show()
    
    # Excersize 1 - Determine paramaeters for the best-it linear regression line for CO2
    X_train_1 = X_train[:,0]
    regressor_1 = linear_model.LinearRegression()
    regressor_1.fit(X_train_1.reshape(-1,1), y_train)
    coef_1 = regressor_1.coef_
    intercept_1 = regressor_1.intercept_
    
    print('Coefficients: ', coef_1)
    print('Intercept: ', intercept_1)

    # Exercise 2 - Create Scatter Plot of CO2 Emmission against Engine Size
    plt.scatter(X_train_1, y_train, color='blue')
    plt.plot(X_train_1, coef_1[0] * X_train_1 + intercept_1, '-r')
    plt.xlabel("Engine Size")
    plt.ylabel("Emission")
    plt.show()

    # Exercise 3 - Generate scatter plot off of test data
    X_test_1 = X_test[:,0]
    plt.scatter(X_test_1, y_test, color='blue')
    plt.plot(X_test_1, coef_1[0] * X_test_1 + intercept_1, '-r')
    plt.xlabel("Engine Size Test")
    plt.ylabel("CO2 Emission Test")
    plt.show()

    # Exercise 4 - repeat modeling but with FUELCONSUMPTION
    X_train_2 = X_train[:,1]
    regressor_2 = linear_model.LinearRegression()
    regressor_2.fit(X_train_2.reshape(-1, 1), y_train)
    coef_2 = regressor_2.coef_
    intercept_2 = regressor_2.intercept_

    print('Coefficients: ', coef_2)
    print('Intercept: ', intercept_2)

    # Exercise 5 - Generate Scatter Plot
    X_test_2 = X_test[:,1]
    plt.scatter(X_test_2, y_test, color='blue')
    plt.plot(X_test_2, coef_2[0] * X_test_2 + intercept_2, '-r')
    plt.xlabel('Fuel Consumption: ')
    plt.ylabel('CO2 Emissions: ')
    plt.show()


if __name__=="__main__":
    main()




