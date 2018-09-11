'''
Created on Jul 24, 2018

@author: tevin

This method is used for finding the correlation between height and temperature/relative humidity.
'''
import matplotlib
#matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
from sklearn.model_selection import train_test_split 
from builtins import int
from matplotlib.pyplot import contour, xticks, ylim
from PIL.ImageFilter import numpy
from scipy.stats import ttest_ind
from numpy import average

 
def graphDataMain(fname):
    df = pd.read_csv(fname)
    print(df.shape)
    print(df.head(5))
    print(df.describe())
#     x = df["Time"][:10080]
#     y1 = df["Ground"][:10080]
#     y2 = df["Middle"][:10080]
    df.plot(x='Time', y= ['Ground',"Middle"], style='o', markersize = 3)  
#     plt.scatter(x,y1,label = "Ground")
#     plt.scatter(x,y2,label = "Middle")
    plt.legend()
    plt.title('Time vs Canopy Temperature Secondary Forest')  
    locs = []
    for i in range(0,20,2):
        locs.append(i*720)
    labels =["JUlY  7","JUlY  8","JUlY  9","JUlY  10","JUlY  11","JUlY  12","JUlY  13","JUlY  14","JUlY  15","JUlY  16","JUlY  17"]
    xticks(locs,labels,rotation=25)
    ylim(16,30) 
    plt.xlabel('Time')  
    plt.ylabel('Celcius')  
    plt.show() 
    
def calculateSLR(fname):
    #calculates single linear regression between temperature and height
    df = pd.read_csv(fname)
    print("Shape",df.shape)
    print()
    print("First 5 lines\n", df.head(5))
    print()
    print("Statistics\n", df.describe()) 
    
    X = df.iloc[:,-1:].values[:10080] 
    y = df.iloc[:, 2].values[:10080]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)    
    regressor = linear_model.LinearRegression()  
    regressor.fit(X_train, y_train)  
    intercept = regressor.intercept_ 
    coefficient = regressor.coef_[0]
    print("Equation");
    print("Middle Temperature = (" + str(coefficient) + ")Ground Temperature + (" + str(intercept) + ")");
    print()
    y_pred = regressor.predict(X_test)  


    print("Metrics")
    print('Correlation Coefficient(R2):', metrics.r2_score(y_test, y_pred)) 
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    x = df["Ground"][:10080]
    line = [coefficient*i+intercept for i in x]
    y = df["Middle"][:10080]
#     df.plot(x='Ground', y= ['Middle'], style='o', markersize = 3)
    plt.scatter(x, y)
    plt.plot(x, line, 'r', label='y={:.2f}x+{:.2f}'.format(coefficient,intercept))  
    plt.legend()
    plt.title('Ground vs Middle Temperature New Forest')  
    plt.xlabel('Ground Temperature (Celcius)')  
    plt.ylabel('Canopy Temperature (Celcius)')  
    plt.show()
    
def anomoly(fname):
    df = pd.read_csv(fname)
    X = df.iloc[:,-1].values #Ground
    y = df.iloc[:, 2].values #Middle
    anom = numpy.array(y[:10080])-numpy.array(X[:10080])
    T = df.iloc[:,0:1].values 
    x = range(len(T[:10080]))
    line = [0 for i in x]
    plt.plot(x, line, 'r')
    plt.plot(x, anom, markersize = 3)  
    plt.title('Middle-Ground Temperature Difference Secondary Forest')  
    locs = []
    for i in range(0,16,2):
        locs.append(i*720)
    labels =["JUlY  7","JUlY  8","JUlY  9","JUlY  10","JUlY  11","JUlY  12","JUlY  13","JUlY  14","JUlY  15"]
    xticks(locs,labels,rotation=25)
    ylim(-1,3)  
    plt.ylabel('Difference (Celcius)')  
    plt.show() 
    
def differenceComparison(file1,file2):
    df = pd.read_csv(file1)
    X = df.iloc[:,-1].values[:10080] #Ground
    y = df.iloc[:, 2].values[:10080] #Middle
    cat1 = numpy.array(y)-numpy.array(X)
    df = pd.read_csv(file2)
    X = df.iloc[:,-1].values[:10080] #Ground
    y = df.iloc[:, 2].values[:10080] #Middle
    cat2 = numpy.array(y)-numpy.array(X)
    plt.hist([cat1,cat2], color = ['blue','red'] , edgecolor = 'black', bins = 30, label = ["New Forest", "Secondary Forest"])
    plt.legend()
    plt.title("Difference Between Middle and Ground Temperature Histogram")
    plt.xlabel("Difference (Celcius)")
    plt.ylabel("Number")
    plt.show()
    print("Average Difference New Forest:",numpy.average(cat1))
    print("Average Difference Old Forest:",numpy.average(cat2))
    print(ttest_ind(cat1, cat2))
    print()
#     b1 = [x for x in cat1 if x >=2]
#     b2 = [x for x in cat2 if x >=2]
#     print("Quantity Difference over 2 Celcius New Forest:",len(b1))
#     print("Quantity Difference over 2 Celcius Old Forest:",len(b2))
#     print()
#     b1 = [x for x in cat1 if x <=0]
#     b2 = [x for x in cat2 if x <=0]
#     print("Quantity Difference Under 0 Celcius New Forest:",len(b1))
#     print("Quantity Difference Under 0 Celcius Old Forest:",len(b2))
#     print()
    
def canopyComparison(file1,file2):
    df = pd.read_csv(file1)
    cat1 = df.iloc[:, 1].values #Canopy
    df = pd.read_csv(file2)
    cat2 = df.iloc[:, 1].values #Canopy
    print("Average Canopy Temperature New Forest:",numpy.average(cat1[:10080]))
    print("Average Canopy Temperature Old Forest:",numpy.average(cat2[:10080]))
    print(ttest_ind(cat1[:10080], cat2[:10080]))
    print()
    
def groundComparison(file1,file2):
    df = pd.read_csv(file1)
    cat1 = df.iloc[:, -1].values #Ground
    df = pd.read_csv(file2)
    cat2 = df.iloc[:, -1].values #Ground
    print("Average Ground Temperature New Forest:",numpy.average(cat1[:10080]))
    print("Average Ground Temperature Old Forest:",numpy.average(cat2[:10080]))
    print(ttest_ind(cat1[:10080], cat2[:10080]))
    print()
    
def middleComparison(file1,file2):
    df = pd.read_csv(file1)
    cat1 = df.iloc[:, 2].values #Middle
    df = pd.read_csv(file2)
    cat2 = df.iloc[:, 2].values #Middle
    print("Average Middle Temperature New Forest:",numpy.average(cat1[:10080]))
    print("Average Middle Temperature Old Forest:",numpy.average(cat2[:10080]))
    print(ttest_ind(cat1[:10080], cat2[:10080]))
    print()
    
# def exponentialRegression(fname):
#     #creates an exponential regression for all the times, and then averages them
#     df = pd.read_csv(fname)
#     x = [0.001,4.85,9.2]
#     l = df.iloc[:,-1].values #Ground
#     m = df.iloc[:, 2].values #Middle
#     n = df.iloc[:, 1].values #High
#     coef = []
#     inter = []
#     for i in range(len(n)):
#         print("yes")
#         regressor = linear_model.LinearRegression()  
#         regressor.fit([numpy.log(x)], [l[i],m[i],n[i]])
#         intercept = regressor.intercept_ 
#         coefficient = regressor.coef_[0]
#         coef.append(coefficient)
#         inter.append(intercept)
#     print(average(coef)) 
#     print(average(inter))
    
if __name__ == '__main__':
    file1 = "Temperature_Profiles_plot1.csv"
    file2 = "Temperature_Profiles_Plot2.csv"
    file3 = "Temperature_Profiles_Plot3.csv"
#     graphDataMain(file2)
#     calculateSLR(file1)
#     anomoly(file2)
    differenceComparison(file1, file2)
#     canopyComparison(file1, file2)
    groundComparison(file1, file2)
    middleComparison(file1, file2)
#     exponentialRegression(file1)
   
    