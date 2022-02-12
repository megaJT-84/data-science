# Tian Xiaoyang
# 26001904581
# roc curve plotting and parameters calculation

# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz


data = pd.read_csv('/Users/wcwe/Desktop/Fall 2021/Data science/13/Data_Science_Week-13_HQ2.csv')

def find_rates(classes, probs, threshold):
    true_posi=0 # initialize true positive as 0
    false_posi=0# initialize flase positive as 0
    true_nega=0# initialize true negative as 0
    false_nega=0# initialize flase negative as 0
    for i in range(len(classes)):
        if(probs[i] >= threshold): # if the probabilities go over the threshold
            if(classes[i] == 1):# classes are set to 1
                true_posi += 1
            else:
                false_posi += 1
        elif(probs[i] < threshold):# if the probabilities go below the threshold
            if(classes[i] == 0): # classes are set to 0
                true_nega += 1
            else:
                false_nega += 1
    #print((tp,fp,tn,fn)) 
    tpr = true_posi/(true_posi + false_nega) # calcualte true positive rate
    fpr = false_posi/(true_nega + false_posi) # calculate false positive rate
    return [fpr,tpr]

fpr,tpr =find_rates(data['Class'],data['Score'],1)

thresholds = data.iloc[:,-1]# assign all rows to the threshold 
final_points = []
for threshold in thresholds:
    rates = find_rates(data['Class'], data['Score'], threshold) # put data attributes in the find_rates function
    #print rates
    final_points.append(rates) # add new rates to the final points
print(final_points)# print the final points
#put the final points in true positive and true negative arrays

plt.plot([0, 1], [0, 1], 'k-', lw=2)
for i in range(len(final_points)-1):
    point1 = final_points[i]
    point2 = final_points[i+1]
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'red', lw=2)
plt.show()    