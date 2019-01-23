#!/usr/bin/python

"""
Author: Elena Garcia
Course: Machine Learning
"""

# Import desired libraries
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import genfromtxt, savetxt
import matplotlib.pyplot as plt
import math
import itertools
import os
from operator import itemgetter


def get_data():
""" Opens the files with the training and test set
and creates a list to be used later"""
    dataset = open('training_set_SELECTED_400_FEATURES.txt','r')
    individuals_labels = open('individuals_labels_TRAININGSET.dat', 'r')

    global Ytrain
    for row in individuals_labels:
        row = row.strip()
        Ytrain = row.split('\t')

    global Xtrain
    Xtrain = []
    FirstLine = True
    for line in dataset:
        if FirstLine == True:
            FirstLine = False
        # there are lines with just \n every 2 rows
        elif line.startswith("\n"):
            continue
        # the first column of each row has the name of the SNP, so not useful
        else:
            line = line.strip()
            line = line.split("\t")
            Xtrain.append(line[1:])

    datasetTest = open('test_set_SELECTED_400_FEATURES.txt','r')
    individuals_labelsTest = open('individuals_labels_TESTSET.dat', 'r')

    global Ytest
    for row2 in individuals_labelsTest:
        row2 = row2.strip()
        Ytest = row2.split('\t')

    global Xtest
    Xtest = []
    FirstLine2 = True
    for line2 in datasetTest:
        if FirstLine2 == True:
            FirstLine2 = False
            # there are lines with just \n every 2 rows
        elif line2.startswith("\n"):
            continue
        # the first column of each row has the name of the SNP, so not useful
        else:
            line2 = line2.strip()
            line2 = line2.split("\t")
            Xtest.append(line2[1:])

def tree_and_accuracy():
""" Builds a tree with Xtrain (data) and Ytrain (labels) set. Calculates
the accuracy (misclassifications) for the Xtest and Ytest"""
    accuracyLabel, rf = mainRF(Xtrain, Ytrain, Xtest, Ytest)
    print accuracyLabel


def crossval():
""" cross-validation """

# Choose number of iterations for cv
    cv = 10

# # Only for GINI THRESHOLD cv
    # gini_variation = [1e-07, 0.25, 0.5, 0.75, 1]
    # gini_variation = [1e-10, 1e-07,1e-05,1e-03, 0.1]
    # error_cvGini = []
    # for gini_threshold in gini_variation:

# # Only for ESTIMATORS (n of trees in forest) cv
    # estim_variation = [2, 5, 10, 50, 100, 150]
    # error_cvEstimLabels = []
    # for estimator in estim_variation:

# # Onlye for DEPTH crossvalidation
    # depth_variation = list(range(21))[2:]
    # error_cvDepth = []
    # for depth in depth_variation:

# After this, for any kind of cv
    sum_error_cv = 0
    for iteration in range (0, cv):
        # Divide the dataset for training and testing
        # global X_train, X_test, y_train, y_test (optional)
        X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size = 0.3)
        accuracyLabel = mainRF(X_train, y_train, X_test, y_test) #, gini_threshold, estimator, depth)
        sum_error_cv += accuracyLabel

    print sum_error_cv/cv

        # error_cvGini.append(sum_error_cv/cv)
        # error_cvEstimLabels.append(sum_error_cv/cv)
        # error_cvDepth.append(sum_error_cv/cv)

    # length_tree = len(error_cvDepth)
    # print "Length tree: " + str(length_tree)

    # # Vary the plot as you want
    # plt.plot(depth_variation[:length_tree], error_cvDepth)
    # # # plt.ylim([0, 600])
    # # # plt.xlim([0, 150])
    # plt.xlabel("depth of the tree")
    # plt.ylabel("avg misclassifications")
    # plt.savefig("CV_depth_entropy")


def forest_classifier(Xtrain, Ytrain):
""" Generate the random forest
This is like in the webpage for scikit"""
#optional parameters: estimators, depth, gini

    global rf
    rf = RandomForestClassifier(n_estimators = 100, max_depth = 10)
    #OPTIONAL: n_estimators=estimators, max_depth = depth, min_impurity_split = gini
    rf = rf.fit(Xtrain, Ytrain)

def error_label(Xtest, Ytest):
""" Error calculation
 Find the error in the final leaves of the tree
 1. output label (no probabilities) """
    rf2 = rf.predict(Xtest)
    RightClass = 0
    WrongClass = 0
    suma = 0
    for i in range(0,len(rf2)):
        realnum = Ytest[i]
        num = rf2[i]
        if num == realnum:
            RightClass += 1
        if num != realnum:
            WrongClass += 1
    total = RightClass + WrongClass
    avg_errorlabel = float(WrongClass)/(total)
    with open("RF_test.txt", "a") as f:
        f.write(rf2)
        f.write("\n\nClass Prediction\n" + "n right classified: " + str(RightClass) + "\n"
        + "n wrongly classified: " + str(WrongClass)+ "\n"
        + "Out of " + str(total)
        + "\nAverage: " + str(avg_errorlabel))

    # return WrongClass, rf2
    return WrongClass, RightClass, total

def error_probability(Xtest, Ytest, rf):
""" Error calculation
Find the error in the final leaves of the tree
2. output probability"""

# If you want to see the vector with the probabilities for
# each sample in the test, un comment this line and the ones below
    # file2 = open("seeVectorsRF_depth.txt", "w")

    rf2 = rf.predict_proba(Xtest)
    count_for_order = 0
    sum_of_error = 0
    total_predictions = 0
    for vector in rf2:
        RealVector = [0.0, 0.0, 0.0, 0.0, 0.0]
        i2 = count_for_order
        realnum2 = int(Ytest[i2])
        RealVector[realnum2 -1] = float(1)
        count_for_order += 1
        # file2.write(str(vector)+"\n"+str(RealVector)+"\n")
        for continent in range(0, 4):
            if vector[continent] == RealVector[continent]:
                continue
            else:
                difference = (RealVector[continent]-vector[continent])**2
                sum_of_error += difference
                break
        total_predictions += 1
        avg_errorprobability = sum_of_error/total_predictions

    # with open("RF_CV_numberTrees.txt", "a") as f:
    #     f.write("\n\nProbability Prediction\n" + "Error " + str(sum_of_error) + "\nOut of " + str(total_predictions)
    #     + "\nAverage: " + str(avg_errorprobability))

    return avg_errorprobability

def mainRF(Xtrain, Ytrain, Xtest, Ytest):
""" This is a secondary main function to build the random forests,
called by tree_and_accuracy and permutation()"""
# OPTIONAL: estimators, depth (as parameter)
    forest_classifier(Xtrain, Ytrain) #optional: estimators, depth, gini

# Choose which one (first for tree_and_accuracy(), second for permutation())
    # accuracyLabel, rf2 = error_label(Xtest, Ytest)
    # return accuracyLabel, rf2
    WrongClass, RightClass, total = error_label(Xtest, Ytest)
    return WrongClass, RightClass, total

# Other options:
    # accuracyProb = error_probability(Xtest, Ytest, rf)

def roc_plot():
""" Creates a ROC plot,
    using the Xtrain(data) and Ytrain(labels) set to build a model,
    and Xtest and Ytest for prediction.
    The ROC plot is built using as threshold the probability
    of each sample to belong to each continent"""

    # First, build a tree
    rf = RandomForestClassifier(n_estimators = 100, max_depth = 10)
    rf = rf.fit(Xtrain, Ytrain)

    # Second, convert the list of Xtest and Ytest to dictionary
    real_dict = {}
    for element in range(len(Ytest)):
        real_dict[element] =  Ytest[element]
    rf3 = rf.predict(Xtest)
    predict_dict = {}
    for elm in range(len(Ytest)):
        predict_dict[elm] =  rf3[elm]

    # Then, do the rocplot
    TPR_continent, FPR_continent,continent_list = [], [], []
    for i in range(5):
        if i == 0:
            continent = 'EAS'
        if i == 1:
            continent = 'SAS'
        if i == 2:
            continent = 'EUR'
        if i == 3:
            continent = 'AMR'
        if i == 4:
            continent = 'AFR'
        TP, FP, TN, FN = 0, 0, 0, 0
        TPR, FPR = [0],[0]

        for individual in range(len(Ytest)):
            if str(Ytest[individual]) == str(i+1):
                FN += 1
            else:
                TN += 1
        estimations_test = dict()
        for individual in range(len(Ytest)):
            test = Xtest[individual]
            test = np.array(test).reshape((1, -1))
            rf2 = rf.predict_proba(test)
            rf2_list = []
            for ch in rf2[0]:
                if ch > 1:
                    ch = 1
                rf2_list.append(ch)
            estimations_test[individual] = rf2_list[i]
        first = True

        for prob_value in sorted(estimations_test.items(), key=itemgetter(1)):
            true_continent = real_dict[prob_value[0]]
            predicted_continent = i+1
            if not first:
                TPR.append(TP/float(TP+FN))
                FPR.append(FP/float(FP+TN))
            first = False
            if (str(predicted_continent) == str(true_continent)):
                TP += 1
                FN -= 1
            else:
                FP += 1
                TN -= 1
        TPR.append(TP/float(TP+FN))
        FPR.append(FP/float(FP+TN))
        continent_list.append(continent)
        TPR_continent.append(TPR), FPR_continent.append(FPR)

        # Save the results in a text file, before making the figure
        with open("results_rocplot.txt", "w") as f:
            f.write(str(TPR_continent) + "\n\n" + str(FPR_continent) + "\n\n" + str(continent_list))

def plot_rocplot():
""" Second part of the creation of a ROCplot, after roc_plot(),
it creates a new image with the ROCplot with a line per continent"""
    TPR, FPR, continent_list = [], [], []
    text = open("results_rocplot.txt", "r")
    trueline = True
    n = 1
    first = True
    for line in text:
        if n in [2, 4]:
            n += 1
            continue
        elif n in [1,3]:
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            line = line.replace("[", "")
            line2 = line.split("]")
            line2 = list(line2)
            list_for_cont = []
            for item in range(0,5):
                list_for_cont.append(line2[item].split(","))
        else:
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.split(",")
        if n==1:
            for i in range(0, 5):
                TPR.append(list_for_cont[i])
        elif n==3:
            for i in range(0, 5):
                FPR.append(line2[item].split(","))
        elif n==5:
            for i in range(0, 5):
                line[i] = line[i].replace("\'", "")
                continent_list.append(line[i])
        n += 1

    TPR[1] = TPR[1][1:]
    TPR[2] = TPR[2][1:]
    TPR[3] = TPR[3][1:]
    TPR[4] = TPR[4][1:]
    FPR[0] = FPR[0][1:]
    FPR[1] = FPR[1][1:]
    FPR[2] = FPR[2][1:]
    FPR[3] = FPR[3][1:]
    FPR[4] = FPR[4][1:]

    TPR = [map(float, x) for x in TPR]
    FPR = [map(float, x) for x in FPR]

    # Make the figure
    plt.figure(figsize=(7,5))
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    for cont in range(len(continent_list)):
        label_cont = continent_list[cont] + '. AUC = ' + str(integrate(TPR[cont],FPR[cont]))
        plt.plot(TPR[cont],FPR[cont],label=label_cont)
    plt.plot([0,1],[0,1],'k:',label='Random method')
    plt.legend(title='Random Forest',fontsize=10,frameon=False,loc=4)
    plt.ylabel('True positive rate'),plt.xlabel('False positive rate')
    plt.tight_layout()
    plt.savefig('ROCplot_RF.png')

def integrate(x,y):
""" Function part of the plot_rocplot(),
it calculates the Area Under the Curve"""
    # As the x values are not equally spaced between them, the numerical integration of the ROC plot (AUC <--> Area Under the Curve) will
    # be performed following the trapezoidal rule for a non-uniformal grid (Documentation: https://en.wikipedia.org/wiki/Trapezoidal_rule)
    integral = 0 # Assign a variable for the sum/integration
    first = True # Assign a boolean operator which will enable to perform a different operation in the 1st iteration of the following loop
    for (point_x, point_y) in zip(x,y):
        # In the first iteration, assign to the "previous_xcoordinate" and "previous_ycoordinate" variables the x and y coordinates, respectively, of the 1st point
        if first:
            previous_ycoordinate = point_y
            previous_xcoordinate = point_x
            first = False
        # In the rest of iterations, update the xcoordinate and ycoordinate values, and assign to the "previous_xcoordinate" and "previous_ycoordinate"
        #     variables the coordinates of the previous point
        # This is done because the trapezoidal rule requires the information of a point and the following one, in order to perform the numerical integration
        if not first:
            ycoordinate = point_y
            xcoordinate = point_x
            # Calculate the area of the trapezoide generated by a point and the following one, and update this area in the integral variable
            integral = integral + (xcoordinate - previous_xcoordinate)*(ycoordinate + previous_ycoordinate)
            previous_ycoordinate = point_y
            previous_xcoordinate = point_x
    # Lastly, the previou sum will be divided by 2, according to the trapezoidal rule equation
    integral = float(integral) / float(2)
    integral_round = round(integral, 3)
    return integral_round

def confusion_matrix(rf):
""" This function prints and plots the confusion matrix.
Normalization can be applied by setting 'normalize=True'."""

    # First, build a tree
    rf = RandomForestClassifier(n_estimators = 100, max_depth = 10)
    rf = rf.fit(Xtrain, Ytrain)

    #Second, create the confusion matrix
    # with a function by sklearn
    classes = ["East Asia", "South Asia", "Europe", "America", "Africa"]
    Ypredict = rf.predict(Xtest)
    CM = confusion_matrix(Ytest, Ypredict)
    normalize = False

    # Third, make the image
    # plt.figure(figsize=(10,8))
    plt.imshow(CM, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        CM = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis],2
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = CM.max() / 2.
    for i, j in itertools.product(range(CM.shape[0]), range(CM.shape[1])):
        plt.text(j, i, CM[i, j],
                 horizontalalignment="center",
                 color="white" if CM[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #save the image from the new window (otherwise some parameters
    # of the image cannot be seen)

def permutation():
""" Performs many trees with permutated labels.
Then, calculates the error for each and creates a histogram showing
the error of the real model vs the permutated-labels models """

    # Generate k-fold subsets of the original trainingset
    generate_subset = False
    if generate_subset:
        generate_kfold_subsets(tr_data,10)

    misclassified, goodclassified, estimations = mainRF(Xtrain, Ytrain, Xtest, Ytest)

    randomisation = True
    if randomisation:
        real_misclassifications = misclassified
        misclassified_list = []
        directory = 'random_labels/randomlabels_TRAININGSET'
        f = open('DT_permutation_approach.dat','w')
        for random_label_file in os.listdir(directory):
            file_labelsTRAIN = directory + '/'+random_label_file
            Ytrainnew = get_data_random(file_labelsTRAIN)
            misclassified, goodclassified, estimations = mainRF(Xtrain, Ytrainnew, Xtest, Ytest)
            misclassified_list.append(misclassified)
            f.write('%s\n'%(str(misclassified)))
        print misclassified_list
        plt.figure()
        plt.hist(misclassified_list,25, facecolor='green', alpha=0.75,label='Permutation approach models')
        plt.scatter(real_misclassifications,1,s=100,label='Real labels model')
        plt.ylim(0)
        plt.xlim(0,500)
        plt.legend(frameon=False,loc=2,fontsize=8)
        plt.xlabel('Misclassifications'),plt.ylabel('Counts'),plt.title('Permutation approach Random Forest')
        plt.savefig('randomisation_test.png')
        f.close()

def get_data_random(file_labelsTRAIN):
    """ For the permutation test, it opens one file
    with the labels. Every file has a different permutation
    pattern of the labels."""
    individuals_labels = open(file_labelsTRAIN, 'r')
    for row in individuals_labels:
        row = row.strip()
        YtrainRANDOM = row.split('\t')
    return YtrainRANDOM[:2004]

def main():
""" Main function.
- get_data() is necessary for every action,
- roc_plot() is followed by plot_rocplot() to create the image
- the rest are independent"""
    get_data()
    tree_and_accuracy()
    # roc_plot()
    # plot_rocplot
    # confusion_matrix()
    # permutation()

if __name__=="__main__":
    main()
