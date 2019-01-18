###### "Machine Learning" course, Vrije Universiteit Amsterdam #################
###### Author: Alberto Gil (student ID 2595259) ################################

# Import desired libraries
import random
import os
import matplotlib
import pylab as plt
from operator import itemgetter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# Define training set, test set, corresponding labels and file with all the real labels
tr_data = 'Data/training_set_transposed/training_set_SELECTED_400_FEATURES.txt' #GOOD
tr_labels = 'Data/labels/individuals_labels_TRAININGSET.dat'  # GOOD
test_data = 'Data/test_set_transposed/test_set_SELECTED_400_FEATURES.txt' # GOOD
goldstandard = 'Data/integrated_call_samples_v3.20130502.ALL.panel'

# Define possible SNP values, and possible continent values (possible output)
snp_possible_values = ['0','0.5','1']
continent_list = ['1','2','3','4','5']

# Gives an output list with all the features (SNP) present in the dataset
def read_SNP(inputfile):
    # Create a list for appending the number of SNP
    snp_list = []
    # Read the header of the inputfile (contains the SNP info)
    with open(inputfile) as fin:
        firstline = fin.readlines()[0].strip('\r\n').split('\t')
        for SNP in firstline:
            if SNP != 'ID': #avoid appending the 1st element of the line ('ID')
                # append in a list the name of the snps
                snp_list.append(SNP)
    # Return the list of SNP read in the dataset
    return snp_list


# Creates an empty probability dictionary (usage: probability[SNP][SNP_value][continent_value])
#     which will contain the probability that a certain SNP has a certain value (0, 0.5 or 1)
#     if the SNP is from an individual from a certain continent
def create_probability_dictionary(snp_list,snp_possible_values,continent_list):
    # initialise probability dictionary
    probability = dict()
    for snp in snp_list:
        probability[snp] = dict()
        for value in snp_possible_values:
            probability[snp][value] = dict()
            for continent in continent_list:
                probability[snp][value][continent] = 0
    # Return the empty dictionary
    return probability


# Computes the prior_probability of being from a continent
#     l is the Dirichlet weight (adjusted with cross-validation)
def prior_probability(labelfile,l):
    # Create a prior probability dictionary (key: continent. value= probability)
    prior_probability_dict = dict()
    # Update the dictionary with the counts of being from a certain continent
    with open(labelfile) as fin:
        firstline = fin.readlines()[0].strip('\t\n')
        for continent in firstline.split('\t'):
            if continent not in prior_probability_dict:
                prior_probability_dict[continent] = 1
            else:
                prior_probability_dict[continent] += 1
    # Normalize the counts for obtaining probabilities
    total_individuals = sum(prior_probability_dict.values())
    for continent in prior_probability_dict.keys():
        prior_probability_dict[continent] = float(prior_probability_dict[continent] + l)/float(total_individuals + 2*l)
    # Return the prior probability dictionary
    return prior_probability_dict

# Trains a naive Bayes estimator, by updating the probability dictionary with
#     the observed occurences of every sub-event in the dataset
#     l is the dirichlet weight (adjusted by cross-validation)
def train_naive_estimator(probability_dict,dataset,labelfile,continent_list,snp_list,snp_possible_values,l):
    # Create a list with all the labels of our training set
    with open(labelfile) as fin:
        label_list = fin.readlines()[0].strip('\n').split('\t')
    ### NEW LINE!! REMOVE IF SOMETHING GOES WRONG
    label_list.remove('')
    # Usage of probability dictionary: probability[snp][value][continent] (for a better understanding)
    with open(dataset) as fin:
        # Create an integer for keeping a track of how many lines have been readed on the file
        line_count = 0
        # Check if the first line of the training set contains a header (and ignore it in that case)
        for line in fin.readlines():
            if line.split('\t')[0] == 'ID':
                continue
            else:
                # Retrieve the SNP values counts of every individual
                values = line.strip('\r\n').split('\t')
                # Update the SNP count, given the continental origin of the individual
                #    and the SNP value observed
                for value_position in range(1,len(values)):
                    probability_dict[snp_list[value_position - 1]][values[value_position]][label_list[line_count]] += 1
                line_count += 1

    # Normalize the probability dictionary (normalize the counts)
    for continent in continent_list:
        for snp in snp_list:
            continent_count = label_list.count(continent)
            for value in snp_possible_values:
                probability_dict[snp][value][continent] = float(probability_dict[snp][value][continent] + l) / float(continent_count + l*len(snp_possible_values))
    # Return the probability estimations
    return probability_dict


# Naive Bayes Classifiers: estimates the probability of coming from a certain
#     continent given the trained probabilities and the prior probabilities
def naive_bayes_classifier(probability_dict,testset,outlabel,snp_list,prior_probability_dict):
    # Estimation dictionary. THe probability of every individual (KEY) being from a
    # specific continent will be appended to this dictionayr
    # (one VALUE list per individual)  [prob(cont=1),prob(cont=2),prob(cont=3),prob(cont=4),prob(cont=5)]
    estimation = dict()
    with open(testset) as fin:
        for line in fin.readlines():
            # Check if the first line of the training set contains a header (and ignore it in that case)
            if line.split('\t')[0] == 'ID':
                continue
            else:
                # Initialize list for the probabilities of being from a different continent
                # for the specific individual (Determined by the line number)
                estimation_individual = []
                for continent in range(1,6):
                    estimation_individual.append(prior_probability_dict[str(continent)])
                # Retrieve the SNP values counts of every individual
                values = line.strip('\r\n').split('\t')
                individual_ID = values[0]
                for value_position in range(1,len(values)):
                    for i in range(len(continent_list)):
                        estimation_individual[i] = estimation_individual[i] * probability_dict[snp_list[value_position - 1]][values[value_position]][continent_list[i]]
                estimation[individual_ID] = estimation_individual
    # Return the estimations dictionary
    return estimation

# Gives the most probable origin continent according to the estimated
#    probabilities from a naive bayes classifier
def argmax(estimations):
    estimated_continent = dict()
    # Look for the maximum probability in the estimations
    # (add 1 to the index because indices go from 0-4, whereas continent labels range from 1-5)
    for individual in estimations.keys():
        estimated_continent[individual] = estimations[individual].index(max(estimations[individual])) + 1
    # return the estimated continent number indentifier (1,2,3,4, or 5)
    return estimated_continent

# Counts the misclassifications from the estimations obtained by a
#    naive Bayes classifier
def count_misclassifications(estimated_continents,goldstandard):
    misclassified = 0
    goodclassified = 0
    possible_continents = ['EAS','SAS','EUR','AMR','AFR']

    real_continent = dict()
    with open(goldstandard) as fin:
        for line in fin.readlines():
            if line.split('\t')[0] == 'sample':
                continue
            else:
                individual = line.split('\t')[0]
                continent = line.split('\t')[2]
                # Sometimes we get continents as labels ('EUR','EAS',...)
                if continent in possible_continents:
                    real_continent[individual] = possible_continents.index(continent) + 1
                # Sometimes we get the continent number as labels ('1','2','3,'4','5')
                else:
                    real_continent[individual] = continent

    for individual in estimated_continents.keys():
        estimated = estimated_continents[individual]
        real_cont = real_continent[individual]
        if estimated != real_cont:
            misclassified += 1
        if estimated == real_cont:
            goodclassified += 1
    print 'misclassified ' + str(misclassified)
    print 'goodclassified ' + str(goodclassified)
    return misclassified, goodclassified

# Splits the original training set in k_value subsets for performing further crosvalidations
# Random k-fold
def generate_kfold_subsets(trainingset,k_value):
    # Append all the lines of the original dataset in "raw_dataset" list
    raw_dataset = []
    with open(trainingset) as fin:
        firstline = True
        for line in fin.readlines():
            if firstline:
                firstline = False
                continue
            else:
                raw_dataset.append(line)

    with open(tr_labels) as fin:
        trainigset_labels = fin.readlines()[0].strip('\n').split('\t')

    total_individuals = len(raw_dataset)
    # Calculate the size of every subset
    subset_size = float(total_individuals) / float(k_value)
    # Randomly assign the indices of the original dataset (raw_dataset) that
    # will be included in every subset
    random_assignments = random.sample(range(len(raw_dataset)),len(raw_dataset))


    # Split the dataset into k_value subsets and save every subset in the desired directory
    for k in range(k_value):
        new_subset = []
        new_labels = []
        # Here we can select to create the subsets randomly or by order of appearance in the training
        # set file. Uncomment the way that you want to do it:
        for index in range(int(k*subset_size),int(k*subset_size + subset_size)):
        #for index in random_assignments[int(k*subset_size):int(k*subset_size + subset_size)]:
            new_subset.append(raw_dataset[index])
            new_labels.append(trainigset_labels[index])
        f = open('Data/10fold_training_subsets/%sfold_subset%s.vcf'%(str(k_value),str(k+1)),'w')
        g = open('Data/10fold_training_subsets_LABELS/%sfold_subset%s_LABELS.dat'%(str(k_value),str(k+1)),'w')
        for line in new_subset:
            f.write('%s'%(str(line)))
        for label in new_labels:
            g.write('%s\t'%(str(label)))
        f.close()
        g.close()
    print '%s fold subsets generated, and saved in directory Data/%sfold_training_subsets'%(str(k_value),(str(k_value)))


# Calculates the cross-validation error give specific subsets of the dataset for
# a specific parameter
def cross_validation(kfold,kfolddirectory,l,snp_possible_values,snp_list,continent_list,goldstandard):
    files = []
    files_labels = []
    for k in range(1,11):
        files.append('10fold_subset%s.vcf'%(str(k)))
        files_labels.append('10fold_subset%s_LABELS.dat'%(str(k)))

    misclassifications_CV, goodclassifications_CV = [], []
    # de momento sin prior probabilities, luego ya las implementare
    for k in range(kfold):
        filesS = files[0:k] + files[k+1:kfold-1]
        filesS_labels = files_labels[0:k] + files_labels[k+1:kfold-1]
        cv_file = kfolddirectory + files[k]
        cv_label_file = kfolddirectory.strip('/') + '_LABELS/' + files_labels[k]

        # Create a temporary file which concatenates all the selected subsets
        g = open('temp.txt','w')
        for calfile in files:
            with open(kfolddirectory + calfile) as fin:
                for line in fin.readlines():
                    if line.split('\t')[0] != 'ID':
                        g.write('%s'%(line))
        g.close()
        # Create a temporary file which concatenates all the labels
        h = open('temp_labels.txt','w')
        for cal_labelfile in files_labels:
            with open('Data/10fold_training_subsets_LABELS/' + cal_labelfile) as fin:
                for line in fin.readlines():
                    h.write('%s'%(line.strip('\n')))
        h.close()

        filein = 'temp.txt'
        filein_labels = 'temp_labels.txt'

        misclassified_CV, goodclassified_CV, estimations = bayes_classifier(l,filein_labels,filein,cv_file,snp_possible_values,snp_list,goldstandard,continent_list)
        misclassifications_CV.append(misclassified_CV), goodclassifications_CV.append(goodclassified_CV)
    return sum(misclassifications_CV), sum(goodclassifications_CV)

# Performs cross-validation for a set range of parameters l_initial:l_increment:l_final
#   and outputs the results and the resulting plots
def multi_cross_validation(outfile,outimage,l_initial,l_final,l_increment,snp_list):
    file_cross = open(outfile,'w')
    file_cross.write('#l(Dirichletweight)\tMisclassified_CV\tGoodclassified_CV\n')
    # Dirichet weight for our model training
    misclassifications_CV, misclassifications_calibration = [], []
    l_values = []
    l = l_initial
    print 'Starting Cross-validation\n'
    while l<l_final:
        misclassified_CV, goodclassified_CV = cross_validation(10,'Data/10fold_training_subsets/',l,snp_possible_values,snp_list,continent_list,goldstandard)
        print 'l is ' + str(l)
        print misclassified_CV, goodclassified_CV
        misclassifications_CV.append(misclassified_CV)
        l_values.append(l)
        file_cross.write('%s\t%s\t%s\n'%(str(l),str(misclassified_CV),str(goodclassified_CV)))
        l = l + l_increment

    plt.figure()
    plt.xlabel('Dirichlet weight'),plt.ylabel('Number of misclassifications')
    plt.plot(l_values,misclassifications_CV)
    plt.title('10-fold cross validation (Naive Bayes classifier)')
    plt.savefig(outimage)
    file_cross.close()
    print 'Cross validation finished. Results succesfully saved in %s\n'%(outfile)

# Naive bayes classifier. Trains a classifier based on the training set observations (tr_data)
# and its labels (tr_labels) and estimates the labels of the test set (test_data)
def bayes_classifier(l,tr_labels,tr_data,test_data,snp_possible_values,snp_list,goldstandard,continent_list):
    # Initialize the probability dictionary
    probability = create_probability_dictionary(snp_list,snp_possible_values,continent_list)
    # Compute the prior probabilities
    prior_probability_dict = prior_probability(tr_labels,l)
    # Train the naive bayes model
    probability = train_naive_estimator(probability,tr_data,tr_labels,continent_list,snp_list,snp_possible_values,l)
    # Retrieve the estimations from the naive Bayes classifier for the TEST set
    estimations = naive_bayes_classifier(probability,test_data,'test',snp_list,prior_probability_dict)
    estimated_continents = argmax(estimations)
    misclassified, goodclassified = count_misclassifications(estimated_continents,goldstandard)
    real_continent_dict = real_continent(goldstandard,continent_list)
    f = open('Results/bayes_TESTSET_predictions.dat','w')
    f.write('#Naive Bayes classifier\n#Individual\tPredicted_continent\tReal_continent\n')
    for ind in estimated_continents.keys():
        if estimated_continents[ind] == 1:
            continent = 'EAS'
        if estimated_continents[ind] == 2:
            continent = 'SAS'
        if estimated_continents[ind] == 3:
            continent = 'EUR'
        if estimated_continents[ind] == 4:
            continent = 'AMR'
        if estimated_continents[ind] == 5:
            continent = 'AFR'
        f.write('%s\t%s\t%s\n'%(ind,continent,real_continent_dict[ind]))
    f.close()
    return misclassified, goodclassified, estimations

# Retrieves a dictionary with the real continent (gold standard) of every individual
def real_continent(goldstandard,possible_continents):
    real_continent = dict()
    with open(goldstandard) as fin:
        for line in fin.readlines():
            if line.split('\t')[0] == 'sample':
                continue
            else:
                individual = line.split('\t')[0]
                continent = line.split('\t')[2]
                # Sometimes we get continents as labels ('EUR','EAS',...)
                if continent in possible_continents:
                    real_continent[individual] = possible_continents.index(continent) + 1
                # Sometimes we get the continent number as labels ('1','2','3,'4','5')
                else:
                    real_continent[individual] = continent
    return real_continent

# FUNCTION: integrate
# ARGUMENTS: x --> list containing the x-coordinates of a points collection
#            y --> list containing the y-coordinates of a points collection
# DESCRIPTION: This function integrates the area under a curve, based on the trapezoidal rule numerical integration method.
def integrate(x,y):
    #########################
    ### START CODING HERE ###
    #########################
    # As the x values are not equally spaced between them, the numerical integration of the ROC plot (AUC <--> Area Under the Curve) will
    # be performed following the trapezoidal rule for a non-uniformal grid (Documentation: https://en.wikipedia.org/wiki/Trapezoidal_rule)
    integral = 0 # Assign a variable for the sum/integration
    first = True # Assign a boolean operator which will enable to perform a different operation in the 1st iteration of the following loop
    for (point_x, point_y) in zip(x,y):
        # In the first iteration, assign to the "previous_xcoordinate" and "previous_ycoordinate" variables the x and y coordinates, respectively, of the 1st point
        if first:
            previous_ycoordinate = point_y
            previous_xcoordinate = point_x
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
        first = False
    # Lastly, the previou sum will be divided by 2, according to the trapezoidal rule equation
    integral = float(integral) / float(2)
    #########################
    ###  END CODING HERE  ###
    #########################
    return integral

# Creates a ROC plot from the probability estimations (one ROC plot per possible classification)
def roc_plot(estimations,goldstandard,possible_continents):
    real_continent_dict = real_continent(goldstandard,possible_continents)
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
        h = open('Results/ROCplot_bayesclassifier_%s.dat'%(continent),'w')
        h.write('## ROC PLOT RESULTS FROM NAIVE BAYES CLASSIFIER, continent = %s\n'%(continent))
        h.write('## TruePositiveRate FalsePositiveRate\n')
        TP, FP, TN, FN = 0, 0, 0, 0
        TPR, FPR = [0],[0]
        h.write('%s\t%s\n'%(str(0),str(0)))
        for individual in estimations.keys():
            if real_continent_dict[individual] == continent:
                FN += 1
            else:
                TN += 1

        estimations_test_new = dict()
        for individual in estimations.keys():
            estimations_test_new[individual] = estimations[individual][i]

        first = True
        for prob_value in sorted(estimations_test_new.items(), key=itemgetter(1)):
            if not first:
                TPR.append(TP/float(TP+FN))
                FPR.append(FP/float(FP+TN))
                h.write('%s\t%s\n'%(str(TP/float(TP+FN)),str(FP/float(FP+TN))))
            first = False
            if real_continent_dict[prob_value[0]] == continent:
                TP += 1
                FN -= 1
            else:
                FP += 1
                TN -= 1
        TPR.append(TP/float(TP+FN))
        FPR.append(FP/float(FP+TN))
        h.write('%s\t%s\n'%(str(TP/float(TP+FN)),str(FP/float(FP+TN))))
        continent_list.append(continent)
        TPR_continent.append(TPR), FPR_continent.append(FPR)
        h.close()
    AUC = []
    plt.figure(figsize=(7,4))
    plt.xlim(0,1), plt.ylim(0,1)
    for cont in range(len(TPR_continent)):
        AUC.append(integrate(TPR_continent[cont],FPR_continent[cont]))
        label_cont = continent_list[cont] + '. AUC = ' + str(round(integrate(TPR_continent[cont],FPR_continent[cont]),3))
        plt.plot(TPR_continent[cont],FPR_continent[cont],label=label_cont)
    plt.plot([0,1],[0,1],'k:',label='Random method')
    plt.legend(title='Naive Bayes',fontsize=10,frameon=False,loc=4)
    plt.ylabel('True positive rate'),plt.xlabel('False positive rate')
    plt.tight_layout()
    plt.savefig('ROCplot_bayes.png')
    mean = 0
    for auc in AUC:
        print auc
        mean += auc
    mean = float(mean) / len(AUC)
    print 'Mean AUC   ' ,mean

# Creates a confussion matrix
def create_confussion_matrix(resultsfile,outputlabel):
    actual_labels, predicted_labels = [],[]
    with open(resultsfile) as fin:
        for line in fin.readlines():
            # Don't read the commented lines
            if line[0] == '#':
                continue
            else:
                actual_labels.append(line.strip('\n').split('\t')[2])
                predicted_labels.append(line.strip('\n').split('\t')[1])
    cnf_matrix = confusion_matrix(actual_labels, predicted_labels)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, normalize=False, title="Confusion matrix Naive Bayes classifier")
    plt.tight_layout()
    plt.savefig('Results/%s.png'%(outputlabel))

# Function created by Stavros Giannoukakos
def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix',cmap=plt.cm.Paired):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Defining the classes
    classes = ["East Asia", "South Asia", "Europe", "America", "Africa"]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return

# Performs a permutation test by building models with scrambled labels
# And returns an histogram with the misclassifications of the random models
def permutation_test(l,snp_list,real_misclassifications,scrambled_labels_directory,outfile):
    real_misclassifications
    misclassified_list = []
    f = open(outfile,'w')
    for random_label_file in os.listdir(scrambled_labels_directory):
        misclassified, goodclassified, estimations = bayes_classifier(l,scrambled_labels_directory + '/'+random_label_file,tr_data,test_data,snp_possible_values,snp_list,goldstandard,continent_list)
        misclassified_list.append(misclassified)
        f.write('%s\n'%(str(misclassified)))
    print '%s file written'%(outfile)
    plt.figure()
    plt.hist(misclassified_list,25, facecolor='green', alpha=0.75,label='Permutation approach models')
    plt.scatter(real_misclassifications,1,s=100,label='Real labels model')
    plt.ylim(0), plt.xlim(0,500)
    plt.legend(frameon=False,loc=2,fontsize=8)
    plt.xlabel('Misclassifications'),plt.ylabel('Counts'),plt.title('Permutation approach Bayes classifier')
    plt.savefig('randomisation_test.png')
    f.close()

# Main function
def main():
    # Retrieve the list of SNP's in the training set
    snp_list = read_SNP(tr_data)

    # Generate k-fold subsets of the original trainingset (if desired)
    generate_subset = False
    if generate_subset:
        generate_kfold_subsets(tr_data,10)

    # Set the dirichlet weight (in this case, it was set by cross-validation)
    l = -0.8
    # Train a naive bayes classifier and return teh misclassifications/goodclassifications of the test set
    misclassified, goodclassified, estimations = bayes_classifier(l,tr_labels,tr_data,test_data,snp_possible_values,snp_list,goldstandard,continent_list)

    # Plot the confussion matrix from the output file (if desired)
    perform_confusion_matrix = False
    if perform_confusion_matrix:
        create_confussion_matrix('Results/bayes_TESTSET_predictions.dat','confussionmatrix_testset_naivebayes')

    # Perform a permutation test (if desired)
    do_permutation_test = True
    if do_permutation_test:
        permutation_test(l,snp_list,misclassified,'Data/labels/training_random_labels','Results/bayes_permutation_approach.dat')

    # Perform a ROC plot (if desired)
    rocplot = False
    if rocplot:
        misclassified, goodclassified, estimations = bayes_classifier(l,tr_labels,tr_data,test_data,snp_possible_values,snp_list,goldstandard,continent_list)
        roc_plot(estimations,goldstandard,continent_list)

    # Perform cross-validation (if desired)
    crossvalidation = True
    if crossvalidation:
        multi_cross_validation('Data/naivebayes_crossavlid10fold_b.dat','Results/10fold_crossvalidation_b.png',-100,100,0.1,snp_list)

if __name__=="__main__":
    main()
