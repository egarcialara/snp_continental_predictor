#  Predicting continental origin 
#  from genomic information on human Chromosome 1 
#### Is Machine Learning racist?
-------------------------------------------------------------------------------
Repository for the Machine Learning course (VU University, 2017)

## Description
A continental predictor trained with Single-nucleotide polymorphism (SNP) data based on a Machine Learning approach.  
The repository contains the source code and the full report.

The trained algorithms include:
* Random Forest (RF)
* Naive Bayes classifier
* Support Vector Machine
* Ensemble (stacking approach)

The feature selection incorporates an approach based on Information Gain quantification.

The model evaluation is based on a Receiver Operating Characteristic plot (ROC) and a confusion matrix. 

The statistical significance of the models was assessed with a permutation test. 

## Dataset
SNP genotype from [The 1000 Genomes project](http://www.internationalgenome.org/). 

## Dependencies
* Numpy
* Pandas

## Authors
* Elena Garcia Lara @[egarcialara](https://github.com/egarcialara)
* Asli Kucukosmanoglu
* Stravros P. Giannoukakos
* Alberto Gil Jimenez @[tropicalberto](https://github.com/tropicalberto)

_All authors contributed equally to this work._
