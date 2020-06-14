Explanations for Learning-to-Rank models
=========================================
This repository is a toolbox conbines with several kinds of methods to explain learning to rank models, the paper can be find here: [paper](https://arxiv.org/abs/2004.13972).
they extract small subset of input features as an explaination to the ranking decision. Specifically, they can be classified into the several categories：

- Shapk methods
- Validity methods
- Completeness methods
- Alpha methods

Simple Introduction
-----------------------------
In simple terms, the shapk methods use shapley value to explore eventful features, we can select features through analysing the top ranked document or top 5 documents,
 for further details, please refer to [shap](https://github.com/slundberg/shap). Validity methods imply that the returned smaller set of features are sufficient to reconstruct the
original ranking output by the model when using all the features. Completeness methods remove or alter the explanation features from the input will change the output function or ranking considerably. The alpha way makes a trade off between validity and completeness with a value alpha. For example, if alpha is 0.1 , then we pay 90% attention on validity score and 
10% on completeness score. In addition, the ranking tool we used is [Ranklib](https://sourceforge.net/p/lemur/wiki/RankLib/).


Project Structure
-----------------------------
In this project there are the following folders:
* **shapk_and_random**
  * **baseline.ipynb** randomly choose features as baseline for dataset MQ2008
  * **baseline_MLSR.ipynb** randomly choose features as baseline for dataset MSLR-WEB10K
  * **matrix_shapk.ipynb** shapk methods for MQ2008, k can be 1 or 5
  * **matrix_shapk_MSLR.ipynb** shapk methods for MSLR-WEB10K, k can be 1 or 5
  * **matrix_shapk_complement.ipynb** this script used to get the result of the complement(validity) of the shapk selected features
* **validity** in this folder there are 5 kinds of validity methods, respectively, 
  * **normal_validity.ipynb** the most normal validity approach
  * **validity_beamsearch.ipynb** based on the above method, this method do beamsearch at the first feature selection step to optimize Kendall rank correlation coefficient
  * **validity_beamsearch_only_positive.ipynb** the difference here is that we stop selecting features when there is no positive cell in the score matrix
  * **validity_beamsearch_threshold.ipynb** in this way we delect document pairs when the cell in over average value of this row
  * **validity_beamsearch_pure_greedy_weight.ipynb** without delecting any document pairs, the stop condition here is the number of positive cells has decreased 
* **completeness** in this folder there are 5 kinds of completeness methods, respectively, 
  * **normal_completeness.ipynb** the most normal completeness approach
  * **completeness_beamsearch.ipynb** based on the above method, this method do beamsearch at the first feature selection step to optimize Kendall rank correlation coefficient
  * **completeness_beamsearch_only_positive.ipynb** the difference here is that we stop selecting features when there is no positive cell in the score matrix
  * **completeness_beamsearch_threshold.ipynb** in this way we delect document pairs when the cell in over average value of this row
  * **completeness_beamsearch_pure_greedy_weight.ipynb** without delecting any document pairs, the stop condition here is the number of positive cells has decreased   
* **alpha**  
  * **normal_alpha.ipynb** the most normal alpha approach
  * **alpha_beamsearch.ipynb.ipynb** based on the above method, this method do beamsearch at the first feature selection step to optimize Kendall rank correlation coefficient
  * **alpha_beamsearch_new.ipynb** define a new way here to make trade off of validity and completeness.
* **utils** utils used in scripts
  * **explainer_tools.py** functions for explaination like select pairs or get best cover
  * **readdata.py** functions used to read datasets
  * **rerank.py** process log files in batches and get the average of each indicator
  * **separate_set.py** used to separate the dataset according to query id
* **evaluate** tools for evaluation
  * **evaluate_x_features.ipynb** get results with different numbers of features like 3 or 7 on the log file
  * **f1.ipynb** calculate F1 value between kendall taus of validity and completness
  * **idea_onlytau_completeness_MQ2008.ipynb** get 3 idea features of completness of dataset MQ2008
  * **idea_onlytau_completeness_MSLR.ipynb** get 3 idea features of completness of dataset MSLR-WEB10K
  * **idea_onlytau_validity_MQ2008.ipynb** get 3 idea features of validity of dataset MQ2008
  * **idea_onlytau_validity_MSLR.ipynb** get 3 idea features of validity of dataset MSLR-WEB10K
  * **IOU.ipynb** get the iou value between features of validity and completness
* **old** old scripts
* **shap** shap repository from slundberg's github

In this repository, the following folders have not been uploaded:
* **logs** log files, generally, the naming rules is like: dataname_methodname_number_of_features_modelname.txt. Except the results(delta_NDCG,kendall_tau,ratio_NDCG), here also save the file of the rank lists before and after we modify the features, the matrix files  
* **temp_file** a place to save temporary files from Ranklib
* **resultfile** save result files temporarily just to facilitate to fill in the result table
* **ideafeatures** save the results of idea features
* **MQ2008** [MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0) dataset
* **MSLR-WEB10K** [MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) dataset
* **model** models of MQ2008 trained by Ranklib
* **MSLR-WEB10K_model** models of MSLR-WEB10K trained by Ranklib

