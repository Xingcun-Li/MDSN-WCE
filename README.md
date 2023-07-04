# MDSN-WCE

## The PyTorch code for the submitted paper entitled "Medical Decision Support Network for Wireless Capsule Endoscopy Diagnosis". 

The specific code details will be uploaded here shortly after undergoing manuscript review and our code review.

## Follow these steps to run the code:

1. Upload zip-compressed datasets to the data directory and unzip them.
2. Use data_split.ipynb in the data directory to automatically divide into the training-validation-test.
3. Use mean_std_classweight.ipynb in the capsule_model directory to obtain the mean and std of the dataset, and the weights needed for weighted cross-entropy.
4. After confirming the parameters, run the .ipynb code for the corresponding dataset in the capsule_model directory, where the dataset name is specified.
5. The log files of the learning process are output to the terminal and to the log folder, and the trained models are saved to the model folder, both under the capsule_model directory.
6. You can load the trained model using plot_cm, plot_heatmap, plot_tSNE_{dataset_name} to plot confusion matrix, heatmap, tSNE representation visualization, respectively.

## Public WCE Datasets:
Kvasir-Capsule: https://datasets.simula.no/kvasir-capsule/  
CAD-CAP: Please contact the authors of the paper  
KID: https://mdss.uth.gr/datasets/endoscopy/kid/  

Kvasir-Capsule can be downloaded directly at the link. For CAD-CAP and KID datasets, please contact the authors or administrators to obtain access to these datasets.
## If you use the code for your research, please kindly cite our paper as follows in BibTeX:
>Not available now.
