# MDSN-WCE

The PyTorch code for the submitted paper entitled "Medical Decision Support Network for Wireless Capsule Endoscopy Diagnosis". 

The specific code details will be uploaded here shortly after undergoing manuscript review and our code review.

Follow these steps

1. Upload zip-compressed datasets to the data directory and unzip them.
2. Use data_split.ipynb in the data directory to automatically divide into the training-validation-test.
3. Use mean_std_classweight.ipynb in the capsule_model directory to obtain the mean and std of the dataset, and the weights needed for weighted cross-entropy.
4. After confirming the parameters, run the .ipynb code for the corresponding dataset in the capsule_model directory, where the dataset name is specified.
5. The log files of the learning process are output to the terminal and to the log folder of the capsule_model directory, and the trained models are saved to the model folder of the capsule_model directory.
