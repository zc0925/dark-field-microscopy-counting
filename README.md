# dark-field-microscopy-counting
# dark-field-microscopy-counting
The training code for this program is partially referenced in the lxztju documentation. The link is below: https://github.com/lxztju


Train:

Our project contains the following serveral steps:
1. preprocess--extract the patches from the raw data (thresholding and cropping)
2. reshape and save the patches
3. train the model on the training dataset (following partition of the dataset)--train.py
The trained model can be saved in the ./trained-models/dense169_xxx.pth

Test and counting:
1. preprocess--extract the patches from the raw data (thresholding and cropping)
2. reshape and save the patches
3. predict the patches (using trained model) and do counting

All the above codes can be found in the pipeline.py. 


