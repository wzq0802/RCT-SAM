# Foundation Model-Based Rock CT Image Segmentation and Interpretable Mechanical Property Prediction from Learned Features

This repository contains the computer codes associated with the manuscript: Foundation Model-Based Rock CT Image Segmentation and Interpretable Mechanical Property Prediction from Learned Features.

## PROGRAMS    

·RCT-SAM.py: Implements the proposed RCT-SAM model and the fine-tuning procedure used in the manuscript.
·SAM.py: Contains the basic functions and supporting modules required by the SAM-based segmentation workflow.
·pred.py: Performs prediction on new rock CT images using the trained RCT-SAM model and outputs the segmentation results.


## EXAMPLE
Example

The repository contains all the codes required to reproduce the results presented in the manuscript, organized into different scripts. Example input images and output results are provided to demonstrate the workflow.
Rock CT images used in the example are included in the dataset folder. These images are used to illustrate the segmentation performance of the proposed RCT-SAM model.
RCT-SAM.py implements the proposed model and fine-tuning strategy described in the manuscript. It is used to train and adapt the SAM model for digital rock image segmentation.
SAM.py contains the core functions and utility modules required for the SAM-based segmentation framework.
pred.py is used to perform inference on new rock CT images and generate segmentation results.
Example prediction results are saved in the Figures folder in .png format. These outputs can be used to visually compare segmentation performance.
Users can run the example by executing the prediction script on the provided sample images. The results will be automatically saved in the corresponding output folder.
