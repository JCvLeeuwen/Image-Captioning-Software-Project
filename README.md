##Image Captioner Software Project##

This Repository contains 4 seperate, stand-alone, pieces of software.
To run the software please follow the instructions below, as well as the instructions in the READMEs of each individual component. 
A full working version of this project, can be found here: https://drive.google.com/drive/folders/1Wv7hxvEx8eUhc44ErGjdHNJafIjYTLGb?usp=drive_link
This Google Drive contains all necessary files to run and test each component.

As GitHub places significant memory limitations on its repositories, and this project requires larger datasets and files to be demonstrated properly, I see myself forced to share the missing data through a secondary channel.
To use the software as provided on this repository please do the following: 

##Image captioner##
image_captioner contains the files necessary to train an image captioning model and generate captions for any desired image.

Please take the folder from the following Google Drive (called data), and place it in image_captioner. 
https://drive.google.com/drive/folders/1--QcgwOxOe2ecQuuy_z_WkX1Sog8x2g1?usp=drive_link
From here please refer to the README found in image_captioner.

##Metric Tester##
metric_tester contains the implementation of several metrics used to evaluate image captioning models. This implementation was used to get a feel for the strengths and weaknesses of the metrics in question and can be tested as it is uploaded here. 

##URL Dataset Loader##
url_dataset_loader contains the code used to extract image-caption pairs from the original Train_GCC-training.tsv Conceptual Captions dataset. To run this software, please place Train_GCC-training.tsv, which can be found here:
https://drive.google.com/file/d/1oauSFU7H52kVnW3lnVbYrV0OV4J7epVK/view?usp=drive_link
in the url_dataset_loader folder. 
From here please refer to the README found in url_dataset_loader.

##Colab Notebook version## 
colab_notebook_version_of_image_captioner contains the notebook used to train the final model. image_captioner is the "production code" counterpart to this notebook and has the same training functionality, but lacks the caption generation functionality. 
To use this notebook, please open Train_FineTune.ipynb in Google Colab and place the ImageCaptionColab folder in a Google Drive. Add the loadedimages.zip file found here:
https://drive.google.com/file/d/1lAs1JPGhPDcuysM2rpf8zMPgIbksNo3L/view?usp=drive_link 
to the ImageCaptionColab folder.
From here please refer to the instructions found in Train_FineTune.ipynb.
