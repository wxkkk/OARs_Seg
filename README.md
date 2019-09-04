# OARs_Seg
thoracic organs segmentation using Keras

#### `data_read_preprocess.py` contains the following functions:<br>
*    CT file reading.<br>
*   CT DICOM file pre-processing, normalization HU.<br>
*   RT Structure file is read and saved as label map.<br>
*   Store all data in NumPy format according to the original directory structure.<br> 
#### `u-net_model.ipynb` contains the following functions:<br>
* Use Keras to build a U-net model.<br>
* Read the training set files in Google Drive<br>
* Train the network and derive the model.<br>
* The visualization of training process.<br>
#### `model_evaluation.ipynb` contains the following functions:<br>
* Load the model and predict the validation set images.<br>
* Output the predicted masks.<br>
* Output MIoU on validation set.<br>
#### `Model_B` is one of the well-performing models trained by U-Net.
#### `predicted_model_B.png` shows the prediction masks of Model B on the validation set
