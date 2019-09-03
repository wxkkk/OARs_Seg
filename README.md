# OARs_Seg
thoracic organs segmentation using Keras

#### `data_read_preprocess.py` contains the following functions:<br>
*    CT file reading.<br>
*   CT DICOM file pre-processing, normalization HU.<br>
*   RT Structure file is read and saved as label map.<br>
*   Store all data formats in the NumPy format according to the original directory structure.<br> 
#### `u-net_model.ipynb` contains the following functions:<br>
* Use Keras to build a U-net model.<br>
* Read the training set files in Google Drive and train the model.<br>
* The output training process visualizes and saves the predicted model.<br>
* Load the model prediction validation set and output the predicted masks.<br>
#### `predicted_model_B.png` shows the prediction masks of Model B on the validation set
