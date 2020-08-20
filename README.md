# Vehicle-classifier
Tensorflow based classifier that can able to predict what category an image of a vehicle belongs too. The project was used in [kaggle competition](https://www.kaggle.com/c/vehicle) and reached the accurasy of 91% resulting in the 2nd place.

# Prerequistes
List of the software used the classifier (Older versions might work as well, but they were not tested):


## Python packages

* tensorflow(>= 2.0.0)
* efficientnet(>=1.0.0)
* opencv-python(>= 4.4.0.42)

# How to use

    python3 train.py -h

    usage: t.py [-h] [-mp MODEL_PATH] [-dp DATA_PATH] [-ps PIC_SIZE]
                [-bs BATCH_SIZE] [-cp CH_PATH] [-sf SAVE_FREQ]
                [-sm] [-lw] [-v VERBOSE] [-e EPOCHS] 


    optional arguments:
      -h, --help                                  show this help message and exit
      -mp MODEL_PATH, --model_path MODEL_PATH     Path to an h5 model file
      -dp DATA_PATH, --data_path DATA_PATH        Path to a folder with image files
      -ps PIC_SIZE, --pic_size PIC_SIZE           Square side size to rescale the pictures. Default (300)
      -bs BATCH_SIZE, --batch_size BATCH_SIZE     Size of the batch to load pictures. Default (30)
      -cp CH_PATH, --checkpoint_path CH_PATH      Path to save/load checkpoints
      -sf SAVE_FREQ, --save_frequency SAVE_FREQ   Checkpoint save frequency in batches default (0)
      -sm, --save_model                           Save model file after the training
      -lw, --load_weights                         Whether to load weights from a checkpoint
      -v VERBOSE, --verbose VERBOSE               Keras verbosity level (0-2)
      -e EPOCHS, --epochs EPOCHS                  Number of epochs to train
      
  
