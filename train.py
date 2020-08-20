	import tensorflow as tf
import numpy as np
from data_load import get_sequence
from multiprocessing import cpu_count
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight



def get_model(model_path):
    #from utils.custom_objects import efficientnet_objects
    #obj = efficientnet_objects() if "eff" in model_path.lower() else None
    model = load_model(model_path) #, custom_objects=obj)
    return model



def train(data_path="", data_seq=None, model_path="model.h5", model=None, pic_size=(300,300), 
                verbose=1, save=False, checkpoint_path="train.ckpt", save_freq=50, load_weights=True,
                stratify=False, epochs=3, batch_size=10):
                

    training_seq, validation_seq = data_seq or get_sequence(data_path, batch_size=batch_size, picture_size=pic_size, validation_split=0.2)


    model = model or get_model(model_path)
    

    if not model:
        raise RuntimeError("Either keras model or path to load it should be provided")
    

    callbacks = []
    if checkpoint_path:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=verbose,
                                                        save_freq=batch_size * save_freq,
                                                        monitor="accuracy")
        callbacks.append(checkpoint)

    
    if load_weights:
        model.load_weights(checkpoint_path)
        

    model.fit(training_seq, validation_data=validation_seq,
              epochs=epochs,
              callbacks=callbacks,
              workers=cpu_count())

    if save:
        model.save(model_path)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-mp", "--model_path", dest="model_path",
                        help="Path to an h5 model file")

    parser.add_argument("-dp", "--data_path", dest="data_path",
                        help="Path to a folder with image files")

    parser.add_argument("-ps", "--pic_size", dest="pic_size", default=300,
                        help="Square side size to rescale the pictures. Default (300) ")

    parser.add_argument("-bs", "--batch_size", dest="batch_size", default=30,
                        help="Size of the batch to load pictures. Default (30) ")


    parser.add_argument("-cp", "--checkpoint_path", dest="ch_path",
                        help="Path to save/load checkpoints")


    parser.add_argument("-sf", "--save_frequency", dest="save_freq",
                        help="Checkpoint save frequency in batches default (0) ")


    parser.add_argument("-sm", "--save_model", action='store_true', dest="save_model",
                        help="Save model file after the training ")


    parser.add_argument("-lw", "--load_weights", action='store_true', dest="load_weights",
                        help="Whether to load weights from a checkpoint")


    parser.add_argument("-v", "--verbose", dest="verbose", default=1,
                        help="Keras verbosity level (0-2)")


    parser.add_argument("-e", "--epochs", dest="epochs", default=1,
                        help="Keras verbosity level (0-2)")


    args = parser.parse_args()

    if not args.model_path or not args.data_path:
        parser.print_help()
    else:
        train(data_path=args.data_path,
              model_path=args.model_path,
              pic_size=(int(args.pic_size), int(args.pic_size)),
              checkpoint_path=args.ch_path,
              save=args.save_model,
              save_freq=args.save_freq,
              load_weights=args.load_weights,
              epochs=int(args.epochs),
              batch_size=int(args.batch_size),
              verbose=int(args.verbose))
