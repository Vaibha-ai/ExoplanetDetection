import argparse
import os
import tensorflow as tf
from astronet import models
from astronet.util import estimator_util
from tf_util import config_util
from tf_util import configdict
from tf_util import estimator_runner
import pickle

parser = argparse.ArgumentParser()

parser.add_argument(
    "--uploaded_folder",
    type=str,
    help="Path to the uploaded folder containing the TFRecord files.")

args = parser.parse_args()

# Define parameters as variables with values initialized within the code itself
MODEL_NAME = "AstroCNNModel"
CONFIG_NAME = "local_global"
DEFAULT_FOLDER = "astronet/testing_tfrecords"
DEFAULT_TRAIN_FILES = os.path.join(DEFAULT_FOLDER, "train*")
DEFAULT_EVAL_FILES = os.path.join(DEFAULT_FOLDER, "val*")
MODEL_DIR = "astronet/model_modified"
PICKLE_FILE_PATH = "astronet/model_modified/test.pkl"
TRAIN_STEPS = 10
SHUFFLE_BUFFER_SIZE = 15000


if args.uploaded_folder:
    TRAIN_FILES = os.path.join(args.uploaded_folder, "train*")
    EVAL_FILES = os.path.join(args.uploaded_folder, "val*")
else:
    TRAIN_FILES = DEFAULT_TRAIN_FILES
    EVAL_FILES = DEFAULT_EVAL_FILES


def main(_):
    model_class = models.get_model_class(MODEL_NAME)

    # Look up the model configuration.
    config = (
        models.get_model_config(MODEL_NAME, CONFIG_NAME)
        if CONFIG_NAME else config_util.parse_json(CONFIG_JSON))

    config = configdict.ConfigDict(config)
    config_util.log_and_save_config(config, MODEL_DIR)

    # Create the estimator.
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1)
    estimator = estimator_util.create_estimator(model_class, config.hparams, run_config, MODEL_DIR)

    # Create an input function that reads the training dataset. We iterate through
    # the dataset once at a time if we are alternating with evaluation, otherwise
    # we iterate infinitely.
    train_input_fn = estimator_util.create_input_fn(
        file_pattern=TRAIN_FILES,
        input_config=config.inputs,
        mode=tf.estimator.ModeKeys.TRAIN,
        shuffle_values_buffer=SHUFFLE_BUFFER_SIZE,
        repeat=1 if EVAL_FILES else None)

    if not EVAL_FILES:
        estimator.train(train_input_fn, max_steps=TRAIN_STEPS)
    else:
        eval_input_fn = estimator_util.create_input_fn(
            file_pattern=EVAL_FILES,
            input_config=config.inputs,
            mode=tf.estimator.ModeKeys.EVAL)
        eval_args = [{"name": "val", "input_fn": eval_input_fn}]

        for _ in estimator_runner.continuous_train_and_eval(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_args=eval_args,
            train_steps=TRAIN_STEPS):
            # continuous_train_and_eval() yields evaluation metrics after each
            # training epoch. We don't do anything here.
            pass
        # Save the trained model using pickle
    with open(PICKLE_FILE_PATH, 'wb') as f:
        pickle.dump(estimator, f)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
