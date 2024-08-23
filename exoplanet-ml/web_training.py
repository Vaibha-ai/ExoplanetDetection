from flask import Flask, render_template, request, jsonify
import argparse
import sys
import tensorflow as tf
import operator

app = Flask(__name__)

class AstroModel(object):
    def __init__(self, features, labels, hparams, mode):
        """Basic setup."""
        valid_modes = [
            tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT
        ]
        if mode not in valid_modes:
            raise ValueError("Expected mode in {}. Got: {}".format(valid_modes, mode))

        self.hparams = hparams
        self.mode = mode
        self.time_series_features = features.get("time_series_features", {})
        self.aux_features = features.get("aux_features", {})
        self.labels = labels
        self.weights = features.get("weights")
        self.is_training = None
        self.global_step = None
        self.time_series_hidden_layers = {}
        self.aux_hidden_layers = {}
        self.pre_logits_concat = None
        self.logits = None
        self.predictions = None
        self.batch_losses = None
        self.total_loss = None

    def build_time_series_hidden_layers(self):
        """Builds hidden layers for the time series features."""
        self.time_series_hidden_layers = self.time_series_features

    def build_aux_hidden_layers(self):
        """Builds hidden layers for the auxiliary features."""
        self.aux_hidden_layers = self.aux_features

    def build_logits(self):
        """Builds the model logits."""
        time_series_hidden_layers = sorted(
            self.time_series_hidden_layers.items(), key=operator.itemgetter(0))
        aux_hidden_layers = sorted(
            self.aux_hidden_layers.items(), key=operator.itemgetter(0))

        hidden_layers = time_series_hidden_layers + aux_hidden_layers
        if not hidden_layers:
            raise ValueError("At least one time series hidden layer or auxiliary "
                            "hidden layer is required.")

        if len(hidden_layers) == 1:
            self.pre_logits_concat = hidden_layers[0][1]
        else:
            self.pre_logits_concat = tf.concat([layer[1] for layer in hidden_layers],
                                            axis=1,
                                            name="pre_logits_concat")

        net = self.pre_logits_concat
        with tf.name_scope("pre_logits_hidden"):
            for i in range(self.hparams.num_pre_logits_hidden_layers):
                dense_op = tf.keras.layers.Dense(
                    units=self.hparams.pre_logits_hidden_layer_size,
                    activation=tf.nn.relu,
                    name="fully_connected_{}".format(i + 1))
                net = dense_op(net)

                if self.hparams.pre_logits_dropout_rate > 0:
                    dropout_op = tf.keras.layers.Dropout(
                        self.hparams.pre_logits_dropout_rate)
                    net = dropout_op(net, training=self.is_training)

            tf.identity(net, "final")

        dense_op = tf.keras.layers.Dense(
            units=self.hparams.output_dim, name="logits")
        self.logits = dense_op(net)

    def build_predictions(self):
        """Builds the output predictions and losses."""
        prediction_fn = (
            tf.sigmoid if self.hparams.output_dim == 1 else tf.nn.softmax)
        self.predictions = prediction_fn(self.logits, name="predictions")

    def build_losses(self):
        """Builds the training losses."""
        if self.hparams.output_dim == 1:
            num_classes = 2
            target_probabilities = tf.cast(self.labels, tf.float32)
        else:
            num_classes = self.hparams.output_dim
            target_probabilities = tf.one_hot(self.labels, depth=num_classes)

        label_smoothing = self.hparams.get("label_smoothing", 0)
        if label_smoothing > 0:
            target_probabilities = (
                target_probabilities * (1 - label_smoothing) +
                label_smoothing / num_classes)

        if self.hparams.output_dim == 1:
            batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=target_probabilities, logits=tf.squeeze(self.logits, [1]))
        else:
            batch_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=target_probabilities, logits=self.logits)

        weights = self.weights if self.weights is not None else 1.0
        self.total_loss = tf.losses.compute_weighted_loss(
            losses=batch_losses,
            weights=weights,
            reduction=tf.losses.Reduction.MEAN)

    def build(self):
        """Creates all ops for training, evaluation or inference."""
        self.global_step = tf.train.get_or_create_global_step()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = tf.placeholder_with_default(True, [], "is_training")
        else:
            self.is_training = False

        self.build_time_series_hidden_layers()
        self.build_aux_hidden_layers()
        self.build_logits()
        self.build_predictions()

        if self.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self.build_losses()

def base():
    return {
        "inputs": {
            "features": {
                "global_view": {
                    "length": 2001,
                    "is_time_series": True,
                    "subcomponents": [],
                },
            },
            "label_feature": "av_training_set",
            "label_map": {
                "PC": 1,
                "AFP": 0,
                "NTP": 0,
                "SCR1": 0,
                "INV": 0,
                "INJ1": 1,
                "INJ2": 0,
            },
        },
        "hparams": {
            "output_dim": 1,
            "num_pre_logits_hidden_layers": 1,  # Add at least one hidden layer
            "pre_logits_hidden_layer_size": 64,  # Adjust the size according to your needs
            "pre_logits_dropout_rate": 0.0,
            "batch_size": 256,
            "learning_rate": 2e-4,
            "learning_rate_decay_steps": 0,
            "learning_rate_end_factor": 0.0,
            "learning_rate_decay_power": 1.0,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
            "optimizer": "adam",
            "clip_gradient_norm": None,
        }
    }

def train_model(model, train_input_fn, train_steps):
    model.build()
    hooks = [
        tf.train.StopAtStepHook(last_step=train_steps),
        tf.train.LoggingTensorHook({"loss": model.total_loss}, every_n_iter=10),
    ]
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        while not sess.should_stop():
            sess.run(train_input_fn)

@app.route('/')
def index():
    return render_template('training.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        num_epochs = int(request.form['num_epochs'])
        learning_rate = float(request.form['learning_rate'])
        batch_size = int(request.form['batch_size'])
        model_config["hparams"]["num_epochs"] = num_epochs
        model_config["hparams"]["learning_rate"] = learning_rate
        model_config["hparams"]["batch_size"] = batch_size

        model = AstroModel(features={}, labels=None, hparams=model_config["hparams"], mode=tf.estimator.ModeKeys.TRAIN)
        train_input_fn = ...  # Define your training input function
        train_steps = ...  # Define the number of training steps
        train_model(model, train_input_fn, train_steps)
        return jsonify({'success': True, 'message': 'Training started successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    model_config = base()
    app.run(debug=True)
