"""Training utilities"""

import os
import logging
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.layers import GaussianNoise, RandomContrast, RandomBrightness
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf


def plot_conf_mtx(y_true, y_pred, labels_enc, labels) -> np.ndarray:
    """Plot confusion matrix with accuracy"""
    accuracy = accuracy_score(y_true, y_pred)
    cmtx = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_enc,
    )

    plt.figure(figsize=(10, 7))
    sns.heatmap(cmtx, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predictions')
    plt.ylabel('True classes')
    plt.suptitle('Confusion matrix')
    plt.title(f"Accuracy : {accuracy :0.2f}")
    plt.show()

    return cmtx


class ConditionalAugmentation(tf.keras.layers.Layer):
    def __init__(self, rate=0.2, **kwargs):
        super(ConditionalAugmentation, self).__init__(**kwargs)
        self.rate = rate
        self.flip = RandomFlip("horizontal")
        self.rotation = RandomRotation(0.25)
        self.zoom = RandomZoom(0.1)
        self.noise = GaussianNoise(0.1)
        self.contrast = RandomContrast(0.1)
        self.brightness = RandomBrightness(0.1)

    def call(self, inputs, training=None):
        if training:
            x = inputs
            x = tf.cond(
                tf.random.uniform(()) < self.rate, lambda: self.flip(x), lambda: x
            )
            x = tf.cond(
                tf.random.uniform(()) < self.rate, lambda: self.rotation(x), lambda: x
            )
            x = tf.cond(
                tf.random.uniform(()) < self.rate, lambda: self.zoom(x), lambda: x
            )
            x = tf.cond(
                tf.random.uniform(()) < self.rate, lambda: self.noise(x), lambda: x
            )
            x = tf.cond(
                tf.random.uniform(()) < self.rate, lambda: self.contrast(x), lambda: x
            )
            x = tf.cond(
                tf.random.uniform(()) < self.rate, lambda: self.brightness(x), lambda: x
            )
            return x
        return inputs


def eval_pretrained_model(
    model,
    train_ds,
    val_ds,
    test_ds,
    LOG_DIR,
    CHKPT_DIR,
    model_name="raw_model",
    input_size=(224, 224),
    batch_size=32,
    n_epochs=10,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
) -> tuple:
    """Train, evaluate and log pre-trained model from architecture and configuration

    Return model, history and plot confusion matrix
    """

    if not os.path.exists(CHKPT_DIR):
        os.makedirs(CHKPT_DIR)
    chkpt_name = model_name + ".weights.h5"
    chkpt_uri = os.path.join(CHKPT_DIR, chkpt_name)

    model_config = f"""
| Config | Value |
|:---:|:---:|
| **model name** | {model_name} |
| **input size** | {input_size} |
| **batch size** | {batch_size} |
| **n epochs** | {n_epochs} |
| **optimizer** | {optimizer} |
| **loss** | {loss} |
| **metrics** | {metrics} |
| **best weights URI** | {chkpt_uri} |
    """

    # set log folder
    log_dir = os.path.join(
        LOG_DIR, model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # COMPLIE
    logging.info("⚙️ compiling")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    # CALLBACKS
    logging.info("🛎️ declaring callbacks")

    class TimingCallback(Callback):
        def __init__(self):
            self.logs = []
            self.start_time = None

        def on_train_begin(self, logs={}):
            self.start_time = time.time()

        # log time by epoch
        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(time.time() - self.start_time)

        # log total time
        def on_train_end(self, logs={}):
            self.tot_time_sec = time.time() - self.start_time
            self.total_time = f"Total train time: {self.tot_time_sec // 60 :.0f}'{self.tot_time_sec % 60 :.0f}s"

    timing_callback = TimingCallback()
    checkpoint = ModelCheckpoint(
        chkpt_uri,
        save_best_only=True,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,  # do not save weights & biases (too much memory)
        write_graph=True,
        write_images=True,
        update_freq="epoch",
    )

    # FIT
    logging.info("💪 starting training")
    model_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=[timing_callback, checkpoint, early_stopping, tensorboard_callback],
    )

    # EVALUATE ON TEST DATASET
    logging.info("🧐 evaluating model")
    model.load_weights(chkpt_uri)
    test_loss, *test_metrics = model.evaluate(test_ds)
    predictions = model.predict(test_ds)

    # CONFUSION MATRIX
    logging.info("📈 plotting results")
    # get true labels from test dataset
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)
    # convert predictions to classes
    predicted_classes = np.argmax(predictions, axis=1)
    # compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_classes)
    # precision & F1 score
    report = classification_report(
        true_labels,
        predicted_classes,
        target_names=test_ds.class_names,
    )
    report_dict = classification_report(
        true_labels,
        predicted_classes,
        target_names=test_ds.class_names,
        output_dict=True,
    )
    print(report)

    # plot it
    conf_mtx_plot = plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_ds.class_names,
        yticklabels=test_ds.class_names,
    )
    plt.suptitle(f"{model_name} model", color="blue", weight="bold")
    plt.title(
        f"acc. {report_dict['accuracy'] :.02f} - loss {test_loss :.02f} - {timing_callback.total_time}",
        fontsize=10,
    )
    plt.xlabel("Predictions", color="red", weight="bold")
    plt.ylabel("True labels", color="green", weight="bold")
    plt.show()

    # convert image for Tensorboard
    conf_mtx_plot.canvas.draw()
    image_array = np.array(conf_mtx_plot.canvas.renderer.buffer_rgba())
    conf_mtx_plot_tf = tf.convert_to_tensor(image_array)
    conf_mtx_plot_tf = tf.expand_dims(conf_mtx_plot_tf, 0)

    plt.close()

    # LOG IN TENSORBOARD
    logging.info("📓 logging results")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    with file_writer.as_default():
        tf.summary.text("configuration", model_config, step=0)
        tf.summary.text("total_training_time", timing_callback.total_time, step=0)
        for i, time_per_epoch in enumerate(timing_callback.logs):
            tf.summary.scalar("time_per_epoch", time_per_epoch, step=i + 1)
        tf.summary.image("confusion_matrix", conf_mtx_plot_tf, step=0)

    return model, model_history


if __name__ == "__main__":
    help()
