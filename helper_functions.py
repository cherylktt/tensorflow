### A few helper functions for TensorFlow Deep Learning

# Unzip a zipfile into current working directory
import zipfile

def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Args:
        filename (str): a filepath to a target zip folder to be unzipped
    """

    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()

# Walkthrough an image classification directory and find out how many files there are
import os

def walk_through_dir(dir_path):
    """
    Walks through dir_path, returning its contents.

    Args:
        dir_path (str): target directory

    Returns:
        A print out of:
            number of subdirectories in dir_path
            number of images (files) in each subdirectory
            name of each subdirectory
    """

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Import an image and resize it to be able to be used with our model
import tensorflow as tf

def load_and_pre_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into (224, 224, 3).

    Args:
        filename (str): path to target image
        img_shape (int): height/width dimension of target image size
        scale (bool): scale pixel values from 0-255 to 0-1 or not
    """

    # Read in the image
    img = tf.io.read_file(filename)

    # Decode the image into a tensor
    img = tf.io.decode_image(img, channels=3) # hardcode 3 colour channels

    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])

    # Rescale the image
    if scale:
        return img/255.
    else:
        return img

# Visualise multiple random images from random classes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

def view_random_image(target_dir, x):

    """
    View x random images from random classes in target_dir.

    Args:
        target_dir (str): path to target directory
        x (int): number of images to view
    """

    for i in range(x):
        ax = plt.subplot(x/2, 2, i+1)

        # Set up target folder
        target_class = random.choice(class_names)
        target_folder = target_dir + "/" + target_class
        random_image = random.sample(os.listdir(target_folder))
        img = mpimg.imread(target_folder + "/" + random_image[0])

        # View the image
        plt.imshow(img/255.)
        plt.title(f"Random image of {target_class}", fontsize=10)
        plt.axis(False)

# Create TensorBoard callback
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.
    Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"
  
    Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboad_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to {log_dir}")
    return tensorboad_callback

# Create a model from a URL
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

def create_model(model_url, num_classes=10):
  """
  Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.
  
  Args:
    model_url (str): a TensorFlow Hub feature extraction URL
    num_classes (int): number of output neurons in the output layer,
        should be equal to number of target classes, default 10.

  Returns:
    A compiled Keras Sequential model with model_url as feature 
    extractor layer and Dense output layer with num_classes output neurons.
    loss = "categorical_crossentropy",
    optimizer = Adam()
    metrics = "accuracy"
  """

  # Download the pretrained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False, # freeze the already learned patterns
                                           name="feature_extraction_layer",
                                           input_shape=IMAGE_SHAPE+(3,)) # (224, 224, 3)

  # Create our own model
  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(num_classes, activation="softmax", name="output_layer")
  ])
  model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

  return model

# Evaluate the model
def evaluate_model(model, x, test_data=test_data):
    """
    Evaluate the loss and accuracy of model x on test data.
    """

    loss, accuracy = model.evaluate(test_data)
    print(f"Loss on model {x}: {loss}")
    print(f"Accuracy on model {x}: {(accuracy*100):.2f}%")

# Plot loss/training curves
import matplotlib.pyplot as plt

def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
        history: TensorFlow model History object
    """

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="training loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.plot(epochs, accuracy, label="training accuracy")
    plt.plot(epochs, val_accuracy, label="validation accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();

# Compare history (used in the case of fine tuning)
import matplotlib.pyplot as plt

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compare two TensorFlow model History objects.
    
    Args:
        original_history: History object from original model (before new_history)
        new_history: History object from continued model training (after original_history)
        initial_epochs: Number of epochs in original history (new_history plot starts from here)
    """

    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss = new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label="Validation Accuracy")
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label="Start Fine Tuning") # reshift plot around epochs
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label="Start Fine Tuning") # reshift plot around epochs
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.xlabel("Epoch")
    plt.show();

# Make predictions on multiple images from test data
import os
import random
import matplotlib.pyplot as plt

def pred_and_plot(model, target_dir, x):
    """
    View the model predictions of x random images from random classes in target_dir.

    Args:
        target_dir (str): path to target directory
        x (int): number of images to view
    """

    plt.figure(figsize=(17, 10))

    for i in range(x):
        # Choose random iamges from random classes
        target_class = random.choice(class_names)
        target_folder = target_dir + "/" + target_class
        random_image = random.sample(os.listdir(target_folder))
        filepath = target_dir + "/" + target_class + "/" + random_image

        # Load the image and make predictions
        img = load_and_pre_image(filepath, scale=False)
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]

        # Plot the images
        plt.subplot(x/3, 3, i+1)
        plt.imshow(img/255.)
        plt.axis(False)

        # Set title colour
        if class_name == pred_class:
            title_colour = "green"
        else:
            title_colour = "red"
        
        plt.title(f"actual: {class_name}\n prediction: {pred_class}\n probability: {pred_prob.max():.2f}", c=title_colour)

# Make predictions on multiple images from custom data
import matplotlib.pyplot as plt
import os

def pred_custom_image(model, filepath):
    """
    View the model predictions on custom images from filepath.

    Args:
        filepath (str): a filepath to the location where the custom images are stored
    """
    custom_images = [filepath + "/" + img_path for img_path in os.listdir(filepath)]
    
    for img in custom_images:
        img = load_and_pre_image(img, scale=False)
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]

        # Plot the appropriate information
        plt.figure()
        plt.imshow(img/255.)
        plt.title(f"prediction: {pred_class}\n probability: {pred_prob.max():.2f}")
        plt.axis(False);

# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """
  Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  

  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")