# Deep Learning in TensorFlow
Collection of Google Colab notebooks created with reference to a TensorFlow course by ZeroToMastery.

## Summary of notebook contents

### 00_tensorflow_fundamentals
* Creating tensors
* Turning NumPy arrays into tensors
* Indexing tensors
* Tensor operations (+, -, x, /, matrix multiplication, dot product)
* Aggregating tensors (minimum, maximum, mean, sum)
* Finding positional maximum / minimum of tensors
* Squeezing a tensor
* One-hot encoding tensors

### 01_neural_network_regression_with_tensorflow (regression probelms)
* Datasets: NumPy arrays, [Insurance dataset](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv), `make_regression()`
* Steps in modelling with TensorFlow
* Visualising the data, model and the model's predictions
* Evaluating the model's predictions with regression evaluation metrics (`MAE`, `MSE`)
* Ways to improve a model
* Saving and loading a model in `SavedModel` / `HDF5` format
* Preprocessing data (normalisation)

### 02_nerual_network_classification_with_tensorflow (binary & multi-class classification problems)
* Datasets: `make_circles()`, [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist), `make_moons()`
* Linear and non-linear activations
* Plotting loss (training) curves
* Finding the best learning rate using `LearningRateScheduler` callback
* Classification evaluation metrics (`accuracy`, `precision`, `recall`, `f1-score`, `confusion matrix`, `classification report`)
* Visualising random images of training data for multi-class classification
* Visualising random images of test data and its predictions for multi-class classification

### 03_introduction_to_computer_vision_with_tensorflow (binary & multi-class classification problems)
* Datasets: Food 101 dataset ([pizza & steak](https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip) & [10 food classes](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip))
* Loading and preprocessing images
* Building a convolutional neural network (CNN)
* Data augmentation
* Making predictions on custom data

### 04_transfer_learning_in_tensorflow_part_1_feature_extraction (EfficientNetB0, ResNetV2 50, MobileNetV2 100)
* Datasets: [10% of 10 food classes](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip) from Food 101
* Creating data loaders
* Setting up callbacks (`TensorBoard`, `ModelCheckpoint`, `EarlyStopping`)
* Creating models using TensorFlow Hub
* Comparing model results using TensorBoard

### 05_transfer_learning_in_tensorflow_part_2_fine_tuning (EfficientNetB0, EfficientNetB4)
* Datasets: [10%](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip) & [100%](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip) of 10 food classes from Food 101
* Creating helper functions
* Building a transfer learning feature extraction model using the Keras Functional API
* Getting a feature vector from a trained model
* Adding data augmentation using Keras preprocessing layers
* Using checkpointed weights
* Setting layers of a feature extraction model to trainable (fine-tuning)

### 06_transfer_learning_in_tensorflow_part_3_scaling_up (EfficientNetB0)
* Datasets: [10% of 101 food classes](https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip)
* Using a classification report
* Finding the most wrong predictions

### 07_milestone_project_1_food_vision (EfficientNetB0, EfficientNetB4)
* Datasets: [Food 101](https://www.tensorflow.org/datasets/catalog/food101)
* Creating preprocessing functions for the data (`float32` datatype, in batches, normalised tensors)
* Batch and prepare datasets with `tf.data.Datasets`
* Setting up `mixed_precision` training to speed up model training
* Checking layer dtype policies to see if mixed precision is used

### 08_introduction_to_nlp_in_tensorflow (Naive Bayes, LSTM, GRU, Bidirectional RNN, Conv1D)
* Datasets: [Kaggle's Natural Language Processing with Disaster Tweets](https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip)
* Visualising a text dataset using `pandas`
* Creating text vectorization (tokenisation) and embedding layers
* Building a baseline model
* Evaluating a model using classification evaluation metrics
* Visualising learned embeddings
* Building a recurrent neural network (RNN)
* Using pretrained embeddings (TF Hub Universal Sentence Encoder)
* Comparing the performance of each model using DataFrames
* Plotting the performance of each model and sorting by evaluation metrics
* Finding the most wrong examples
* Making predictions on the test dataset
* The speed / score tradeoff
* Making a submission to the Kaggle competition
* Combining ensemble predictions using the majority vote (mode)

### 09_milestone_project_2_skimlit (Naive Bayes, Conv1D)
* Datasets: [PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)
* Making numeric labels
* Creating character vectorization and embedding layers
* Creating positional embeddings
* Label smoothing
* Building a hybrid & tribrid model
* Combining hybrid & tribrid data in datasets
* Using pretrained GloVe embeddings & pretrained TF Hub BERT PubMed expert embeddings

### 10_time_series_forecasting_with_tensorflow (Naive forecast, Conv1D, LSTM, N-BEATS)
* Datasets: Historical Bitcoin price
* Importing time series with `pandas` and `Python's CSV module`
* Plotting a time series
* Building a baseline model (naive forecast)
* Evaluating a time series model using regression metrics (`MAE`, `MSE`, `RMSE`, `(s)MAPE`, `MASE`)
* Windowing a dataset using array indexing
* Making, evaluating and plotting predictions
* Using `layers.Lambda` to reshape input data
* Making a multivarirate time series dataset
* Making a windowed dataset using `pandas.DataFrame.shift()`
* Creating custom layers and models using subclassing
* Getting ready for residual connections using TensorFlow `subtract` and `add` layers
* Building, compiling and fitting the N-BEATS algorithm
* Stacking different models together (an ensemble)
* Plotting prediction intervals (uncertainty estimates) of an ensemble
* Two types of uncertainty (coconut and subway)
* Making predictions into the future
* The turkey problem

## TensorFlow resources
* [Documentation on Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
* [Documentation on Softmax Activation](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax)
* [TensorFlow Classification Tutorial](https://www.tensorflow.org/tutorials/keras/classification)
* [TensorFlow Data Augmentation Tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation)
* [TensorFlow Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
* [Transfer Learning with TensorFlow Hub Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)
* [Fine-tuning a TensorFlow Hub Model Tutorial](https://www.tensorflow.org/hub/tf2_saved_model#fine-tuning)
* [Documentation on Learning Rate Scheduling](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler)
* [Documentaion on ModelCheckpoint Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
* [Documentation on EarlyStopping Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
* [Overview of TensorFlow Datasets](https://www.tensorflow.org/datasets/overview)
* [TensorFlow Data Performance Guide](https://www.tensorflow.org/guide/data_performance)
* [TensorFlow Mixed Precision Training Guide](https://www.tensorflow.org/guide/mixed_precision)
* [TensorFlow Text loading Tutorial](https://www.tensorflow.org/tutorials/load_data/text)
* [Classification on Imbalanced Data Tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
* [Documentation on Embedding Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
* [Documentation on clone_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model)
* [TensorFlow Save and Load Model Tutorial](https://www.tensorflow.org/tutorials/keras/save_and_load)
* [TensorFlow Recurrent Neural Network Guide](https://www.tensorflow.org/guide/keras/rnn)
* [TensorFlow tf.data Guide](https://www.tensorflow.org/guide/data)
* [Keras Guide on Using Pre-trained GloVe Embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/)
* [TensorFlow Making new Layers and Models via Subclassing Guide](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
* [Documentation on timeseries_dataset_from_array](https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array)

## List of reading materials and resources
* [MIT's Introduction to Deep Learning Lecture](https://youtu.be/njKP3FqW3Sk)
* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [Details on EfficientNet](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
* [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
* [Scikit-learn Machine Learning Map](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
* [Sklearn Documentation on Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
* [A Simple Introduction to NLP](https://becominghuman.ai/a-simple-introduction-to-natural-language-processing-ea66a1747b32)
* [How to solve 90% of NLP problems: a step-by-step guide](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e)
* [The Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/)
* [MIT's Sequence Modelling Lecture](https://youtu.be/QvkQ1B3FBqA?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Illustrated Guide to LSTM's and GRU's](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
* [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
* [Text Classification with NLP: Tf-Idf vs Word2Vec vs BERT](https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794)
* [PubMed 200k RCT: a Dataset for Seqeuntial Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071)
* [Neural Networks for Joint Sentence Classification in Medical Paper Abstracts](https://arxiv.org/abs/1612.05251)
* [Label smoothing with Keras, TensorFlow and Deep Learning](https://pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/)
* [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
* [How (not) to use Machine Learning for time series forecasting: Avoiding the pitfalls](https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424)
* [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/pdf/1905.10437)
* [Engineering Uncertainty Estimation in Neural Networks for Time Series Prediction at Uber](https://eng.uber.com/neural-networks-uncertainty-estimation/)
* [MIT 6.S191: Evidential Deep Learning and Uncertainty](https://youtu.be/toTcf7tZK8c)
* [Why you should care about the Nate Silver vs. Nassim Taleb Twitter war](https://towardsdatascience.com/why-you-should-care-about-the-nate-silver-vs-nassim-taleb-twitter-war-a581dce1f5fc)
* [3 facts about time series forecasting that surprise experienced machine learning practitioners](https://towardsdatascience.com/3-facts-about-time-series-forecasting-that-surprise-experienced-machine-learning-practitioners-69c18ee89387)
* [Engineering Uncertainty Estimation in Neural Networks for Time Series Prediction at Uber](https://eng.uber.com/neural-networks-uncertainty-estimation/)
