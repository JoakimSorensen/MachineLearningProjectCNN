# An Ensemble of CNN
## Machine Learning project 2017
### Read me

NOTE: All commands should be run inside the tensorflow environment

###### Dependencies
Python 3.5.1
Tensorflow 1.3
numpy 1.13

Although the model might work with previous version of above libraries, these version are what were used
in development. Link to dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols

![](MathExprJpeg/cnn-ex-kopia.jpeg)

###### Preparation
Do the following steps only if the folders MathExprJpeg/train_data and MathExprJpeg/train_data and files x_data.npy, y_data.npy and labels.npy
are none existent, otherwise go directly to Running Main section.

  In order to run the model the training and testing folders must be created. If test_data folder and train_data
folder does not exist in the MathExprJpeg folder, these must be created. Create them by running the script
create_train_test_data.py like this:
```
python create_train_test_data.py
```

Then the datafiles for the training data must be created in order to speed up training. Called x_data.npy, y_data.npy and labels.npy. If non existent create by 
running the script create_datafiles.py like this:

```
python create_datafiles.py
```

###### Running main
After the preparation steps are done, set the preferred modes in cnn_math_main.py file. This is done by changing the parameters
at the top of the file:

```python
# set if data shold be read in advance
fileread = True
ensemble_mode = False 
```

fileread = True means that the data will be read from the previously created x_data.npy and y_data.npy files. 
Setting this to True is highly recommended. The ensemble_mode = False means that the model will not be run in 
ensemble mode. This is recommended as the ensemble mode is performance heavy and can not be guaranteed to work
in the latest releases.

It is recommended to change the name of the logging file for each run:

```python
writer = tf.summary.FileWriter('./logs/cnn_math_logs_true_2ep_r1')
writer.add_graph(sess.graph)
```

Also set the preferred value to the epoch and batch_size:

```python
training_epochs = 40
batch_size = 20
```

When all of the above has been done, the model can be run with the command:

```
python cnn_math_main.py
```

The model has been known to sometimes get errors while reading files. The source of which is unknown. If such
an error is to occur, run the following commands:

```
rm -r MatchExprJpeg/train_data
rm -r MatchExprJpeg/test_data
rm x_data.npy
rm y_data.npy
rm labels.npy

python create_train_test_data.py
python create_datafiles.py
```

And then try to run the cnn_math_main.py again. start tensorboard with:
```
tensorboard --logdir=./logs
```
to see the graphs for the scalars, the image being processed etc.

The model is implemented in the cnn_model.py class. If this file is tempered with it is possible that the
model breaks.

  Happy predicting

# MachineLearningProjectCNN
