# EasyLSTM
#### EasyLSTM is a tool that eases the creation of LSTM's and shaping/slicing data to the correct shape.
#### With EasyLSTM you can go from import to prediction **in less than 10 lines of python code!**

## Requirements
- Python 3.8.0 _(Not limited to, if you can install the other dependencies)_
- TensorFlow >= 2.0
- Scikit-learn
- Keras
- Pandas
- Numpy

## Usage
```python
from easy_lstm import EasyLSTM

helper = EasyLSTM(data, timesteps)
model, X_train, y_train, X_test, y_test = helper.do_magic()
```
#### The _**do_magic()**_ method returns both a compiled model and the train/test split and reshaped data.

#### The data goes through a series of processes that transform a pandas DataFrame into numpy arrays, with the X being an array of time windows of size _**n_steps**_ from the dataframe feature columns, and the _**y**_ being the corresponding labels in the dataset ['y'] column.

## Data format
#### The data provided to the EasyLSTM's __\_init\_\__ method has to be a pandas dataframe in the following format:
##### The number of features/feature columns can be any number, as long as there is a at least one feature and a 'y' column

## Sample DataFrame
#### Note: All feature columns can be named anything, just the label column that need to be named _**'y'**_
| feature_0 | feature_1 | some_other_feature | y |
| :---------: | :---------: | :------------------: | - |
|0.1| 2.12 | 1600 | 0.2|
|0.2| 3.15 | 3000 | 0.3|
|0.3| 3.98 | 400 | 0.4|

## 4 Lines from import to prediction example
```python
from easy_lstm import EasyLSTM
import pandas
dataset = pandas.read_csv('./path_to_dataset.csv')
dataset['y'] = dataset['feature'].shift(-1, axis=0)[:-1] #Turning a time series into a supervised learning problem
model, X_train, y_train, X_test, y_test = EasyLSTM(data=dataset, n_steps=4).do_magic()
model.fit(X_train, y_train, epochs=20)
predictions = model.predict(X_test)
```
