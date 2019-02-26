# Linear Regression

In this chapter, we introduce example for Linear Regression and as before we will start with data preparation stage.

## Data Preparation

This time we are going to use synthetic data. As you can see in the code below,

```python
# Synthetic Data
# Define one-dimensional feature vector
feature = 5.0 * np.random.random(size=(1000, 1)) - 1
# Creates random noise with amplitude 0.1, which we add to the target values
noise = 0.1 * np.random.normal(scale=1, size=feature.shape)
# Defines two-dimensional target array
target_1 = 2.0 * feature + 3.0 + noise
target_2 = -1.2 * feature / 6.0 + 1.01 + noise
target = np.concatenate((target_1, target_2), axis=1)

# Split data sets into Training, Validation and Test sets
X_train_val, X_test, Y_train_val, Y_test = train_test_split(feature, target, test_size=0.33, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.33, random_state=42)
```

features are just a randomly generated numbers in the range -1 and 4. The shape of the array is `[1000, 1]`. To make life just a bit more interesting, we also create a random noise with the maximum amplitude of 0.1. Further, we create two target arrays, that are later concatenated into one `[1000, 2]` numpy array. The parameters and coefficients that are used in the example are arbitrary and therefore feel free to play around. However, note that if your target or/and feature values are very very small or very very large you may need to rescale them or change hyperparameters. Otherwise, it will be very difficult for the model to make good predictions. The final step in the data preparation stage, as before, is splitting the feature and the target arrays into train, validation and test datasets.

## Graph Construction

Although in this example feature and target arrays have changed the shape when compared with the example for the logistic regression, the inputs in the graph remain the same, as well as the structure of the graph itself.

```python
with tf.variable_scope("inputs"):
    # placeholder for input features
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_FEATURES], name="predictors")
    # placeholder for true values
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, Y_FEATURES], name="target")
```

Both `X_FEATURES` and `Y_FEATURES` are computed during the script execution, as each numpy array contains a [`shape`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) option that returns the tuple of array dimensions, therefore we do not need to worry about what values are assigned to these variables. Because we used `None` for the first dimension in the shape parameter for both placeholders we do not need to worry about the length of the data either.

In general, when we create placeholders for dense neural networks, the shape parameter should be a vector of the form: `[BATCH_SIZE, FEATURE NUMBER]`. As mentioned in the previous example, providing an explicit value for `BATCH_SIZE` could potentially cause a problem, thus it is common to use `None` instead. Therefore, a rule of thumb is to use the following for the `shape` parameter `[None, FEATURE NUMBER]`.

You might noticed the following command before the input definition, [`tf.reset_default_graph()`](https://www.tensorflow.org/api_docs/python/tf/reset_default_graph). This function, as the name suggests, clears and resets values in the default graph stack. This means that every time before we construct our graph, we ensure that all previously attached elements to the graph are removed.

### Linear Regression Model

Further, we create the model, and as we can see apart from the variable scope name change, only the loss function has been changed, when compared with the logistic regression example. In this situation, we use the [mean square error](https://en.wikipedia.org/wiki/Mean_squared_error) as the cost function.

```python
# Define logistic regression model
with tf.variable_scope("linear_regression"):
    # Predictions are performed by Y_FEATURES neurons in the output layer
    prediction = tf.layers.dense(inputs=x, units=Y_FEATURES, name="prediction")
    # Define loss function as mean square error (MSE)
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=prediction)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
```

In this example, as before, we use the gradient descent algorithm to optimize the weights and biases. In addition, as the changes to the current example are minimal, the model hyperparameters remain the same as in the previous example.

### Metrics

For completeness, we have also kept metrics section, but we have changed metrics that are actually computed.

```python
with tf.variable_scope("metrics"):
    # Determin total RMSE
    _, rmse = tf.metrics.root_mean_squared_error(labels=y_true, predictions=prediction)
    # Define total r_squared score as 1 - Residual sum of squares (rss) /  Total sum of squares (tss)
    y_true_bar = tf.reduce_mean(input_tensor=y_true, axis=0)
    tss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=y_true, y=y_true_bar)), axis=0)
    rss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=y_true, y=prediction)), axis=0)
    r_squared = tf.reduce_mean(tf.subtract(x=1.0, y=tf.divide(x=rss, y=tss)))
```

First is the [root mean squared error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) \(RMSE\) that is already implemented in TensorFlow as `tf.metrics.root_mean_squared_error()`. This function required two parameters `labels` and `predictions`, which in our case are `y_true` and `prediction` tensors, respectively. The second metric is the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) \(\), this, unfortunately, has not been implemented in TensorFlow yet. Hence we do it ourselves. TensorFlow has [implementation of basic mathematical operations](https://www.tensorflow.org/api_guides/python/math_ops) that can be utilised to build more advanced operations. So, our task is to build general definition for the coefficient of determination, which is defined as,



where  stands for observations and  are predictions.

As the names of the operations used in the code above are self-explanatory, we will limit explanation to only two functions, `tf.reduce_mean()` and `tf.reduce_sum()`. To begin, `tf.reduce_mean()` function computes a mean value along given tensor axis, this operation is equivalent to equation for . In our situations, this functions yields tensor of rank 1 \(vector\) which contains two mean values for each target. Next, `tf.reduce_sum()` is equivalent to  operation with option to specify the axis along which it has to perform summation.

## Model Training and Testing

Model training or graph execution stage remains exactly the same as for the logistic regression example, with the only difference in metrics that are evaluated and printed on the console.

```python
if e % 10 == 0:
      # Evaluate metrics on training and validation data sets
      train_loss = loss.eval(feed_dict={x: X_train, y_true: Y_train})
      val_loss = loss.eval(feed_dict={x: X_val, y_true: Y_val})
      # Prints the loss to the console
      msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
             "Train MSE: {tr_ls}; ".format(tr_ls=train_loss) +
             "Validation MSE: {val_ls}; ".format(val_ls=val_loss))
      print(msg)
```

Similarly, model testing,

```python
# Evaluate loss (MSE), total RMSE and R2 on test data
test_loss = loss.eval(feed_dict={x: X_test, y_true: Y_test})
rmse = rmse.eval(feed_dict={x: X_test, y_true: Y_test})
r_squared = r_squared.eval(feed_dict={x: X_test, y_true: Y_test})
# Evaluate prediction on Test data
y_pred = prediction.eval(feed_dict={x: X_test})
# Print Test loss (MSE), total RMSE and R2 in console
msg = "\nTest MSE: {test_loss}, RMSE: {rmse} and R2: {r2}".format(test_loss=test_loss, rmse=rmse, r2=r_squared)
print(msg)
```

For comparison we also compute `mean_squared_error` and `r2_score` using functions from `scikit-learn`,

```python
# Calculates RMSE and R2 metrics using sklearn
sk_rmse = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_pred))
sk_r2 = r2_score(y_true=Y_test, y_pred=y_pred)
print("Test sklearn RMSE: {rmse} and R2: {r2}".format(rmse=sk_rmse, r2=sk_r2))
```

For completeness, we compare both targets and predicted values by plotting them on one plot.

## Next

In the [next chapter](nonlinear-regression.md) we will see how to modify the code presented here for a fully-connected neural network which will allow us to perform the regression task for nonlinear functions. To return to the previous chapter press [here](logistic-regression.md).

## Code

* [02\_linear\_regression.py](https://github.com/satonreb/tensorflow-tutorial/blob/master/02_linear_regression.py)

## References

* [Numpy Manual](https://docs.scipy.org/doc/numpy/index.html)
* Wikipedia articles on [Mean Square Error](https://en.wikipedia.org/wiki/Mean_squared_error), [Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) and [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

