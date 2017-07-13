import tempfile
import pandas as pd
import tensorflow as tf
from urllib.request import urlretrieve

# Retrieving census data
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()

urlretrieve(url="http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", filename=train_file.name)
urlretrieve(url="http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", filename=test_file.name)

# Read-in the census data and add column names
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship",
           "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]

df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

# Create label columns that will be predicted (everyone whose income is above 50K is 1 and 0 otherwise)
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = df_train.income_bracket.apply(lambda x: x == ">50K").astype(int)
df_test[LABEL_COLUMN] = df_test.income_bracket.apply(lambda x: x == ">50K").astype(int)

# Remove income column
df_train = df_train.drop(labels="income_bracket", axis=1)
df_test = df_test.drop(labels="income_bracket", axis=1)

# Determine which columns are categorical and which are continuous
df_train.head(5)
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "gender",
                       "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


def create_tensors(dataframe, label_col, cont_col, cat_col):
    # puts continuous columns into dict (col_name: tensor)
    continuous_cols = {k: tf.constant(value=dataframe[k].values) for k in cont_col}
    # puts categorical columns and creates dict (col_name: sparse_tensor)
    categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(dataframe[k].size)],
      values=dataframe[k].values,
      dense_shape=[dataframe[k].size, 1])
                      for k in cat_col}
    # combines both dictionaries
    feature_cols = {**continuous_cols, **categorical_cols}
    # transforms label column into tensor
    label = tf.constant(dataframe[label_col].values)
    return feature_cols, label

# This is stupid! from Tensorflow tutorial example!
# Categorical variables
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
education = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="education", hash_bucket_size=1000)
race = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="race", hash_bucket_size=100)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="native_country", hash_bucket_size=1000)
# Continuous  variables
age = tf.contrib.layers.real_valued_column(column_name="age")
education_num = tf.contrib.layers.real_valued_column(column_name="education_num")
capital_gain = tf.contrib.layers.real_valued_column(column_name="capital_gain")
capital_loss = tf.contrib.layers.real_valued_column(column_name="capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column(column_name="hours_per_week")


# Splits age into buckets
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Intersecting Multiple Columns
# =================================
# Using each base feature column separately may not be enough to explain the data. For example, the correlation between
# education and the label (earning > 50,000 dollars) may be different for different occupations. Therefore, if we only
# learn a single model weight for education="Bachelors" and education="Masters", we won't be able to capture every
# single education-occupation combination (e.g. distinguishing between education="Bachelors" AND
# occupation="Exec-managerial" and education="Bachelors" AND occupation="Craft-repair"). To learn the differences
# between different feature combinations, we can add crossed feature columns to the model.
# =================================
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation],
                                                          hash_bucket_size=int(1e4))
age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation],
                                                                        hash_bucket_size=int(1e6))

# Logistic Regression Model
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir)


def train_input_fn():
    return create_tensors(dataframe=df_train,
                          label_col=LABEL_COLUMN,
                          cont_col=CONTINUOUS_COLUMNS,
                          cat_col=CATEGORICAL_COLUMNS)


def eval_input_fn():
    return create_tensors(dataframe=df_test,
                          label_col=LABEL_COLUMN,
                          cont_col=CONTINUOUS_COLUMNS,
                          cat_col=CATEGORICAL_COLUMNS)


m.fit(input_fn=train_input_fn, steps=200)

results = m.evaluate(input_fn=eval_input_fn, steps=1)

for key in sorted(results):
    print("%s: %s" % (key, results[key]))
