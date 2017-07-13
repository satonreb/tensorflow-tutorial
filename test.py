import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Global configuration
X_RANGE_MIN, X_RANGE_MAX = 0, 2 * np.pi
N = 10 * 4
TEST_SIZE = 0.33
PAST_SEQUENCE_LENGTH = 10
FUTURE_SEQUENCE_LENGTH = 5
STEPS_AHEAD = 1




# y_test_input = (None, PAST_SEQUENCE_LENGTH, 1)
# y_test_output = (None, FUTURE_SEQUENCE_LENGTH, 1)



# Create the data
X = np.arange(start=X_RANGE_MIN,
              stop=X_RANGE_MAX,
              step=(X_RANGE_MAX-X_RANGE_MIN)/N)
# np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
X = np.sort(X)
y = np.sin(X)

n_sequences = len(X) - np.max([PAST_SEQUENCE_LENGTH,
                               FUTURE_SEQUENCE_LENGTH]) - STEPS_AHEAD


# Build the time series
def sub_sequence(x, i, sequence_length):
    return x[i:i + sequence_length]

X_past_sequences = np.array([sub_sequence(
    x=X, i=i,
    sequence_length=PAST_SEQUENCE_LENGTH)
    for i in range(n_sequences)])
y_past_sequences = np.array([sub_sequence(
    x=y, i=i,
    sequence_length=PAST_SEQUENCE_LENGTH)
    for i in range(n_sequences)])

X_future_sequences = np.array([sub_sequence(
    x=X, i=i + PAST_SEQUENCE_LENGTH + (STEPS_AHEAD - 1),
    sequence_length=FUTURE_SEQUENCE_LENGTH)
    for i in range(n_sequences)])

y_future_sequences = np.array([sub_sequence(
    x=y, i=i + PAST_SEQUENCE_LENGTH + (STEPS_AHEAD - 1),
    sequence_length=FUTURE_SEQUENCE_LENGTH)
    for i in range(n_sequences)])

# Build training testing sample
tscv = TimeSeriesSplit(n_splits=2)
for train_index, test_index in tscv.split(X_past_sequences):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_past_sequences_train = X_past_sequences[train_index]
    X_past_sequences_test = X_past_sequences[test_index]

    X_future_sequences_train = X_future_sequences[train_index]
    X_future_sequences_test = X_future_sequences[test_index]

    y_past_sequences_train = y_past_sequences[train_index]
    y_past_sequences_test = y_past_sequences[test_index]

    y_future_sequences_train = y_future_sequences[train_index]
    y_future_sequences_test = y_future_sequences[test_index]

# Let's try one-dimensional
y_train_input = list(
    np.reshape(y_past_sequences_train,
               tuple(list(y_past_sequences_train.shape) + [1])))
y_train_output = list(
    np.reshape(y_future_sequences_train,
               tuple(list(y_future_sequences_train.shape))))

y_test_input = list(
    np.reshape(y_past_sequences_test,
               tuple(list(y_past_sequences_test.shape) + [1])))
y_test_output = list(
    np.reshape(y_future_sequences_test,
               tuple(list(y_future_sequences_test.shape))))

num_units = PAST_SEQUENCE_LENGTH
lstm_keep_prob = 1