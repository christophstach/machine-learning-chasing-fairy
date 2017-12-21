import numpy as np
import tensorflow as tf

featureColumns = [
    tf.feature_column.numeric_column("x", shape=[1])
]

estimator = tf.estimator.LinearRegressor(feature_columns=featureColumns)

xTrain = np.array([1., 2., 3., 4.])
yTrain = np.array([1., 2., 3., 4.])

xEval = np.array([6., 7., 8.])
yEval = np.array([6., 7., 8.])

inputFn = tf.estimator.inputs.numpy_input_fn(
    x={"x": xTrain},
    y=yTrain,
    batch_size=128,
    num_epochs=None,
    shuffle=False
)
trainInputFn = tf.estimator.inputs.numpy_input_fn(
    x={"x": xTrain},
    y=yTrain,
    batch_size=128,
    num_epochs=1000,
    shuffle=True
)
evalInputFn = tf.estimator.inputs.numpy_input_fn(
    x={"x": xEval},
    y=yEval,
    batch_size=128,
    num_epochs=1000,
    shuffle=False
)

estimator.train(input_fn=inputFn, steps=1000)

trainMetrics = estimator.evaluate(input_fn=trainInputFn)
evalMetrics = estimator.evaluate(input_fn=evalInputFn)


print(trainMetrics)
print(evalMetrics)