import tensorflow as tf


def roundWithPrecision(tensor, precision_bits=2):
    N = 2 ** precision_bits
    return tf.round(N * tensor) / N


session = tf.Session()

a = tf.Variable(3, dtype=tf.float32)
x = tf.placeholder(tf.float32)
b = tf.Variable(-15, dtype=tf.float32)
y = tf.placeholder(tf.float32)

# y = a * x + b

linearModel = a * x + b
linearModelRounded = roundWithPrecision(linearModel, precision_bits=2)
squaredDeltas = tf.square(linearModel - y)
loss = tf.reduce_sum(squaredDeltas)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
session.run(init)

for i in range(1000):
    session.run(train, {x: [1, 2, 3, 4], y: [4, 8, 16, 20]})




print(session.run(linearModelRounded, {x: [44]}))
print("----------------------------------------------------------")
print(session.run([roundWithPrecision(a), roundWithPrecision(b)]))