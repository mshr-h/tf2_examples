import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_dataset():
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset = tfds.load('mnist', as_supervised=True)
    train_ds, test_ds = dataset['train'], dataset['test']
    train_ds = train_ds.map(preprocess).shuffle(10000).batch(32)
    test_ds = test_ds.map(preprocess).batch(32)
    return train_ds, test_ds


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self._layers = ([
            tf.keras.layers.Conv2D(32, 3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


train_ds, test_ds = prepare_dataset()

model = CNN()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def main():
    for epoch in range(5):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            .format(epoch + 1, train_loss.result(),
                    train_accuracy.result() * 100, test_loss.result(),
                    test_accuracy.result() * 100))


if __name__ == '__main__':
    main()
