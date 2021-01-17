from classifier.ImdbSubwordsClassifier import ImdbSubwordsClassifier
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import tensorflow_datasets as tfds


if __name__ == '__main__':
    # 设置显存自动增长
    clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)

    data, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
    token: tfds.deprecated.text.SubwordTextEncoder = info.features["text"].encoder
    data_train: tf.raw_ops.PrefetchDataset = data["train"]
    # data_train: tf.raw_ops.ShuffleDataset = data_train.shuffle(10000)
    # data_train: tf.raw_ops.PaddedBatchDataset = data_train.padded_batch(64, tf.compat.v1.data.get_output_shapes(data_train))
    data_validation: tf.raw_ops.PrefetchDataset = data["test"]
    # data_validation: tf.raw_ops.PaddedBatchDataset = data_validation.padded_batch(64, tf.compat.v1.data.get_output_shapes(data_validation))
    clf = ImdbSubwordsClassifier(
        _model_weights="./models/ImdbSubwordsClassifier.h5",
        _logs="./logs/ImdbSubwordsClassifier",
        _num_words=token.vocab_size
    )
    clf.build_model()
    clf.fit(
        _x_train=None,
        _y_train=None,
        _x_val=None,
        _y_val=None,
        _data_train=data_train,
        _data_validation=data_validation
    )
    clf.save_weights()

