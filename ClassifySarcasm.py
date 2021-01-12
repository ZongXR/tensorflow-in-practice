from classifier.SarcasmClassifier import SarcasmClassifier
import tensorflow as tf
from json import load


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    with open("./data/sarcasm/sarcasm.json") as f:
        datas = load(f)
    x = []
    y = []
    for data in datas:
        x.append(data["headline"])
        y.append(data["is_sarcastic"])
    x_train = x[0:20000]
    y_train = y[0:20000]
    x_validation = x[20000:]
    y_validation = y[20000:]
    clf = SarcasmClassifier("./models/SarcasmClassifier.h5", "./logs/SarcasmClassifier")
    clf.build_model()
    clf.fit(
        _x_train=x_train,
        _y_train=y_train,
        _x_validation=x_validation,
        _y_validation=y_validation
    )
    clf.save_weights()
    clf.words_vectors("./data/sarcasm/words.tsv", "./data/sarcasm/vectors.tsv")