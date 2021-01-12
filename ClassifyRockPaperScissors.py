from classifier.RockPaperScissorsClassifier import RockPaperScissorsClassifier
from tensorflow.keras.backend import clear_session
import tensorflow as tf


if __name__ == '__main__':
    # 设置显存自动增长
    clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    clf = RockPaperScissorsClassifier(
        _model_weights="./models/RockPaperScissorsWeights.h5",
        _logs="./logs/RockPaperScissors",
        _model_path="./models/RockPaperScissorsModel.h5",
        _input_shape=(150, 150, 3)
    )
    clf.build_model()
    clf.fit("./data/Rock-Paper-Scissors/rps", "./data/Rock-Paper-Scissors/rps-test-set")
    clf.save("./models/RockPaperScissorsModel.h5")
    print(clf.predict_one(_filepath="./data/Rock-Paper-Scissors/rps-validation/rock2.png"))
