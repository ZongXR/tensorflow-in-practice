from classifier.HandwritingClassifier import HandwritingClassifier
from IPython import get_ipython
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.backend import clear_session


if __name__ == '__main__':
    # 设置显存自动增长
    clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 获取样本
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 构建模型
    clf = HandwritingClassifier("./models/HandwritingClassifier.h5", "./logs/HandwritingClassifier")
    clf.build_model()
    x_train, x_test = HandwritingClassifier.pre_process(x_train, x_test)
    history = clf.fit(x_train, y_train)
    if history is not None:
        print(history.history['accuracy'][-1])
    clf.save_weights()
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')