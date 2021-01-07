from tensorflow.keras.callbacks import Callback


class EarlyStopByAccuracy(Callback):
    _accuracy: float

    def __init__(self, _accuracy: float):
        """
        early stop 构造方法\n
        :param _accuracy: 精度
        """
        super().__init__()
        self._accuracy = _accuracy

    def on_epoch_end(self, epoch, logs=None):
        """
        按照精度的early stop
        :param epoch: 轮数
        :param logs: 日志
        :return:
        """
        if logs["accuracy"] > self._accuracy:
            print("\n精度到达%f，停止训练" % self._accuracy)
            self.model.stop_training = True
