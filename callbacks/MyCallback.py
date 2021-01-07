from tensorflow.keras.callbacks import Callback


class MyCallBack(Callback):
    """
    自定义回调\n
    """
    _ACCURACY = 0.998

    def on_epoch_end(self, epoch, logs=None):
        """
        early stop\n
        :param epoch: 轮数
        :param logs: 日志
        :return: 空
        """
        if logs["accuracy"] > MyCallBack._ACCURACY:
            print("\n精度已到达%s停止训练" % MyCallBack._ACCURACY)
            self.model.stop_training = True
