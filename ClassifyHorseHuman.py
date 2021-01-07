from classifier.HorseHumanClassifier import HorseHumanClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import matplotlib.image as mpimg


def show_imgs(idxs: (int, ), path: str) -> None:
    """
    显示样本图片\n
    :param idxs: 样本索引
    :param path: 样本路径
    :return: 空
    """
    plt.figure(1)
    plt.subplots(len(os.listdir(path)), len(idxs))
    for i in range(0, len(os.listdir(path))):
        for j, jj in enumerate(idxs):
            idx = j + 1 + i * len(idxs)
            plt.subplot(len(os.listdir(path)), len(idxs), idx)
            _img = mpimg.imread("%s/%s/%s" % (path, os.listdir(path)[i], os.listdir(path + "/" + os.listdir(path)[i])[jj]))
            plt.imshow(_img)
            plt.title(os.listdir(path + "/" + os.listdir(path)[i])[jj])
    plt.tight_layout()
    plt.savefig("./imgs/人马图片.png")
    plt.show()


if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 展示样本
    print(os.listdir("./data/HorseAndHuman/train_data/horses")[0:10])
    print(os.listdir("./data/HorseAndHuman/train_data/humans")[0:10])
    print(len(os.listdir("./data/HorseAndHuman/train_data/horses")))
    print(len(os.listdir("./data/HorseAndHuman/train_data/humans")))
    show_imgs((1, 20, 300, 43), "./data/HorseAndHuman/train_data")
    # 训练模型
    clf = HorseHumanClassifier("./models/HorseHumanClassifier.h5", "./logs/HorseHumanClassifier")
    clf.build_model()
    clf.fit("./data/HorseAndHuman")
    clf.save_weights("./models/HorseHumanClassifier.h5")
    print(clf.predict("./data/HorseAndHuman/predict"))
    print(clf.predict_one("./data/HorseAndHuman/predict/7-1Q00Q64U1O0.jpg"))

