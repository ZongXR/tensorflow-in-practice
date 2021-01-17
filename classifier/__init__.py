"""
classifier
这个包是分类器\n
"""
import tensorflow as tf

# 重置
tf.compat.v1.reset_default_graph()

# 设置显存自动增长
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, enable=True)

__all__ = [
    "HandwritingClassifier",
    "CatDogClassifier",
    "HappySadClassifier",
    "HorseHumanClassifier",
    "FashionClassifier",
    "CatDogClassifierInceptionV3",
    "RockPaperScissorsClassifier",
    "ImdbClassifier",
    "SarcasmClassifier",
    "ImdbSubwordsClassifier",
    "TextGenerator"
]