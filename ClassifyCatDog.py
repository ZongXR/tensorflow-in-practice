from classifier.CatDogClassifier import CatDogClassifier


if __name__ == '__main__':
    # 加载模型并训练
    clf = CatDogClassifier("./models/CatDogClassifier.h5", "./logs/CatDogClassifier")
    clf.build_model()
    clf.fit("./data/cats_and_dogs_filtered")
    clf.save_weights()
