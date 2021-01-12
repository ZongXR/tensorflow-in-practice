import tensorflow_datasets as tdfs
from classifier.ImdbClassifier import ImdbClassifier


if __name__ == '__main__':
    data, info = tdfs.load("imdb_reviews", with_info=True, as_supervised=True)
    data_train = data["train"]
    data_validation = data["test"]
    x_train = []
    y_train = []
    x_validation = []
    y_validation = []
    for x, y in data_train:
        x_train.append(x.numpy().decode("utf8"))
        y_train.append(y.numpy())
    for x, y in data_validation:
        x_validation.append(x.numpy().decode("utf8"))
        y_validation.append(y.numpy())
    clf = ImdbClassifier(
        "./models/ImdbClassifier.h5",
        "./logs/ImdbClassifier"
    )
    clf.build_model()
    clf.fit(
        _x_train=x_train,
        _y_train=y_train,
        _x_validation=x_validation,
        _y_validation=y_validation
    )
    clf.save_weights()
    word_index = clf.get_word_index(reverse=True)
    weights = clf.get_weight(index=0)
    f_words = open("./data/ImdbReviews/words.tsv", "w+", encoding="utf8")
    f_vec = open("./data/ImdbReviews/vectors.tsv", "w+", encoding="utf8")
    for index, word in word_index.items():
        if index - 1 < weights.shape[0]:
            f_words.write(word + "\n")
            f_vec.write("\t".join(weights[index - 1, :].astype(str).tolist()) + "\n")
    f_vec.close()
    f_words.close()

