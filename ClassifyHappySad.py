from classifier.HappySadClassifier import HappySadClassifier
from IPython import get_ipython


if __name__ == '__main__':
    clf = HappySadClassifier("./models/HappySadClassifier.h5", "./logs/HappySadClassifier")
    clf.build_model()
    clf.fit("./data/HappySad")
    clf.save_weights()
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')