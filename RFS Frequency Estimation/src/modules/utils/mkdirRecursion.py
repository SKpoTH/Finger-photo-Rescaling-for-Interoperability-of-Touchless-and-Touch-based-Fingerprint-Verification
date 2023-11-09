import os

def mkdirRecursion(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except:
            mkdirRecursion(os.path.dirname(path))
            if not os.path.isdir(path):
                os.mkdir(path)