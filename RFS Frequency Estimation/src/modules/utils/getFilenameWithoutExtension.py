import os

def getFilenameWithoutExtension(file_abs_path, getFileType=False):
    file_basename = [os.path.basename(x) for x in file_abs_path]
    
    if not getFileType:
        file_purename = [x.split('.')[0] for x in file_basename]
        return file_purename

    return file_basename
