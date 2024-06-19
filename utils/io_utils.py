import os
import pandas as pd

def load_corpus(file_name, header=None, names=["x_text", "y_label"], sep='\t'):
    df_test = pd.read_csv(file_name, header=header, names=names, sep=sep)
    x, y = df_test[names[0]].tolist(), df_test[names[1]].tolist()
    return x, y

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)