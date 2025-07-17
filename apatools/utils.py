import numpy as np
import pandas as pd
import zipfile



def read_zip(zip_file_path, 
             use_file_name=None, 
             use_file_index=0, 
             print_file_names=False,
             **kwargs):
    fname = None
    zf = zipfile.ZipFile(zip_file_path)
    if use_file_name is not None:
        for f in zf.infolist():
            f = f.filename
            if print_file_names:
                print(f)
            if use_file_name in f:
                fname = f
                break
    else:
        fname = zf.infolist()[use_file_index].filename
        if print_file_names:
            print(fname)
    if fname is not None:
        if '.xls' in fname.lower():
            return pd.read_excel(zf.open(fname), **kwargs)
        return pd.read_csv(zf.open(fname), **kwargs)


