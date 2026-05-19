import numpy as np
import pandas as pd
import itertools
import asyncio
import zipfile
import scipy



def df_check_intercept(df): # look for the intercept column name
        if 'CONSTANT' in df.columns:
            return 'CONSTANT'
        if 'const' in df.columns:
            return 'const'
        elif "Intercept" in df.columns:
            return 'Intercept'
        else:
            return None


def df_standardize(df, func='z'):
    df = df.select_dtypes(include=[np.number, "bool"])
    const_name = df_check_intercept(df)
    
    if const_name is not None: # rem const
        if df.shape[1] == 1: # if only const in the data frame
            return df
        const_pos = df.columns.get_loc(const_name)
        const = df[const_name].copy()
        df.drop(const_name, axis=1, inplace=True) 

    if callable(func):
        df = df.apply(func)
    elif func == 'z' or (isinstance(func, bool) and func):
        df = df.apply(scipy.stats.zscore)
    else:
        raise NotImplementedError(
                f"Func '{func}' is not implemented, try df.apply-compatible method or use 'z'."
            )

    if const_name is not None: # insert const back
        df.insert(const_pos, const_name, const) 

    return df


def read_zip(zip_file_path, 
             use_file_name=None, 
             use_file_index=0, 
             print_file_names=False,
             return_io=False,
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
        if return_io:
            return zf.open(fname)
        if '.xls' in fname.lower():
            return pd.read_excel(zf.open(fname), **kwargs)
        return pd.read_csv(zf.open(fname), **kwargs)


def take_until_timeout(iterator, 
                       max_elements=None, 
                       per_item_timeout=None, 
                       total_timeout=None, 
                       loop=None,
                       patch_loop=False):
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for

    if patch_loop:
        import nest_asyncio
        nest_asyncio.apply(loop)
    
    async def get_elements(start):
        elements = []
        for i in itertools.count():
            if max_elements is not None and i >= max_elements:
                break
            if total_timeout is not None and loop.time() - start > total_timeout:
                break
            try:
                item = await asyncio.wait_for(
                    asyncio.to_thread(next, iterator),
                    timeout=per_item_timeout
                )
            except asyncio.TimeoutError:
                break
            except StopIteration:
                break

            elements.append(item)
        return elements
    
    loop = asyncio.get_event_loop() if loop is None else loop
    return loop.run_until_complete(get_elements(loop.time()))