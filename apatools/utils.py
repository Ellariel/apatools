import numpy as np
import pandas as pd
import itertools
import asyncio
import zipfile



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