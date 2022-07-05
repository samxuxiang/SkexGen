"""
Create solids by the DeepCAD dataset
"""
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from converter import DeepCADReconverter
import signal
from contextlib import contextmanager

# Time out function
@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        raise Exception("time out")
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
def raise_timeout(signum, frame):
    raise TimeoutError


def load_json_data(pathname):
    """Load data from a json file"""
    with open(pathname, encoding='utf8') as data_file:
        return json.load(data_file)


def convert_folder_parallel(data):
    fileName, output_folder = data
    save_fileFolder = Path(output_folder) / fileName.stem
    if not save_fileFolder.exists():
        save_fileFolder.mkdir()

    data = load_json_data(fileName)
    reconverter = DeepCADReconverter(data, fileName)

    try:
        with timeout(30):
            reconverter.parse_json(save_fileFolder)
    except Exception as ex:
        return [fileName, str(ex)[:50]]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the containing DeepCAD data")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to write the output")
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir()
    
    # Pre-load all json data
    deepcad_json = []
    skexgen_obj = []
    data_folder = Path(args.data_folder)
    
    for i in range(100): # all 100 folders
        cur_in = data_folder / str(i).zfill(4)
        cur_out = output_folder / str(i).zfill(4)
        if not cur_out.exists():
            cur_out.mkdir()
        files = [ f for f in cur_in.glob("**/*.json")]
        deepcad_json += files
        skexgen_obj += [cur_out]*len(files)
        
    assert len(skexgen_obj) == len(deepcad_json), "JSON & OBJ length different"

    # Parallel convert to JSON & STL
    iter_data = zip(
        deepcad_json,
        skexgen_obj,
    )
   
    threads = 36  # number of threads in your PC
    convert_iter = Pool(threads).imap(convert_folder_parallel, iter_data) 
    for invalid in tqdm(convert_iter, total=len(deepcad_json)):
        ### Surpress Warnings ###
        # if invalid is not None:
        #     print(f'Error converting {invalid}...')
        pass
    
    



   
