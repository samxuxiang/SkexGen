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

def find_file_id(file_or_save_folder):
    """
    The json file will have a pathname of the form

    /foo/bar/0040/00408502.json

    The save folder will have a pathname of the form

    /foo2/bar2/0040/00408502/

    We want to find what DeepCAD call the file id.  In this 
    example it's '0040/00408502'
    """
    parent = file_or_save_folder.parent
    return f"{parent.stem}/{file_or_save_folder.stem}"


def find_files_already_processed_in_sub_folder(sub_folder):
    already_processed_ids = set()
    for save_folder in sub_folder.glob("*"):
        file_id = find_file_id(save_folder)
        already_processed_ids.add(file_id)
    return already_processed_ids


def find_files_already_processed_in_output_folder(output_folder):
    already_processed_ids = set()
    for sub_folder in output_folder.glob("*"):
        already_processed_ids = set.union(
            already_processed_ids,
            find_files_already_processed_in_sub_folder(sub_folder)
        )
    return already_processed_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the containing DeepCAD data")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to write the output")
    parser.add_argument("--verbose", action="store_true", help="Print extra information about convertion failures")
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir()

    # Find the list of files which were already 
    # processed
    already_processed_ids = find_files_already_processed_in_output_folder(output_folder)
    
    # Pre-load all json data
    deepcad_json = []
    skexgen_obj = []
    data_folder = Path(args.data_folder)
    
    for i in range(100): # all 100 folders
        cur_in = data_folder / str(i).zfill(4)
        cur_out = output_folder / str(i).zfill(4)
        if not cur_out.exists():
            cur_out.mkdir()
        files = []
        for f in cur_in.glob("**/*.json"):
            file_id = find_file_id(f)
            if not file_id in already_processed_ids:
                files.append(f)
        deepcad_json += files
        skexgen_obj += [cur_out]*len(files)
        
    assert len(skexgen_obj) == len(deepcad_json), "JSON & OBJ length different"

    # Parallel convert to JSON & STL
    iter_data = zip(
        deepcad_json,
        skexgen_obj,
    )
   
    threads = 50  # number of threads in your computer
    convert_iter = Pool(threads).imap(convert_folder_parallel, iter_data) 
    for invalid in tqdm(convert_iter, total=len(deepcad_json)):
        if invalid is not None:
            if args.verbose:
                print(f'Error converting {invalid}...')
    
    



   
