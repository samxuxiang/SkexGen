import os 
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob 
from converter import OBJReconverter
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from geometry.obj_parser import OBJParser
from utils import write_stl_file
from OCC.Extend.DataExchange import write_step_file 

import signal
from contextlib import contextmanager
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
    
NUM_TRHEADS = 36 

def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def run_parallel(project_folder):
    output_folder = project_folder

    param_objs = find_files(project_folder, 'param.obj')

    cur_solid = None
    extrude_idx = 0
    for obj in param_objs:
        try:
          with timeout(30):
            parser = OBJParser(obj)
            _, faces, meta_info = parser.parse_file(1.0)
            converter = OBJReconverter()
            ext_solid, _, _ = converter.parse_obj(faces, meta_info)
            set_op = meta_info["set_op"]
            if set_op == "NewBodyFeatureOperation" or set_op == "JoinFeatureOperation":
                if cur_solid is None:
                    cur_solid = ext_solid
                else:
                    cur_solid = converter.my_op(cur_solid, ext_solid, 'fuse')
            elif set_op == "CutFeatureOperation":
                cur_solid = converter.my_op(cur_solid, ext_solid, 'cut')
            elif set_op == "IntersectFeatureOperation":
                cur_solid = converter.my_op(cur_solid, ext_solid, 'common')
            else:
                raise Exception("Unknown operation type")

            analyzer = BRepCheck_Analyzer(cur_solid)
            if not analyzer.IsValid():
                raise Exception("brep check failed")
            
            extrude_idx += 1
        
        except Exception as ex:
            msg = [project_folder, str(ex)[:100]]
            return None 
 
    try:
      with timeout(30):
        stl_name = Path(output_folder).stem + '_'+ str(extrude_idx).zfill(3) + "_final.stl"
        output_path =  os.path.join(output_folder, stl_name)
        write_stl_file(cur_solid, output_path, linear_deflection=0.001, angular_deflection=0.5)

        step_name = Path(output_folder).stem + '_'+ str(extrude_idx).zfill(3) + "_final.step"
        output_path =  os.path.join(output_folder, step_name)
        write_step_file(cur_solid, output_path)

    except Exception as ex:
        msg = [project_folder, str(ex)[:500]]
        return None 

    return cur_solid 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    args = parser.parse_args()

    solids = []
    cad_folders = sorted(glob(args.data_folder+'/*/'))

    convert_iter = Pool(NUM_TRHEADS).imap(run_parallel, cad_folders) 
    for solid in tqdm(convert_iter, total=len(cad_folders)):
        pass
