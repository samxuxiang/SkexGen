import meshio
import os 
import signal
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
from geometry.obj_parser import OBJParser
from converter import OBJReconverter
import argparse
from utils import find_files
from pathlib import Path
from OCC.Core.BRepCheck import BRepCheck_Analyzer
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


class NormalizeSE:

    def __init__(self, cube_size, norm_factor, extrude_size, sketch_size):
        self.MR = cube_size
        self.ER = extrude_size
        self.F = norm_factor
        self.SR = sketch_size
    

    def flatten(self, t):
        return [item for sublist in t for item in sublist]


    def update_bbox(self, bbox_s, bbox_t):
        if bbox_s[0] > bbox_t[0]:
            bbox_s[0] = bbox_t[0]
        if bbox_s[1] < bbox_t[1]:
            bbox_s[1] = bbox_t[1]
        if bbox_s[2] > bbox_t[2]:
            bbox_s[2] = bbox_t[2]
        if bbox_s[3] < bbox_t[3]:
            bbox_s[3] = bbox_t[3]
        return bbox_s

    
    def normalize(self, stl_files, extrude_param, output_folder):
        """
        find best normalization scale,
        normalize cad mesh and modify the .obj file accordingly
        """
        # Normalize mesh
        verts = []
        for stl in stl_files:
            # Load mesh 
            mesh = meshio.read(str(stl))
            vertex = mesh.points.copy() 
            verts.append(vertex)
        all_verts = np.vstack(verts)
        mesh_scale = self.MR / np.max(np.abs(all_verts))
        
        # Reduce size to satisfy extrude
        scales = []
        for obj in extrude_param:
            # Load sketch and extrude parameters
            parser = OBJParser(obj)
            _, _, meta_info = parser.parse_file(mesh_scale)

            # Extrude value too large 
            extrudes = [abs(x) for x in meta_info['extrude_value']]
            if max(extrudes) > self.ER:
                scale = (self.ER) / max(extrudes)
                scales.append(scale)

            # Extrude translation too large
            origins = [abs(x) for x in meta_info['t_orig']]
            if max(origins) > self.ER:
                scale = (self.ER) / max(origins)
                scales.append(scale)
        
        if len(scales) == 0:
            obj_scale = mesh_scale
        else:
            obj_scale = mesh_scale * min(scales)  

        # Reduce size to satisfy sketch
        scales = []
        for obj in extrude_param:
            parser = OBJParser(obj)
            sketch_v, faces, meta_info = parser.parse_file(obj_scale)
            
            # Compute vertex bbox size
            final_bbox = [999, -999, 999, -999]
            for curve in self.flatten(self.flatten(faces)):
                final_bbox = self.update_bbox(final_bbox, curve.bbox) 

            # Rescale vertex bbox to sketch range
            if max([abs(x) for x in final_bbox]) > self.F*self.SR:
                scale = (self.F*self.SR) / max([abs(x) for x in final_bbox])
                scales.append(scale)

        final_scale = obj_scale
        if len(scales) > 0:
            final_scale *= min(scales) 

        # Reconstruct the normalized brep 
        cur_solid = None
        extrude_idx = 0
        for obj in extrude_param:
            parser = OBJParser(obj)
            sketch_v, faces, meta_info = parser.parse_file(final_scale)
            converter = OBJReconverter()
            ext_solid, curve_str, curve_count = converter.parse_obj(faces, meta_info)
            v_str = converter.convert_vertices()
            
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

            # Save obj
            obj_out = os.path.join(output_folder, obj.stem+'.obj')
            parser.write_obj(obj_out, curve_str, curve_count, v_str, len(converter.vertex_dict), meta_info, scale=final_scale)
        assert len(extrude_param) == extrude_idx, "number of operations mismatch"
        return

         

def run_parallel(project_folder):
    """
    Parallel normalization
    """
    subfolder1 = project_folder.split('/')[-3]
    subfolder2 = project_folder.split('/')[-2]
    output_folder = os.path.join(args.out_folder, subfolder1, subfolder2)

    extrude_param = find_files(os.path.join(args.data_folder, subfolder1, subfolder2), 'param.obj') 
    stl_files =  find_files(project_folder, 'extrude.stl') 
    converter = NormalizeSE(cube_size=5.0, norm_factor=0.98, extrude_size=1.0, sketch_size=1.0)  
    if len(stl_files) == 0:
        return 'Empty folder, skipping'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    msg = []
    try:
        with timeout(60):
            converter.normalize(stl_files, extrude_param, output_folder)
    except Exception as ex:
        msg = [output_folder, str(ex)[:50]]
    finally:
        return msg
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    project_folders = []
    for idx in range(100): # all 100 folder
        cur_dir = data_folder / str(idx).zfill(4)
        project_folders += glob(str(cur_dir)+'/*/')

    # Parallel
    threads = 36  # number of threads in your pc 
    convert_iter = Pool(threads).imap(run_parallel, project_folders)
    for msg in tqdm(convert_iter, total=len(project_folders)):
        ### Surpress Warnings ###
        # if len(msg)>0:
        #     print(f'Normalization Error: {msg}')
        pass
        
    

    

    