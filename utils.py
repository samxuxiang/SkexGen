import os 
import numpy as np 
from pathlib import Path
import math 
from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR

SKETCH_R = 1
RADIUS_R = 1
EXTRUDE_R = 1.0
SCALE_R =  1.4
OFFSET_R = 0.9
PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)



def dequantize_verts(verts, n_bits=8, min_range=-0.5, max_range=0.5, add_noise=False):
  """Convert quantized vertices to floats."""
  range_quantize = 2**n_bits - 1
  verts = verts.astype('float32')
  verts = verts * (max_range - min_range) / range_quantize + min_range
  return verts

def quantize(data, n_bits=8, min_range=-1.0, max_range=1.0):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize) # clip values
    return data_quantize.astype('int32')


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def find_files_path(folder, extension):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)])


def rads_to_degs(rads):
    return 180*rads/math.pi


def angle_from_vector_to_x(vec):
    assert vec.size == 2
    # We need to find a unit vector
    angle = 0.0

    l = np.linalg.norm(vec)
    uvec = vec/l

    # 2 | 1
    #-------
    # 3 | 4
    if uvec[0] >=0:
        if uvec[1] >= 0:
            # Qadrant 1
            angle = math.asin(uvec[1])
        else:
            # Qadrant 4
            angle = 2.0*math.pi - math.asin(-uvec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(uvec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-uvec[1])
    return angle


def find_arc_geometry(a, b, c):
        A = b[0] - a[0] 
        B = b[1] - a[1]
        C = c[0] - a[0]
        D = c[1] - a[1]
    
        E = A*(a[0] + b[0]) + B*(a[1] + b[1])
        F = C*(a[0] + c[0]) + D*(a[1] + c[1])
    
        G = 2.0*(A*(c[1] - b[1])-B*(c[0] - b[0])) 

        if G == 0:
            raise Exception("zero G")

        p_0 = (D*E - B*F) / G
        p_1 = (A*F - C*E) / G

        center = np.array([p_0, p_1])
        radius = np.linalg.norm(center - a)

        angles = []
        for xx in [a,b,c]:
            angle = angle_from_vector_to_x(xx - center)
            angles.append(angle)

        ab = b-a
        ac = c-a
        cp = np.cross(ab, ac)
        if cp >= 0:
            start_angle_rads = angles[0]
            end_angle_rads = angles[2]
        else:
            start_angle_rads = angles[2]
            end_angle_rads = angles[0]

        return center, radius, start_angle_rads, end_angle_rads



class CADparser:
    """ Parse into OBJ files """
    def __init__(self, bit):
        self.bit = bit
        x=np.linspace(0, 2**self.bit-1, 2**self.bit)
        y=np.linspace(0, 2**self.bit-1, 2**self.bit)
        xx,yy=np.meshgrid(x,y)
        self.vertices = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
        self.vertex_dict = OrderedDict()
    

    def perform(self, tokens_tuple):   
        tokens = tokens_tuple
        se_datas = []   
        # (0) Remove padding
        tokens = tokens[:np.where(tokens==0)[0][0]]

        # (1) Divide into sketch and extrude
        assert tokens[-1] == 1  # last one must be end token
        groups = np.split(tokens, np.where(tokens==1)[0]+1)
        assert len(groups[-1]) == 0 # last one is empty, remove
        groups = groups[:-1]
        assert len(groups) % 2 == 0
        sketches = groups[0::2] # odd index
        extrudes = groups[1::2] # even index
        
        # (2) Sequentially parse each pair of SE into obj 
        for (sketch, extrude) in zip(sketches, extrudes):
            assert extrude[-1] == 1
            # scale, offset, remove flag
            scale = extrude[-4]-EXT_PAD-EXTRA_PAD
            offset = extrude[-3:-1]-EXT_PAD-EXTRA_PAD
            scale = dequantize_verts(scale, n_bits=self.bit, 
                                     min_range=0.0, max_range=SCALE_R, add_noise=False)
            offset = dequantize_verts(offset, n_bits=self.bit, 
                                     min_range=-OFFSET_R, max_range=OFFSET_R, add_noise=False)

            assert sketch[-1] == 1
            assert extrude[-1] == 1
            sketch = sketch[:-1]
            extrude = extrude[:-1]
            
            faces = np.split(sketch, np.where(sketch==2)[0]+1)
            assert len(faces[-1]) == 0 
            faces = faces[:-1]

            # Each face
            se_str = ""
            for face_idx, face in enumerate(faces):
                face_str = "face\n"
                assert face[-1] == 2
                face = face[:-1]
                loops = np.split(face, np.where(face==3)[0]+1)
                assert len(loops[-1]) == 0
                loops = loops[:-1]

                # Each loop
                for loop_idx, loop in enumerate(loops):
                    assert loop[-1] == 3
                    loop = loop[:-1]
                    curves = np.split(loop, np.where(loop==4)[0]+1)
                    assert len(curves[-1]) == 0
                    curves = curves[:-1]

                    loop_curves = []
                    for curve in curves:
                        assert curve[-1] == 4
                        curve = curve[:-1]-PIX_PAD-EXTRA_PAD # remove padding
                        loop_curves.append(curve)

                    # Draw a single loop curves
                    next_loop_curves = loop_curves[1:]
                    next_loop_curves += loop_curves[:1]
                    
                    cur_str = []
                    for cur, next_cur in zip(loop_curves, next_loop_curves):
                        self.obj_curve(cur, next_cur, cur_str, scale, offset)
                    
                    loop_string = ""
                    for cur in cur_str:
                        loop_string += f"{cur}\n"
                    
                    if loop_idx == 0:
                        face_str += f"out\n{loop_string}\n"
                    else:
                        face_str += f"in\n{loop_string}\n"

                se_str += face_str 

            vertex_str = self.convert_vertices() 

            # (3) Convert extrusion parameters
            extrude_value = dequantize_verts(extrude[0:2]-EXT_PAD-EXTRA_PAD, n_bits=self.bit, 
                                             min_range=-EXTRUDE_R, max_range=EXTRUDE_R, add_noise=False)
            extrude_T = dequantize_verts(extrude[2:5]-EXT_PAD-EXTRA_PAD, n_bits=self.bit, 
                                         min_range=-EXTRUDE_R, max_range=EXTRUDE_R, add_noise=False)
            extrude_R = extrude[5:14] - R_PAD - EXTRA_PAD
            extrude_op = extrude[14] - EXTRA_PAD
            extrude_param = {'value': extrude_value, 
                            'R': extrude_R,
                            'T': extrude_T,
                            'op': extrude_op}

            se_data = {'vertex': vertex_str, 'curve': se_str, 'extrude': extrude_param}
            se_datas.append(se_data)
            self.vertex_dict.clear()

        return se_datas


    def obj_curve(self, curve, next_curve, cur_str, scale, offset):

        if len(curve) == 4: # Circle
            assert len(list(set(np.unique(curve))-set(curve))) == 0
            p1 = dequantize_verts(self.vertices[curve[0]], n_bits=self.bit, 
                                  min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)
            p2 = dequantize_verts(self.vertices[curve[1]], n_bits=self.bit, 
                                  min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)
            p3 = dequantize_verts(self.vertices[curve[2]], n_bits=self.bit, 
                                  min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)
            p4 = dequantize_verts(self.vertices[curve[3]], n_bits=self.bit, 
                                  min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)

            center = np.asarray([0.5*(p1[0]+p2[0]),  0.5*(p3[1]+p4[1])])
            radius = (np.linalg.norm(p1-p2) + np.linalg.norm(p3-p4))/4.0

            center = center*scale + offset
            radius = radius*scale

            center_idx = self.save_vertex(center[0], center[1], 'p')
            radius_idx = self.save_vertex(radius, 0.0, 'r')
            cur_str.append(f"c {center_idx} {radius_idx}")

        elif len(curve) == 2: # Arc
            assert curve[0] != curve[1]
            assert curve[0] != next_curve[0]
            assert curve[1] != next_curve[0]
            start_v = dequantize_verts(self.vertices[curve[0]], n_bits=self.bit, 
                                       min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)
            mid_v = dequantize_verts(self.vertices[curve[1]], n_bits=self.bit, 
                                     min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)
            end_v = dequantize_verts(self.vertices[next_curve[0]], n_bits=self.bit, 
                                     min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)
            center, _, _, _ = find_arc_geometry(start_v, mid_v, end_v)

            start_v = start_v*scale + offset
            mid_v = mid_v*scale + offset
            end_v = end_v*scale + offset 
            center = center*scale + offset 
       
            center_idx = self.save_vertex(center[0], center[1], 'p')
            start_idx = self.save_vertex(start_v[0], start_v[1], 'p')
            mid_idx = self.save_vertex(mid_v[0], mid_v[1], 'p')
            end_idx = self.save_vertex(end_v[0], end_v[1], 'p')
            cur_str.append(f"a {start_idx} {mid_idx} {center_idx} {end_idx}")

        elif len(curve) == 1: # Line
            assert curve[0] != next_curve[0]
            start_v = dequantize_verts(self.vertices[curve[0]], n_bits=self.bit, 
                                       min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)
            end_v = dequantize_verts(self.vertices[next_curve[0]], n_bits=self.bit, 
                                     min_range=-SKETCH_R, max_range=SKETCH_R, add_noise=False)

            start_v = start_v*scale + offset
            end_v = end_v*scale + offset 

            start_idx = self.save_vertex(start_v[0], start_v[1], 'p')
            end_idx = self.save_vertex(end_v[0], end_v[1], 'p')
            cur_str.append(f"l {start_idx} {end_idx}")

        else:
            assert False
                                            

    def save_vertex(self, h_x, h_y, text):
        unique_key = f"{text}:x{h_x}y{h_y}"
        index = 0
        for key in self.vertex_dict.keys():
            # Vertex location already exist in dict
            if unique_key == key: 
                return index 
            index += 1
        # Vertex location does not exist in dict
        self.vertex_dict[unique_key] = [h_x, h_y]
        return index


    def convert_vertices(self):
        """ Convert all the vertices to .obj format """
        vertex_strings = ""
        for pt in self.vertex_dict.values():
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt[0]} {pt[1]}\n"
            vertex_strings += vertex_string
        return vertex_strings


def write_obj(save_folder, data):
    for idx, write_data in enumerate(data):
        obj_name = Path(save_folder).stem + '_'+ str(idx).zfill(3) + "_param.obj"
        obj_file = Path(save_folder) / obj_name
        extrude_param = write_data['extrude']
        vertex_strings = write_data['vertex']
        curve_strings = write_data['curve']

        """Write an .obj file with the curves and verts"""
        if extrude_param['op'] == 1: #'add'
            set_op = 'NewBodyFeatureOperation'
        elif extrude_param['op'] == 2: #'cut'
            set_op = 'CutFeatureOperation'
        elif extrude_param['op'] == 3: #'cut'
            set_op = 'IntersectFeatureOperation'

        with open(obj_file, "w") as fh:
            # Write Meta info
            fh.write("# WaveFront *.obj file\n")
            fh.write("# ExtrudeOperation: "+set_op+"\n")
            fh.write("\n")

            # Write vertex and curve
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)
            fh.write("\n")

            #Write extrude value 
            extrude_string = 'Extrude '
            for value in extrude_param['value']:
                extrude_string += str(value)+' '
            fh.write(extrude_string)
            fh.write("\n")

            #Write refe plane value 
            p_orig = parse3d(extrude_param['T'])
            x_axis = parse3d(extrude_param['R'][0:3])
            y_axis = parse3d(extrude_param['R'][3:6])
            z_axis = parse3d(extrude_param['R'][6:9])
            fh.write('T_origin '+p_orig)
            fh.write("\n")
            fh.write('T_xaxis '+x_axis)
            fh.write("\n")
            fh.write('T_yaxis '+y_axis)
            fh.write("\n")
            fh.write('T_zaxis '+z_axis)


def parse3d(point3d):
    x = point3d[0]
    y = point3d[1]
    z = point3d[2]
    return str(x)+' '+str(y)+' '+str(z)

