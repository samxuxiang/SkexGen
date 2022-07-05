import os 
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_XYZ, gp_Ax3, gp_Trsf, gp_Pln
from OCC.Display.SimpleGui import init_display
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from pathlib import Path
import os 
import numpy as np 
from pathlib import Path
from geometry.obj_parser import OBJParser
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


def print_loop(loop):
    for curve in loop:
        if curve.type == 'arc':
            print (f"{curve.start_idx} {curve.mid_idx} {curve.center_idx} {curve.end_idx} {curve.is_outer}")
        elif curve.type == 'line':
            print (f"{curve.start_idx} {curve.end_idx} {curve.is_outer}")
        else:
            print (f"{curve.center_idx} {curve.radius_idx} {curve.is_outer}")


def round_float(point):
    point['x'] = round(point['x'], 9)
    point['y'] = round(point['y'], 9)
    point['z'] = round(point['z'], 9)
    return


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])



def plot(shape_list):
    pyqt5_display, start_display, add_menu, add_function_to_menu = init_display('qt-pyqt5')
    for shape in shape_list:
        pyqt5_display.DisplayShape(shape, update=True)
    start_display()


def write_stl_file(a_shape, filename, mode="ascii", linear_deflection=0.001, angular_deflection=0.5):
    """ export the shape to a STL file
    Be careful, the shape first need to be explicitely meshed using BRepMesh_IncrementalMesh
    a_shape: the topods_shape to export
    filename: the filename
    mode: optional, "ascii" by default. Can either be "binary"
    linear_deflection: optional, default to 0.001. Lower, more occurate mesh
    angular_deflection: optional, default to 0.5. Lower, more accurate_mesh
    """
    if a_shape.IsNull():
        raise AssertionError("Shape is null.")
    if mode not in ["ascii", "binary"]:
        raise AssertionError("mode should be either ascii or binary")
    if os.path.isfile(filename):
        print("Warning: %s file already exists and will be replaced" % filename)
    # first mesh the shape
    mesh = BRepMesh_IncrementalMesh(a_shape, linear_deflection, False, angular_deflection, True)
    #mesh.SetDeflection(0.05)
    mesh.Perform()
    if not mesh.IsDone():
        raise AssertionError("Mesh is not done.")

    stl_exporter = StlAPI_Writer()
    if mode == "ascii":
        stl_exporter.SetASCIIMode(True)
    else:  # binary, just set the ASCII flag to False
        stl_exporter.SetASCIIMode(False)
    stl_exporter.Write(a_shape, filename)

    if not os.path.isfile(filename):
        raise IOError("File not written to disk.")



def same_plane(plane1, plane2):
    same = True 
    trans1 = plane1['pt']
    trans2 = plane2['pt']
    for key in trans1.keys():
        v1 = trans1[key]
        v2 = trans2[key]
        if v1['x'] != v2['x'] or v1['y'] != v2['y'] or v1['z'] != v2['z']:
            same = False 
    return same 


def create_xyz(xyz):
    return gp_XYZ(
        xyz["x"],
        xyz["y"],
        xyz["z"]
    )


def get_ax3(transform_dict):
    origin = create_xyz(transform_dict["origin"])
    x_axis = create_xyz(transform_dict["x_axis"])
    y_axis = create_xyz(transform_dict["y_axis"])
    z_axis = create_xyz(transform_dict["z_axis"])
    # Create new coord (orig, Norm, x-axis)
    axis3 = gp_Ax3(gp_Pnt(origin), gp_Dir(z_axis), gp_Dir(x_axis)) 
    return axis3


def get_transform(transform_dict):
    axis3 = get_ax3(transform_dict)
    transform_to_local = gp_Trsf()
    transform_to_local.SetTransformation(axis3) 
    return transform_to_local.Inverted()


def create_sketch_plane(transform_dict):
    axis3 = get_ax3(transform_dict)
    return gp_Pln(axis3)


def create_point(point_dict, transform):
    pt2d = gp_Pnt(
        point_dict["x"],
        point_dict["y"],
        point_dict["z"]
    )
    return pt2d.Transformed(transform)


def create_vector(vec_dict, transform):
    vec2d = gp_Vec(
        vec_dict["x"],
        vec_dict["y"],
        vec_dict["z"]
    )
    return vec2d.Transformed(transform)


def create_unit_vec(vec_dict, transform):
    vec2d = gp_Dir(
        vec_dict["x"],
        vec_dict["y"],
        vec_dict["z"]
    )
    return vec2d.Transformed(transform)


def write_obj(file, curve_strings, curve_count, vertex_strings, vertex_count, extrude_info, refP_info):
    """Write an .obj file with the curves and verts"""
        
    with open(file, "w") as fh:
        # Write Meta info
        fh.write("# WaveFront *.obj file\n")
        fh.write(f"# Vertices: {vertex_count}\n")
        fh.write(f"# Curves: {curve_count}\n")
        fh.write("# ExtrudeOperation: "+extrude_info['set_op']+"\n")
        fh.write("\n")

        # Write vertex and curve
        fh.write(vertex_strings)
        fh.write("\n")
        fh.write(curve_strings)
        fh.write("\n")

        #Write extrude value 
        extrude_string = 'Extrude '
        for value in extrude_info['extrude_values']:
            extrude_string += str(value)+' '
        fh.write(extrude_string)
        fh.write("\n")

        #Write refe plane value 
        p_orig = parse3d(refP_info['pt']['origin'])
        x_axis = parse3d(refP_info['pt']['x_axis'])
        y_axis = parse3d(refP_info['pt']['y_axis'])
        z_axis = parse3d(refP_info['pt']['z_axis'])
        fh.write('T_origin '+p_orig)
        fh.write("\n")
        fh.write('T_xaxis '+x_axis)
        fh.write("\n")
        fh.write('T_yaxis '+y_axis)
        fh.write("\n")
        fh.write('T_zaxis '+z_axis)


def parse3d(point3d):
    x = point3d['x']
    y = point3d['y']
    z = point3d['z']
    return str(x)+' '+str(y)+' '+str(z)



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


def get_loop_bbox(loop):
    bbox = []
    for curve in loop:
        bbox.append(curve.bbox)
    bbox_np = np.vstack(bbox)
    bbox_tight = [np.min(bbox_np[:,0]), np.max(bbox_np[:,1]), np.min(bbox_np[:,2]), np.max(bbox_np[:,3])]
    return np.asarray(bbox_tight) 


def get_face_bbox(face):
    bbox = []
    for loop in face:
        bbox.append(get_loop_bbox(loop))
    bbox_np = np.vstack(bbox)
    bbox_tight = [np.min(bbox_np[:,0]), np.max(bbox_np[:,1]), np.min(bbox_np[:,2]), np.max(bbox_np[:,3])]
    return np.asarray(bbox_tight) 


def sort_faces(sketch):
    bbox_list = []
    for face in sketch:
        bbox_list.append(get_face_bbox(face))
    bbox_list = np.vstack(bbox_list)
    min_values = bbox_list[:, [0,2]]
    # Sort by X min, then by Y min (increasing)
    ind = np.lexsort(np.transpose(min_values)[::-1])
    sorted_sketch = [sketch[x] for x in ind]
    return sorted_sketch


def sort_loops(face):
    assert face[0][0].is_outer
    if len(face) == 1:
        return face # only one outer loop, no need to sort 

    # Multiple loops in a face
    bbox_list = []
    inner_faces = face[1:]
    for loop_idx, loop in enumerate(inner_faces):
        assert not loop[0].is_outer # all loops are inner
        bbox_list.append(get_loop_bbox(loop))
    bbox_list = np.vstack(bbox_list)
    min_values = bbox_list[:, [0,2]]
    # Sort by X min, then by Y min (increasing)
    ind = np.lexsort(np.transpose(min_values)[::-1])
    
    # Outer + sorted inner 
    sorted_face = [face[0]] + [inner_faces[x] for x in ind]
    return sorted_face


def curve_connection(loop):
    adjM = np.zeros((len(loop), 2)) # 500 should be large enough
    for idx, curve in enumerate(loop):
        assert curve.type != 'circle'
        adjM[idx, 0] = curve.start_idx
        adjM[idx, 1] = curve.end_idx
    return adjM


def find_adj(index, adjM):
    loc_start = adjM[index][0]
    loc_end = adjM[index][1]
    matching_adj = np.where(adjM[:, 0]==loc_start)[0].tolist() + np.where(adjM[:, 0]==loc_end)[0].tolist() +\
                   np.where(adjM[:, 1]==loc_start)[0].tolist() + np.where(adjM[:, 1]==loc_end)[0].tolist()
    matching_adj = list(set(matching_adj) - set([index])) # unique, do not count itself 
    assert len(matching_adj) >= 1
    return matching_adj


def flip_curve(curve):
    tmp = curve.start_idx
    tmp2 = curve.start
    curve.start_idx = curve.end_idx
    curve.start = curve.end
    curve.end_idx = tmp 
    curve.end = tmp2 


def sort_start_end(sorted_loop):
    prev_curve = sorted_loop[0]
    for next_curve in sorted_loop[1:]:
        if prev_curve.end_idx != next_curve.start_idx:

            shared_idx = list(set([prev_curve.start_idx, prev_curve.end_idx]).intersection(
                set([next_curve.start_idx, next_curve.end_idx])
            )) 
            
            # back to itself
            if len(shared_idx) == 2:
                flip_curve(next_curve) 

            else:
                assert len(shared_idx) == 1
                shared_idx = shared_idx[0]
                if prev_curve.end_idx != shared_idx:
                    flip_curve(prev_curve)
                if next_curve.start_idx != shared_idx:
                    flip_curve(next_curve)
        prev_curve = next_curve 

    return
    
    
def print_loop(loop):
    for curve in loop:
        if curve.type == 'arc':
            print (f"{curve.start_idx} {curve.mid_idx} {curve.center_idx} {curve.end_idx} {curve.is_outer}")
        elif curve.type == 'line':
            print (f"{curve.start_idx} {curve.end_idx} {curve.is_outer}")
        else:
            print (f"{curve.center_idx} {curve.radius_idx} {curve.is_outer}")


def sort_curves(loop):
    """ sort loop start / end vertex """
    if len(loop) == 1:
        assert loop[0].type == 'circle'
        return loop # no need to sort circle

    curve_bbox = []
    for curve in loop:
        curve_bbox.append(curve.bbox)
    curve_bbox = np.vstack(curve_bbox)
    min_values = curve_bbox[:, [0,2]]
    # Sort by X min, then by Y min (increasing)
    ind = np.lexsort(np.transpose(min_values)[::-1])

    # Start from bottom left 
    start_curve_idx = ind[0]
    sorted_idx = [start_curve_idx]
    curve_adjM = curve_connection(loop)

    iter = 0
    while True:
        # Determine next curve
        matching_adj = find_adj(sorted_idx[-1], curve_adjM)
        matching_adj = list(set(matching_adj) - set(sorted_idx)) # remove exisiting ones

        if len(matching_adj) == 0:
            break # connect back to itself 

        if iter > 10000: # should be enough?
            raise Exception('fail to sort loop')

        # Second curve has two options, choose increasing x direction 
        if len(matching_adj) > 1: 
            bottom_left0 = loop[matching_adj[0]].bottom_left
            bottom_left1 = loop[matching_adj[1]].bottom_left
            if bottom_left1[0] > bottom_left0[0]:
                sorted_idx.append(matching_adj[1]) 
            else:
                sorted_idx.append(matching_adj[0]) 
        else:
            # Follow curve connection
            sorted_idx.append(matching_adj[0])

        iter += 1

    assert len(list(set(sorted_idx))) == len(sorted_idx) # no repeat
    sorted_loop = [loop[x] for x in sorted_idx]

    # Make sure curve end is next one's start 
    sort_start_end(sorted_loop)
    assert sorted_loop[0].start_idx == sorted_loop[-1].end_idx
    return sorted_loop


def quantize(data, n_bits=8, min_range=-1.0, max_range=1.0):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize) # clip values
    return data_quantize.astype('int32')

    
def parse_curve(line, curve, center, scale, command, bit):

    if line.type == 'line':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        curve.append(start)
        curve.append(np.array([-1,-1])) # -1 for curve end
        command.append(0)  # 0 for line 

    elif line.type == 'arc':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        mid = quantize((line.mid-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        curve.append(start)
        curve.append(mid)
        curve.append(np.array([-1,-1])) # -1 for curve end
        command.append(1)  # 1 for Arc 

    elif line.type == 'circle':
        pt1 = quantize((line.pt1-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt2 = quantize((line.pt2-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt3 = quantize((line.pt3-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt4 = quantize((line.pt4-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        curve.append(pt1)
        curve.append(pt2)
        curve.append(pt3)
        curve.append(pt4)
        curve.append(np.array([-1,-1])) # -1 for curve end
        command.append(2)  # 2 for circle 
    return


def convert_code(sketch, bit):
    """ convert to code format """

    # Sort faces in sketch based on min bbox coords (X->Y) 
    sorted_sketch = sort_faces(sketch)

    # Get normalization values
    vertex_total = []
    for face in sorted_sketch:
        sorted_face = sort_loops(face)
        for loop in sorted_face:
            sorted_loop = sort_curves(loop)
            vertices = get_vertex(sorted_loop)
            vertex_total.append(vertices)
    vertex_total = np.vstack(vertex_total)
    center_v, center = center_vertices(vertex_total)
    _, scale = normalize_vertices_scale(center_v)

    curve = []
    command = []
    # Multiple faces in a sketch
    for face in sorted_sketch:
        # Sort inner loops in face based on min bbox coords(X->Y)
        sorted_face = sort_loops(face)
        
        # Multiple loops in a face
        for loop in sorted_face:
            # Sort curves in a loop, 
            sorted_loop = sort_curves(loop)

            # XY coordinate
            for line in sorted_loop:
                parse_curve(line, curve, center, scale, command, bit)

            curve.append(np.array([-2,-2])) # -2 for loop end
            command.append(-1)

        curve.append(np.array([-3,-3])) # -3 for face end
        command.append(-2)

    curve.append(np.array([-4,-4])) # -4 for sketch end
    command.append(-3)
    curve_xy = np.vstack(curve).astype(int)
    command = np.array(command).astype(int)
    
    # Pixel index
    pixel_index = [] # x first then y
    for xy in curve_xy:
        if xy[0] < 0: 
            pixel_index.append(xy[0])
            continue 
        pixel_index.append(xy[1]*(2**bit)+xy[0])
    pixel_flat = np.array(pixel_index).astype(int).ravel()
    assert len(curve_xy) == len(pixel_flat)

    return curve_xy, pixel_flat, center, scale, command


def get_vertex(loop):
    """
    Fetch all vertex in curvegen style
    """
    vertices = []
    for curve in loop:
        if curve.type == 'line':
            vertices.append(curve.start)
            vertices.append(curve.end)
        if curve.type == 'arc':
            vertices.append(curve.start)
            vertices.append(curve.mid)
            vertices.append(curve.end)
        if curve.type == 'circle':
            vertices.append(curve.pt1)
            vertices.append(curve.pt2)
            vertices.append(curve.pt3)
            vertices.append(curve.pt4)
    vertices = np.vstack(vertices)
    return vertices
                
             
def center_vertices(vertices):
  """Translate the vertices so that bounding box is centered at zero."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vertices - vert_center, vert_center


def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = 0.5*np.sqrt(np.sum(extents**2))  # -1 ~ 1
  return vertices / scale, scale


def process_obj_se(data):
    """Load a sequence of obj files and convert to vector format."""
    project_folder, bit = data
    obj_files = find_files(project_folder, '.obj')
    if len(obj_files) == 0:
        return []

    se_xy = []
    se_pix = []
    se_ext = []
    se_cmd = []

    for obj_file in obj_files:
        # Read in the obj file 
        parser = OBJParser(Path(obj_file)) 
        _, sketch, meta_info = parser.parse_file(1.0)  
       
        # Convert it to a vector code 
        try:
            xy_coords, pixel_coords, center, scale, command = convert_code(sketch, bit)
        except Exception as ex:
            print(f'Can not convert code for {str(obj_file.parent)}')
            return []

        xy_coords += COORD_PAD # smallest is 0
        pixel_coords += PIX_PAD # smallest is 0
        command += CMD_PAD # smalles is 0
    
        # Set operation 
        set_op = meta_info['set_op']
        if set_op == 'JoinFeatureOperation' or set_op == 'NewBodyFeatureOperation':
            extrude_op = 1 #'add'
        elif set_op == 'CutFeatureOperation':
            extrude_op = 2 #'cut'
        elif set_op == 'IntersectFeatureOperation':
            extrude_op = 3 #'intersect'
        ext_type = np.array([extrude_op])
        
        # Extrude values
        ext_v = quantize(np.array(meta_info['extrude_value']), n_bits=bit, min_range=-EXTRUDE_R, max_range=+EXTRUDE_R)
        ext_v += EXT_PAD

        # Transformation origin
        ext_T =  quantize(np.array(meta_info['t_orig']), n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        ext_T += EXT_PAD
        
        # Transformation rotation
        ext_TX = np.clip(np.rint(np.array(meta_info['t_x'])).astype(int), -1, 1) + R_PAD    # -1 / 0 / 1 => 1 / 2 / 3
        ext_TY = np.clip(np.rint(np.array(meta_info['t_y'])).astype(int), -1, 1) + R_PAD
        ext_TZ = np.clip(np.rint(np.array(meta_info['t_z'])).astype(int), -1, 1) + R_PAD
        ext_R = np.concatenate([ext_TX, ext_TY, ext_TZ]) 
  
        # Scale and offset of the normalized sketch
        scale_quan = quantize(scale, n_bits=bit, min_range=0.0, max_range=SCALE_R)
        scale_quan += EXT_PAD
        offset_quan = quantize(center, n_bits=bit, min_range=-OFFSET_R, max_range=OFFSET_R)
        offset_quan += EXT_PAD
        extrude_param = np.concatenate([ext_v, ext_T, ext_R, ext_type, 
                        np.array([scale_quan]).astype(int), offset_quan, np.array([0]).astype(int)]) 

        se_xy.append(xy_coords)
        se_ext.append(extrude_param)
        se_pix.append(pixel_coords)
        se_cmd.append(command)

    len_xy = np.asarray([len(x) for x in se_xy]).sum()
    len_ext = np.asarray([len(x) for x in se_ext]).sum()
    len_pix = np.asarray([len(x) for x in se_pix]).sum()
    len_cmd = np.asarray([len(x) for x in se_cmd]).sum()
    num_se = len(se_xy)

    try:
        assert len(se_xy) == len(se_ext)
        assert len(se_xy) == len(se_pix)
        assert len(se_xy) == len(se_cmd)
    except Exception as ex:
        print(f'Length mismatch for {str(obj_file.parent)}')
        return []

    data = {'name': str(obj_file.parent)[-13:],
            'len_xy':  len_xy,
            'len_ext': len_ext,
            'len_pix': len_pix,
            'len_cmd': len_cmd,
            'num_se': num_se,   
            'se_xy': se_xy,
            'se_cmd': se_cmd,
            'se_pix': se_pix,
            'se_ext': se_ext,
            }

    return [data]


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
        x=np.linspace(0, 2**bit-1, 2**bit)
        y=np.linspace(0, 2**bit-1, 2**bit)
        xx,yy=np.meshgrid(x,y)
        self.vertices = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
        self.vertex_dict = OrderedDict()
        self.bit = bit
    

    def perform(self, tokens):     
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
                        curve = curve[:-1] - PIX_PAD - EXTRA_PAD # remove padding
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