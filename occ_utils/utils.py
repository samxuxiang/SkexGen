import os 
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_XYZ, gp_Ax3, gp_Trsf, gp_Pln
from OCC.Display.SimpleGui import init_display
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from pathlib import Path


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

