import os, sys, argparse, math, base64, zlib, struct

if sys.version_info < (3,0):
    range = xrange

vertices = []
numVertices = [32, 144, 32, 144]
vertices_base64 = "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"
triangles = [[ 0, 2, 1, 0, 3, 2, 0, 1, 4, 4, 1, 5, 1, 2, 5, 5, 2, 6, 2, 3, 6, 6, 3, 7, 3, 0, 7, 7, 0, 4, 4, 5, 6, 4, 6, 7, 8, 10, 9, 8, 11, 10, 8, 9, 12, 12, 9, 13, 9, 10, 13, 13, 10, 14, 10, 11, 14, 14, 11, 15, 11, 8, 15, 15, 8, 12, 12, 13, 14, 12, 14, 15, 16, 18, 17, 16, 19, 18, 16, 17, 20, 20, 17, 21, 17, 18, 21, 21, 18, 22, 18, 19, 22, 22, 19, 23, 19, 16, 23, 23, 16, 20, 20, 21, 22, 20, 22, 23, 24, 26, 25, 24, 27, 26, 24, 25, 28, 28, 25, 29, 25, 26, 29, 29, 26, 30, 26, 27, 30, 30, 27, 31, 27, 24, 31, 31, 24, 28, 28, 29, 30, 28, 30, 31], [ 0, 1, 72, 1, 73, 72, 36, 108, 37, 37, 108, 109, 1, 2, 73, 2, 74, 73, 37, 109, 38, 38, 109, 110, 2, 3, 74, 3, 75, 74, 38, 110, 39, 39, 110, 111, 3, 4, 75, 4, 76, 75, 39, 111, 40, 40, 111, 112, 4, 5, 76, 5, 77, 76, 40, 112, 41, 41, 112, 113, 5, 6, 77, 6, 78, 77, 41, 113, 42, 42, 113, 114, 6, 7, 78, 7, 79, 78, 42, 114, 43, 43, 114, 115, 7, 8, 79, 8, 80, 79, 43, 115, 44, 44, 115, 116, 8, 9, 80, 9, 81, 80, 44, 116, 45, 45, 116, 117, 9, 10, 81, 10, 82, 81, 45, 117, 46, 46, 117, 118, 10, 11, 82, 11, 83, 82, 46, 118, 47, 47, 118, 119, 11, 12, 83, 12, 84, 83, 47, 119, 48, 48, 119, 120, 12, 13, 84, 13, 85, 84, 48, 120, 49, 49, 120, 121, 13, 14, 85, 14, 86, 85, 49, 121, 50, 50, 121, 122, 14, 15, 86, 15, 87, 86, 50, 122, 51, 51, 122, 123, 15, 16, 87, 16, 88, 87, 51, 123, 52, 52, 123, 124, 16, 17, 88, 17, 89, 88, 52, 124, 53, 53, 124, 125, 17, 18, 89, 18, 90, 89, 53, 125, 54, 54, 125, 126, 18, 19, 90, 19, 91, 90, 54, 126, 55, 55, 126, 127, 19, 20, 91, 20, 92, 91, 55, 127, 56, 56, 127, 128, 20, 21, 92, 21, 93, 92, 56, 128, 57, 57, 128, 129, 21, 22, 93, 22, 94, 93, 57, 129, 58, 58, 129, 130, 22, 23, 94, 23, 95, 94, 58, 130, 59, 59, 130, 131, 23, 24, 95, 24, 96, 95, 59, 131, 60, 60, 131, 132, 24, 25, 96, 25, 97, 96, 60, 132, 61, 61, 132, 133, 25, 26, 97, 26, 98, 97, 61, 133, 62, 62, 133, 134, 26, 27, 98, 27, 99, 98, 62, 134, 63, 63, 134, 135, 27, 28, 99, 28, 100, 99, 63, 135, 64, 64, 135, 136, 28, 29, 100, 29, 101, 100, 64, 136, 65, 65, 136, 137, 29, 30, 101, 30, 102, 101, 65, 137, 66, 66, 137, 138, 30, 31, 102, 31, 103, 102, 66, 138, 67, 67, 138, 139, 31, 32, 103, 32, 104, 103, 67, 139, 68, 68, 139, 140, 32, 33, 104, 33, 105, 104, 68, 140, 69, 69, 140, 141, 33, 34, 105, 34, 106, 105, 69, 141, 70, 70, 141, 142, 34, 35, 106, 35, 107, 106, 70, 142, 71, 71, 142, 143, 35, 0, 107, 0, 72, 107, 71, 143, 36, 36, 143, 108], [ 0, 2, 1, 0, 3, 2, 0, 1, 4, 4, 1, 5, 1, 2, 5, 5, 2, 6, 2, 3, 6, 6, 3, 7, 3, 0, 7, 7, 0, 4, 4, 5, 6, 4, 6, 7, 8, 10, 9, 8, 11, 10, 8, 9, 12, 12, 9, 13, 9, 10, 13, 13, 10, 14, 10, 11, 14, 14, 11, 15, 11, 8, 15, 15, 8, 12, 12, 13, 14, 12, 14, 15, 16, 18, 17, 16, 19, 18, 16, 17, 20, 20, 17, 21, 17, 18, 21, 21, 18, 22, 18, 19, 22, 22, 19, 23, 19, 16, 23, 23, 16, 20, 20, 21, 22, 20, 22, 23, 24, 26, 25, 24, 27, 26, 24, 25, 28, 28, 25, 29, 25, 26, 29, 29, 26, 30, 26, 27, 30, 30, 27, 31, 27, 24, 31, 31, 24, 28, 28, 29, 30, 28, 30, 31], [ 0, 1, 72, 1, 73, 72, 36, 108, 37, 37, 108, 109, 1, 2, 73, 2, 74, 73, 37, 109, 38, 38, 109, 110, 2, 3, 74, 3, 75, 74, 38, 110, 39, 39, 110, 111, 3, 4, 75, 4, 76, 75, 39, 111, 40, 40, 111, 112, 4, 5, 76, 5, 77, 76, 40, 112, 41, 41, 112, 113, 5, 6, 77, 6, 78, 77, 41, 113, 42, 42, 113, 114, 6, 7, 78, 7, 79, 78, 42, 114, 43, 43, 114, 115, 7, 8, 79, 8, 80, 79, 43, 115, 44, 44, 115, 116, 8, 9, 80, 9, 81, 80, 44, 116, 45, 45, 116, 117, 9, 10, 81, 10, 82, 81, 45, 117, 46, 46, 117, 118, 10, 11, 82, 11, 83, 82, 46, 118, 47, 47, 118, 119, 11, 12, 83, 12, 84, 83, 47, 119, 48, 48, 119, 120, 12, 13, 84, 13, 85, 84, 48, 120, 49, 49, 120, 121, 13, 14, 85, 14, 86, 85, 49, 121, 50, 50, 121, 122, 14, 15, 86, 15, 87, 86, 50, 122, 51, 51, 122, 123, 15, 16, 87, 16, 88, 87, 51, 123, 52, 52, 123, 124, 16, 17, 88, 17, 89, 88, 52, 124, 53, 53, 124, 125, 17, 18, 89, 18, 90, 89, 53, 125, 54, 54, 125, 126, 18, 19, 90, 19, 91, 90, 54, 126, 55, 55, 126, 127, 19, 20, 91, 20, 92, 91, 55, 127, 56, 56, 127, 128, 20, 21, 92, 21, 93, 92, 56, 128, 57, 57, 128, 129, 21, 22, 93, 22, 94, 93, 57, 129, 58, 58, 129, 130, 22, 23, 94, 23, 95, 94, 58, 130, 59, 59, 130, 131, 23, 24, 95, 24, 96, 95, 59, 131, 60, 60, 131, 132, 24, 25, 96, 25, 97, 96, 60, 132, 61, 61, 132, 133, 25, 26, 97, 26, 98, 97, 61, 133, 62, 62, 133, 134, 26, 27, 98, 27, 99, 98, 62, 134, 63, 63, 134, 135, 27, 28, 99, 28, 100, 99, 63, 135, 64, 64, 135, 136, 28, 29, 100, 29, 101, 100, 64, 136, 65, 65, 136, 137, 29, 30, 101, 30, 102, 101, 65, 137, 66, 66, 137, 138, 30, 31, 102, 31, 103, 102, 66, 138, 67, 67, 138, 139, 31, 32, 103, 32, 104, 103, 67, 139, 68, 68, 139, 140, 32, 33, 104, 33, 105, 104, 68, 140, 69, 69, 140, 141, 33, 34, 105, 34, 106, 105, 69, 141, 70, 70, 141, 142, 34, 35, 106, 35, 107, 106, 70, 142, 71, 71, 142, 143, 35, 0, 107, 0, 72, 107, 71, 143, 36, 36, 143, 108]]
decoration = [[ 1450000.000000, 0.000000, 0.000000, 1450000.000000, 0.000000, 0.500000, -1450000.000000, 0.000000, 0.000000, -1450000.000000, 0.000000, 0.500000, 0.000000, 1450000.000000, 0.000000, 0.000000, 1450000.000000, 0.500000, 0.000000, -1450000.000000, 0.000000, 0.000000, -1450000.000000, 0.500000], [ ], [ 1450000.000000, 0.000000, 1.000000, 1450000.000000, 0.000000, 1.500000, -1450000.000000, 0.000000, 1.000000, -1450000.000000, 0.000000, 1.500000, 0.000000, 1450000.000000, 1.000000, 0.000000, 1450000.000000, 1.500000, 0.000000, -1450000.000000, 1.000000, 0.000000, -1450000.000000, 1.500000], [ ]]

color = [2, 8, 2, 8]

index_base64 = 'eJylVlGL4zYQft9foXMoeCGRkyulx24SyOYWbiHbW/aWwrX0QZHHsQ5bSiU5ubTcf+9I8ia2Ex8tNcSWNDPffDMjjTJ98/7j8uXz0z3JbVnMr6bhQ/CZ5sDSMPTTEixDLbsdwZ+V2M2ipZIWpB29HLYQER5ms8jCV5s4mFvCc6YN2Flls9G7iCTzqxOcFbaA+R1bHwol6RdDDCu3BSBOCtMkSE/ab0Yj0tAdjRoyw7XYWmI0n0WOn7lJEp5Kug76Xwzlqkw4k9IbR/NpEkz+M8b6yOAyiD00SbvH5WFI1io9kL9bAveoHeisUPsbkos0BXl7prEXqc1vyGQ8/uFcmIPY5LZPWjK9EfKGjM9FW5amQm7OZN+uWtOBBpmCXjK5Y+YC/f9BzqqK5yPGrVBIEesCXSavI0zzKavTJOzJqUtovcQDPZHOoiZfV6AgqvW61doxTWrbGUkVr0rcvXQD9r4AN7w7PKRxG/L6tmUNEhMMaC1hT+4Wd59XH3+h934xDshDYnUFaNb2qoFZ+MTBG2eV9Gkg8XUnx07X1FpNF94yDt6b2O5JkoWvLWGkcAU4y7wD9ZIO6AcohdnmoAVfOXEcuZVo2FL6FbhV+sd4PCSTIRlfDwO/6/MCexdUuJZghD2gszF916eWiiyrTDfMpSqcrwl13upXr6eNVpVMvcllmDH9GRnXr14YswVeFawXw0Vdv/ozL8lCc/KsLFaZLFkJml0sA/eijis0DZbBMI7CN3I+MQePzOb06WFIfsJJpyz0N9Aq7ilKcEaZtYznrnNrVRx3acYKc6mKtdE+ByieNHBh3EadkbfjTvAunPJQgslROhi8X7wsBoPbTn7ISw4kE9pY7EAaoS34I0jWQLD8KXYF4gogsgPZ54LnxAPiqii3SltKPoAGsod6TlhReBUwLU+tc7JSDA8wffAWj6gcR5hL9wt862wNG+cQy/HoUbsHsg7jE1hiMRSLHRaHKvOzuprI1h7DDDwxMuepL7s1jN8Gwe/v4z86+T3LNW4QLVgx6bYGy2TKdPpYy2N/G1caJlH/WT2FdoSle6EhczVCB66HnVs12dJXQ9Q+YvxLE+oajxE7+O7hfev2f3h1IvjWPYgaMGAZoj2p4t3WasKvjbXRjOMmUmivVFfy2V8BK6W2cX+v9nA03BZxg2GLHe6dZzDiLziu7IVM1Z7ifXy/wztnJQw2TETAe8fpRcPvXA+vDL1m16f7nv6hTJNwYU79/7L5P+Qvs/k='

def decodeVertices():
    vertices_binary = zlib.decompress(base64.b64decode(vertices_base64))
    k = 0
    for i in range(len(numVertices)):
        current = []
        for j in range(numVertices[i] * 3):
            current.append(float(struct.unpack('=d', vertices_binary[k:k+8])[0]))
            k += 8
        vertices.append(current)

def showVTK():
    try:
        import numpy as np
        import pyvista as pv
    except ModuleNotFoundError:
        sys.stderr.write("PyVista is not installed.\n")
        sys.stderr.write("Install it in the local virtual environment with:\n")
        sys.stderr.write("  source /Users/adelmann/git/opalx/.venv-h6/bin/activate\n")
        sys.stderr.write("  python -m pip install pyvista\n")
        return 1

    vtk_file = "FODO-no-spacecharge_ElementPositions.vtk"
    script_file = os.path.abspath(__file__)
    needs_export = not os.path.exists(vtk_file)
    if not needs_export:
        needs_export = os.path.getmtime(script_file) > os.path.getmtime(vtk_file)
    if needs_export:
        exportVTK()

    mesh = pv.read(vtk_file)
    plotter = pv.Plotter()
    add_mesh_kwargs = {}
    active_scalars_name = mesh.active_scalars_name
    if active_scalars_name is not None:
        active_scalars = mesh.active_scalars
        if getattr(active_scalars, 'ndim', 0) == 2 and active_scalars.shape[1] in (3, 4):
            add_mesh_kwargs['scalars'] = active_scalars_name
            add_mesh_kwargs['rgb'] = True
            add_mesh_kwargs['preference'] = 'cell'
            unique_colors = active_scalars[:, :3].astype(float)
            if np.nanmax(unique_colors) > 1.0:
                unique_colors /= 255.0
            unique_colors = np.unique(unique_colors, axis=0)
            legend = [
                ('Other', (0.5, 0.5, 0.5)),
                ('Dipole', (1.0, 0.847, 0.0)),
                ('Quadrupole', (1.0, 0.0, 0.0)),
                ('Sextupole', (0.537, 0.745, 0.525)),
                ('Octupole', (0.5, 0.5, 0.0)),
                ('Solenoid', (1.0, 138.0 / 255.0, 0.0)),
                ('RFCavity', (1.0, 1.0, 0.0)),
                ('TravelingWave', (0.0, 0.6, 0.0)),
                ('Drift', (0.0, 0.0, 1.0)),
            ]
            present = []
            for label, color in legend:
                color_arr = np.asarray(color, dtype=float)
                if np.any(np.all(np.isclose(unique_colors, color_arr, atol=5e-3), axis=1)):
                    present.append((label, color))
            if present:
                plotter.add_legend(present, bcolor=(1.0, 1.0, 1.0), face='circle', border=True, size=(0.2, 0.24))
    plotter.add_mesh(mesh, **add_mesh_kwargs)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
    return 0

def dot(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def normalize(a):
    length = math.sqrt(dot(a, a))
    for i in range(3):
        a[i]/=length
    return a

def distance(a, b):
    c = [0] * 2
    for i in range(len(a)):
        c[i] = a[i] - b[i]
    return math.sqrt(dot(c, c))

def cross(a, b):
    c = [0, 0, 0]
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

class Quaternion:
    def __init__(self, values):
        self.R = values

    def __str__(self):
        return str(self.R)

    def __mul__(self, other):
        imagThis = self.imag()
        imagOther = other.imag()
        cro = cross(imagThis, imagOther)

        ret = ([self.R[0] * other.R[0] - dot(imagThis, imagOther)] +
               [self.R[0] * imagOther[0] + other.R[0] * imagThis[0] + cro[0],
                self.R[0] * imagOther[1] + other.R[0] * imagThis[1] + cro[1],
                self.R[0] * imagOther[2] + other.R[0] * imagThis[2] + cro[2]])

        return Quaternion(ret)

    def imag(self):
        return self.R[1:]

    def conjugate(self):
        ret = [0] * 4
        ret[0] = self.R[0]
        ret[1] = -self.R[1]
        ret[2] = -self.R[2]
        ret[3] = -self.R[3]

        return Quaternion(ret)

    def inverse(self):
        return self.conjugate()

    def rotate(self, vec3D):
        quat = Quaternion([0.0] + vec3D)
        conj = self.conjugate()

        ret = self * (quat * conj)
        return ret.R[1:]

def getQuaternion(u, ref):
    u = normalize(u)
    ref = normalize(ref)
    axis = cross(u, ref)
    normAxis = math.sqrt(dot(axis, axis))

    if normAxis < 1e-12:
        if math.fabs(dot(u, ref) - 1.0) < 1e-12:
            return Quaternion([1.0, 0.0, 0.0, 0.0])

        return Quaternion([0.0, 1.0, 0.0, 0.0])

    axis = normalize(axis)
    cosAngle = math.sqrt(0.5 * (1 + dot(u, ref)))
    sinAngle = math.sqrt(1 - cosAngle * cosAngle)

    return Quaternion([cosAngle, sinAngle * axis[0], sinAngle * axis[1], sinAngle * axis[2]])

def exportVTK():
    vertices_str = ""
    triangles_str = ""
    color_str = ""
    cellTypes_str = ""
    startIdx = 0
    vertexCounter = 0
    cellCounter = 0
    lookup_table = []
    lookup_table.append([0.5, 0.5, 0.5, 0.5])
    lookup_table.append([1.0, 0.847, 0.0, 1.0])
    lookup_table.append([1.0, 0.0, 0.0, 1.0])
    lookup_table.append([0.537, 0.745, 0.525, 1.0])
    lookup_table.append([0.5, 0.5, 0.0, 1.0])
    lookup_table.append([1.0, 0.5411764706, 0.0, 1.0])
    lookup_table.append([1.0, 1.0, 0.0, 1.0])
    lookup_table.append([0.0, 0.6, 0.0, 1.0])
    lookup_table.append([0.0, 0.0, 1.0, 1.0])

    decodeVertices()

    for i in range(len(vertices)):
        for j in range(0, len(vertices[i]), 3):
            vertices_str += ("%f %f %f\n" %(vertices[i][j], vertices[i][j+1], vertices[i][j+2]))
            vertexCounter += 1

        for j in range(0, len(triangles[i]), 3):
            triangles_str += ("3 %d %d %d\n" % (triangles[i][j] + startIdx, triangles[i][j+1] + startIdx, triangles[i][j+2] + startIdx))
            cellTypes_str += "5\n"
            tmp_color = lookup_table[color[i]]
            color_str += ("%f %f %f %f\n" % (tmp_color[0], tmp_color[1], tmp_color[2], tmp_color[3]))
            cellCounter += 1
        startIdx = vertexCounter

    fh = open('FODO-no-spacecharge_ElementPositions.vtk','w')
    fh.write("# vtk DataFile Version 2.0\n")
    fh.write("test\nASCII\n\n")
    fh.write("DATASET UNSTRUCTURED_GRID\n")
    fh.write("POINTS " + str(vertexCounter) + " float\n")
    fh.write(vertices_str)
    fh.write("CELLS " + str(cellCounter) + " " + str(cellCounter * 4) + "\n")
    fh.write(triangles_str)
    fh.write("CELL_TYPES " + str(cellCounter) + "\n")
    fh.write(cellTypes_str)
    fh.write("CELL_DATA " + str(cellCounter) + "\n")
    fh.write("COLOR_SCALARS type 4\n")
    fh.write(color_str + "\n")
    fh.close()

def getNormal(tri_vertices):
    vec1 = [tri_vertices[1][0] - tri_vertices[0][0],
            tri_vertices[1][1] - tri_vertices[0][1],
            tri_vertices[1][2] - tri_vertices[0][2]]
    vec2 = [tri_vertices[2][0] - tri_vertices[0][0],
            tri_vertices[2][1] - tri_vertices[0][1],
            tri_vertices[2][2] - tri_vertices[0][2]]
    return normalize(cross(vec1,vec2))

def exportWeb(bgcolor):
    lookup_table = []
    lookup_table.append([0.5, 0.5, 0.5])
    lookup_table.append([1.0, 0.847, 0.0])
    lookup_table.append([1.0, 0.0, 0.0])
    lookup_table.append([0.537, 0.745, 0.525])
    lookup_table.append([0.5, 0.5, 0.0])
    lookup_table.append([1.0, 0.5411764706, 0.0])
    lookup_table.append([1.0, 1.0, 0.0])
    lookup_table.append([0.0, 0.6, 0.0])
    lookup_table.append([0.0, 0.0, 1.0])

    decodeVertices()

    mesh = "'data:"
    mesh += "{"
    mesh += "\"autoClear\":true,"
    mesh += "\"clearColor\":[0.0000,0.0000,0.0000],"
    mesh += "\"ambientColor\":[0.0000,0.0000,0.0000],"
    mesh += "\"gravity\":[0.0000,-9.8100,0.0000],"
    mesh += "\"cameras\":["
    mesh += "{"
    mesh += "\"name\":\"Camera\","
    mesh += "\"id\":\"Camera\","
    mesh += "\"position\":[21.7936,2.2312,-85.7292],"
    mesh += "\"rotation\":[0.0432,-0.1766,-0.0668],"
    mesh += "\"fov\":0.8578,"
    mesh += "\"minZ\":10.0000,"
    mesh += "\"maxZ\":10000.0000,"
    mesh += "\"speed\":1.0000,"
    mesh += "\"inertia\":0.9000,"
    mesh += "\"checkCollisions\":false,"
    mesh += "\"applyGravity\":false,"
    mesh += "\"ellipsoid\":[0.2000,0.9000,0.2000]"
    mesh += "}],"
    mesh += "\"activeCamera\":\"Camera\","
    mesh += "\"lights\":["
    mesh += "{"
    mesh += "\"name\":\"Lamp\","
    mesh += "\"id\":\"Lamp\","
    mesh += "\"type\":0.0000,"
    mesh += "\"position\":[4.0762,34.9321,-63.5788],"
    mesh += "\"intensity\":1.0000,"
    mesh += "\"diffuse\":[1.0000,1.0000,1.0000],"
    mesh += "\"specular\":[1.0000,1.0000,1.0000]"
    mesh += "}],"
    mesh += "\"materials\":[],"
    mesh += "\"meshes\":["

    for i in range(len(triangles)):
        vertex_list = []
        indices_list = []
        normals_list = []
        color_list = []
        for j in range(0, len(triangles[i]), 3):
            tri_vertices = []
            idcs = triangles[i][j:j + 3]
            tri_vertices.append(vertices[i][3 * idcs[0]:3 * (idcs[0] + 1)])
            tri_vertices.append(vertices[i][3 * idcs[1]:3 * (idcs[1] + 1)])
            tri_vertices.append(vertices[i][3 * idcs[2]:3 * (idcs[2] + 1)])
            indices_list.append(','.join(str(n) for n in range(len(vertex_list),len(vertex_list) + 3)))
            # left hand order!
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[0]))
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[2]))
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[1]))
            normal = getNormal(tri_vertices)
            normals_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in normal * 3))
            color_list.append(','.join([str(n) for n in lookup_table[color[i]]] * 3))
        mesh += "{"
        mesh += "\"name\":\"element_" + str(i) + "\","
        mesh += "\"id\":\"element_" + str(i) + "\","
        mesh += "\"position\":[0.0,0.0,0.0],"
        mesh += "\"rotation\":[0.0,0.0,0.0],"
        mesh += "\"scaling\":[1.0,1.0,1.0],"
        mesh += "\"isVisible\":true,"
        mesh += "\"isEnabled\":true,"
        mesh += "\"useFlatShading\":false,"
        mesh += "\"checkCollisions\":false,"
        mesh += "\"billboardMode\":0,"
        mesh += "\"receiveShadows\":false,"
        mesh += "\"positions\":[" + ','.join(vertex_list) + "],"
        mesh += "\"normals\":[" + ','.join(normals_list) + "],"
        mesh += "\"indices\":[" + ','.join(indices_list) + "],"
        mesh += "\"colors\":[" + ','.join(color_list) + "],"
        mesh += "\"subMeshes\":["
        mesh += "{"
        mesh += "\"materialIndex\":0,"
        mesh += " \"verticesStart\":0,"
        mesh += " \"verticesCount\":" + str(len(triangles[i])) + ","
        mesh += " \"indexStart\":0,"
        mesh += " \"indexCount\":" + str(len(triangles[i])) + ""
        mesh += "}]"
        mesh += "}"
        mesh += ","

        del normals_list[:]
        del vertex_list[:]
        del color_list[:]
        del indices_list[:]

    mesh = mesh[:-1] + "]"
    mesh += "}'"
    index_compressed = base64.b64decode(index_base64)
    index = zlib.decompress(index_compressed).decode('utf-8')
    if (len(bgcolor) == 3):
        mesh += ";\n            "
        mesh += "scene.clearColor = new BABYLON.Color3(%f, %f, %f)" % (bgcolor[0], bgcolor[1], bgcolor[2])

    index = index.replace('##DATA##', mesh)
    fh = open('FODO-no-spacecharge_ElementPositions.html','w')
    fh.write(index)
    fh.close()

def computeMinAngle(idx, curAngle, positions, connections, check):
    matrix = [[-math.cos(curAngle), -math.sin(curAngle)], [math.sin(curAngle), -math.cos(curAngle)]]

    minAngle = 2 * math.pi
    nextIdx = -1

    for j in connections[idx]:
        direction = [positions[j][0] - positions[idx][0],
                     positions[j][1] - positions[idx][1]]
        direction = [dot(matrix[0],direction), dot(matrix[1],direction)]

        if math.fabs(dot([1.0, 0.0], direction) / distance(positions[j], positions[idx]) - 1.0) < 1e-6: continue

        angle = math.fmod(math.atan2(direction[1],direction[0]) + 2 * math.pi, 2 * math.pi)

        if angle < minAngle:
            nextIdx = j
            minAngle = angle
        if angle == minAngle and check:
            dire =  [positions[j][0] - positions[idx][0],
                     positions[j][1] - positions[idx][1]]
            minA0 = math.atan2(dire[1], dire[0])
            minA1 = computeMinAngle(nextIdx, minA0, positions, connections, False)
            minA2 = computeMinAngle(j, minA0, positions, connections, False)
            if minA2[1] < minA2[1]:
                nextIdx = j

    if nextIdx == -1:
        nextIdx = connections[idx][0]

    return (nextIdx, minAngle)

def squashVertices(positionDict, connections):
    removedItems = []
    indices = [int(k) for k in positionDict.keys()]
    idxChanges = indices
    for i in indices:
        if i in removedItems:
            continue
        for j in indices:
            if j in removedItems or j == i:
                continue

            if distance(positionDict[i], positionDict[j]) < 1e-6:
                connections[j] = list(set(connections[j]))
                if i in connections[j]:
                    connections[j].remove(i)
                if j in connections[i]:
                    connections[i].remove(j)

                connections[i].extend(connections[j])
                connections[i] = list(set(connections[i]))
                connections[i].sort()

                for k in connections.keys():
                    if k == i: continue
                    if j in connections[k]:
                        idx = connections[k].index(j)
                        connections[k][idx] = i

                idxChanges[j] = i
                removedItems.append(j)

    for i in removedItems:
        del positionDict[i]
        del connections[i]

    for i in connections.keys():
        connections[i] = list(set(connections[i]))
        connections[i].sort()

    return idxChanges

def computeLineEquations(positions, connections):
    cons = set()
    for i in connections.keys():
        for j in connections[i]:
            cons.add((min(i, j), max(i, j)))

    lineEquations = {}
    for item in cons:
        a = (positions[item[1]][1] - positions[item[0]][1])
        b = -(positions[item[1]][0] - positions[item[0]][0])

        xm = 0.5 * (positions[item[0]][0] + positions[item[1]][0])
        ym = 0.5 * (positions[item[0]][1] + positions[item[1]][1])
        c = -(a * xm +  b * ym)

        lineEquations[item] = (a, b, c)

    return lineEquations

def checkPossibleSegmentIntersection(segment, positions, lineEquations, minAngle, lastIntersection):
    item1 = (min(segment), max(segment))

    (a1, b1, c1) = (0,0,0)
    A = [0]*2
    B = A

    if segment[0] == None:
        A = lastIntersection
        B = positions[segment[1]]

        a1 = B[1] - A[1]
        b1 = -(B[0] - A[0])
        xm = 0.5 * (A[0] + B[0])
        ym = 0.5 * (A[1] + B[1])
        c1 = -(a1 * xm + b1 * ym)

    else:
        A = positions[segment[0]]
        B = positions[segment[1]]

        (a1, b1, c1) = lineEquations[item1]

        if segment[1] < segment[0]:
            (a1, b1, c1) = (-a1, -b1, -c1)

    curAngle = math.atan2(a1, -b1)
    matrix = [[-math.cos(curAngle), -math.sin(curAngle)], [math.sin(curAngle), -math.cos(curAngle)]]

    origMinAngle = minAngle

    segment1 = [B[0] - A[0], B[1] - A[1], 0.0]

    intersectingSegments = []
    distanceAB = distance(A, B)
    for item2 in lineEquations.keys():
        item = item2
        C = positions[item[0]]
        D = positions[item[1]]

        if segment[0] == None:
            if (segment[1] == item[0] or
                segment[1] == item[1]): continue
        else:
            if (item1[0] == item[0] or
                item1[1] == item[0] or
                item1[0] == item[1] or
                item1[1] == item[1]): continue

        nodes = set([item1[0],item1[1],item[0],item[1]])
        if len(nodes) < 4: continue       # share same vertex

        (a2, b2, c2) = lineEquations[item]

        segment2 = [C[0] - A[0], C[1] - A[1], 0.0]
        segment3 = [D[0] - A[0], D[1] - A[1], 0.0]

        # check that C and D aren't on the same side of AB
        if cross(segment1, segment2)[2] * cross(segment1, segment3)[2] > 0.0: continue

        if cross(segment1, segment2)[2] < 0.0 or cross(segment1, segment3)[2] > 0.0:
            (C, D, a2, b2, c2) = (D, C, -a2, -b2, -c2)
            item = (item[1], item[0])

        denominator = a1 * b2 - b1 * a2
        if math.fabs(denominator) < 1e-9: continue

        px = (b1 * c2 - b2 * c1) / denominator
        py = (a2 * c1 - a1 * c2) / denominator

        distanceCD = distance(C, D)

        distanceAP = distance(A, [px, py])
        distanceBP = distance(B, [px, py])
        distanceCP = distance(C, [px, py])
        distanceDP = distance(D, [px, py])

        # check intersection is between AB and CD
        check1 = (distanceAP - 1e-6 < distanceAB and distanceBP - 1e-6 < distanceAB)
        check2 = (distanceCP - 1e-6 < distanceCD and distanceDP - 1e-6 < distanceCD)
        if not check1 or not check2: continue

        if math.fabs(dot(segment1, [D[0] - C[0], D[1] - C[1], 0.0]) / (distanceAB * distanceCD) + 1.0) < 1e-9: continue

        if ((distanceAP < 1e-6) or
            (distanceBP < 1e-6 and distanceCP < 1e-6) or
            (distanceDP < 1e-6)):
            continue

        direction = [D[0] - C[0], D[1] - C[1]]
        direction = [dot(matrix[0], direction), dot(matrix[1], direction)]
        angle = math.fmod(math.atan2(direction[1], direction[0]) + 2 * math.pi, 2 * math.pi)

        newSegment = ([px, py], item[1], distanceAP, angle)

        if distanceCP < 1e-6 and angle > origMinAngle: continue
        if distanceBP > 1e-6 and angle > math.pi: continue

        if len(intersectingSegments) == 0:
            intersectingSegments.append(newSegment)
            minAngle = angle
        else:
            if intersectingSegments[0][2] - 1e-9 > distanceAP:
                intersectingSegments[0] = newSegment
                minAngle = angle
            elif intersectingSegments[0][2] + 1e-9 > distanceAP and angle < minAngle:
                intersectingSegments[0] = newSegment
                minAngle = angle

    return intersectingSegments

def projectToPlane(normal):
    fh = open('FODO-no-spacecharge_ElementPositions.gpl','w')
    normal = normalize(normal)
    ori = getQuaternion(normal, [0, 0, 1])

    left2Right = [0, 0, 1]
    if math.fabs(math.fabs(dot(normal, left2Right)) - 1) < 1e-9:
        left2Right = [1, 0, 0]
    rotL2R = ori.rotate(left2Right)
    deviation = math.atan2(rotL2R[1], rotL2R[0])
    rotBack = Quaternion([math.cos(0.5 * deviation), 0, 0, -math.sin(0.5 * deviation)])
    ori = rotBack * ori

    decodeVertices()

    for i in range(len(vertices)):
        positions = {}
        connections = {}
        for j in range(0, len(vertices[i]), 3):
            nextPos3D = ori.rotate(vertices[i][j:j+3])
            nextPos2D = nextPos3D[0:2]
            positions[j/3] = nextPos2D

        if len(positions) == 0:
            continue

        idx = 0
        maxX = positions[0][0]
        for j in positions.keys():
            if positions[j][0] > maxX:
                maxX = positions[j][0]
                idx = j
            if positions[j][0] == maxX and positions[j][1] > positions[idx][1]:
                idx = j

        for j in range(0, len(triangles[i]), 3):
            for k in range(0, 3):
                vertIdx = triangles[i][j + k]
                if not vertIdx in connections:
                    connections[vertIdx] = []
                for l in range(1, 3):
                    connections[vertIdx].append(triangles[i][j + ((k + l) % 3)])

        numConnections = 0
        for j in connections.keys():
            connections[j] = list(set(connections[j]))
            numConnections += len(connections[j])

        numConnections /= 2
        idChanges = squashVertices(positions, connections)

        lineEquations = computeLineEquations(positions, connections)

        idx = idChanges[int(idx)]
        fh.write("%.6f    %.6f\n" % (positions[idx][0], positions[idx][1]))

        curAngle = math.pi
        origin = idx
        count = 0
        passed = []
        while (count == 0 or distance(positions[origin], positions[idx]) > 1e-9) and not count > numConnections:
            nextGen = computeMinAngle(idx, curAngle, positions, connections, False)
            nextIdx = nextGen[0]
            direction = [positions[nextIdx][0] - positions[idx][0],
                         positions[nextIdx][1] - positions[idx][1]]
            curAngle = math.atan2(direction[1], direction[0])

            intersections = checkPossibleSegmentIntersection((idx, nextIdx), positions, lineEquations, nextGen[1], [])
            if len(intersections) > 0:
                while len(intersections) > 0 and not count > numConnections:
                    fh.write("%.6f    %.6f\n" %(intersections[0][0][0], intersections[0][0][1]))
                    count += 1
    
                    idx = intersections[0][1]
                    direction = [positions[idx][0] - intersections[0][0][0],
                                 positions[idx][1] - intersections[0][0][1]]
                    curAngle = math.atan2(direction[1], direction[0])
                    nextGen = computeMinAngle(idx, curAngle, positions, connections, False)
    
                    intersections = checkPossibleSegmentIntersection((None, idx), positions, lineEquations, nextGen[1], intersections[0][0])
            else:
                idx = nextIdx
    
            fh.write("%.6f    %.6f\n" % (positions[idx][0], positions[idx][1]))
    
            if idx in passed:
                direction1 = [positions[idx][0] - positions[passed[-1]][0],
                              positions[idx][1] - positions[passed[-1]][1]]
                direction2 = [positions[origin][0] - positions[passed[-1]][0],
                              positions[origin][1] - positions[passed[-1]][1]]
                dist1 = distance(positions[idx], positions[passed[-1]])
                dist2 = distance(positions[origin], positions[passed[-1]])
                if dist1 * dist2 > 0.0 and math.fabs(math.fabs(dot(direction1, direction2) / (dist1 * dist2)) - 1.0) > 1e-9:
                    sys.stderr.write("error: projection cycling on element id: %d, vertex id: %d\n" %(i, idx))
                break

            passed.append(idx)
            count += 1
        fh.write("\n")

        if count > numConnections:
            sys.stderr.write("error: projection cycling on element id: %d\n" % i)

        for j in range(0, len(decoration[i]), 6):
            for k in range(j, j + 6, 3):
                nextPos3D = ori.rotate(decoration[i][k:k+3])
                fh.write("%.6f    %.6f\n" % (nextPos3D[0], nextPos3D[1]))
            fh.write("\n")

    fh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-vtk', action='store_true')
    parser.add_argument('--export-web', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--background', nargs=3, type=float)
    parser.add_argument('--project-to-plane', action='store_true')
    parser.add_argument('--normal', nargs=3, type=float)
    args = parser.parse_args()

    if (args.export_vtk):
        exportVTK()
        sys.exit()

    if (args.export_web):
        bgcolor = []
        if (args.background):
            validBackground = True
            for comp in bgcolor:
                if comp < 0.0 or comp > 1.0:
                    validBackground = False
                    break
            if (validBackground):
                bgcolor = args.background
        exportWeb(bgcolor)
        sys.exit()

    if (args.show):
        sys.exit(showVTK())

    if (args.project_to_plane):
        normal = [0.0, 1.0, 0.0]
        if (args.normal):
            normal = args.normal
        projectToPlane(normal)
        sys.exit()

    parser.print_help()
