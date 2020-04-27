import pyrender
from io import BytesIO
import numpy as np
import trimesh
import requests
import matplotlib.pyplot as plt

def rotate_vector(vector, angle):
    '''
    Rotates a vector by a given angle.
    '''
    return [vector[0]*np.cos(angle) - vector[1]*np.sin(angle), vector[0]*np.sin(angle) + vector[1]*np.cos(angle)]

def scale_vector(vector, length):
    '''
    Scales a vector to be a certain length.
    '''
    mag = np.sqrt(vector[0]**2 + vector[1]**2)
    vector = length*(np.array(vector)/mag)
    return vector

def add_vector_bi(vector, point):
    '''
    Takes a vector and point, returns point + vector and point - vector
    '''
    new_point_pos = [point[0] + vector[0], point[1] + vector[1], point[2]]
    new_point_neg = [point[0] - vector[0], point[1] - vector[1], point[2]]
    return (new_point_pos, new_point_neg)

def get_corner_normals(line, line_width):
    '''
    Generates a 2-dimensional representation of a given 1-dimensional line. That
    is, it returns a list of more points that defines a line that has width. It
    does this by going across the given line and expanding outwards from each
    point of the line (hence 'corner_normals', as it's getting points normal to
    the general direction of the line at any given point)
    '''

    result = []

    # TODO: make width be based on variable, on extrusion amount
    width = line_width

    if len(line) > 1:

        #Getting vector normal to very first line, to get our starting points
        first_normal = scale_vector(rotate_vector([line[1][0] - line[0][0], line[1][1] - line[0][1]], np.pi/2), width)
        start_points = add_vector_bi(first_normal, line[0])
        result.append(start_points[0])
        result.append(start_points[1])

        for i in range(1, len(line) - 1):
            #Getting the points before and after the current point in the line
            last_point = line[i - 1]
            current_point = line[i]
            next_point = line[i + 1]

            """
            Getting the vectors between the last and next points, effectiely
            getting each line segment before and after our current point
            """
            last_dif = [current_point[0] - last_point[0], current_point[1] - last_point[1]]
            next_dif = [current_point[0] - next_point[0], current_point[1] - next_point[1]]

            """
            Getting vectors normal to the lines above. I rotate by 90° instead
            of actually doing a normal calculation because the formulas to find
            normals I found were never reliable; the normals generated would
            sometimes be on one side or another of the line, and this messes
            things up later on.
            """
            last_normal = rotate_vector(last_dif, np.pi/2)
            next_normal = rotate_vector(next_dif, -np.pi/2)

            normal = []

            """
            We want to now add both normals together to get the final combined
            normal. However, if both difference vectors are negatives of eachother,
            the result will be the 0 vector and mess everything up, so we check for
            that here. This can often be a problem when there's a stretch of line
            that's completely horizontal or completely vertical.
            """
            if last_normal[0] == next_normal[0]*-1 and last_normal[1] == next_normal[1]*-1:
                normal = last_normal
            else:
                normal = [(last_normal[0] + next_normal[0]), (last_normal[1] + next_normal[1])]

            #Scaling result back to width
            magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
            normal = [(normal[0] / magnitude) * width, (normal[1] / magnitude) * width]

            #Adds the normal on to the current_point, in both directions, to get our new corner points
            corner_points = add_vector_bi(normal, current_point)

            result.append(corner_points[0])
            result.append(corner_points[1])

        #Getting vector normal to very last line, to get our ending points
        last_normal = scale_vector(rotate_vector([line[-1][0] - line[-2][0], line[-1][1] - line[-2][1]], np.pi/2), width)
        end_points = add_vector_bi(last_normal, line[-1])
        result.append(end_points[0])
        result.append(end_points[1])

    return result

def parse_gcode_file(filename, hotend_distance=0):
    '''
    Reads a gcode file and generates a series of lines that represent where the
    extruder(s) would put down material when printing. These lines are defined
    by when the extruder should stop or start extruding - so from when the hotend
    starts pushing material out to when it stops is one line. On top of that,
    each line is organized into layers, each being a list of lines for a given height.
    hotend_distance is the distance between the two hotends if the printer has
    multiple. On Ultimakers, a slicer has to compensate for this distance by
    offseting the position of the head, and setting hotend_distance will compensate
    for that.
    Returns two lists of layers of lines representing a 3D object, as defined in a gcode file.
    The first list has lines made by hotend 0, the second list has lines made by hotend 1. If
    only one hotend was used, the list for the other will be empty.
    '''
    file = open(filename, "r")

    one_layers = []
    two_layers = []

    one_lines = []
    two_lines = []

    one_new_line = []
    two_new_line = []

    current_x = 0.0
    current_y = 0.0
    current_z = 0.0
    last_extruded_z = 0.0
    last_e = 0.0
    e_change = 0.0

    core_switch_save = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    printing = False


    printcore = 0

    count = 0
    for line in file:

        #Need to check if this line has or is a comment, and shave off the comment or skip it if it is/does, respectively
        line = line.split(";")[0]
        if line == "":
            continue


        #Checking for when the active core is switched
        if line[:2] == "T0" or line[:2] == "T1":
            new_core = int(line[1])
            #In case T0 or T1 commands get called for some reason while we're still on the same core
            #I don't think that'll ever happen but I want the code to work if it does
            if new_core != printcore:
                """
                Now we need to switch all of our values to be the ones we want
                for the new hotend, and save all of our current values for when
                we switch back to the current (soon to be previous) hotend
                """
                printcore = new_core
                current_values = [current_x, current_y, current_z, last_extruded_z, last_e, e_change]
                current_x = core_switch_save[0]
                current_y = core_switch_save[1]
                current_z = core_switch_save[2]
                last_extruded_z = core_switch_save[3]
                last_e = core_switch_save[4]
                e_change = core_switch_save[5]
                core_switch_save = current_values


        #Only start recording once we see this command. M204 prolly would work too.
        if line[:4] == "M205":
            printing = True

        if printing:
            #G0 or G1 = movement, G0 typically used for non-extrusion, G1 for extrusion
            if line[:2] == "G0" or line[:2] == "G1":

                parameters = line.split(" ")

                new_x = current_x
                new_y = current_y
                new_z = current_z
                new_e = last_e

                #Going through command parameters to get info out. Can skip first thing in list since it'll just be G0 or G1
                for p in parameters[1:]:
                    if p == "":
                        continue

                    if p[0] == "X":
                        new_x = float(p[1:])
                        """
                        Position of the printhead in gcode doesn't care about
                        which hotend we're using, so our slicer needs to shift
                        over a bit so the second hotend is aligned correctly.
                        Here, we undo this shifting so we record the actual
                        coordinates that the nozzle moved over.
                        """
                        if printcore == 1:
                            new_x += hotend_distance
                    elif p[0] == "Y":
                        new_y = float(p[1:])
                    elif p[0] == "Z":
                        new_z = float(p[1:])
                    elif p[0] == "E":
                        new_e = float(p[1:])

                did_extrude = False
                # If E is specified, we're extruding on this move
                # This is probably not necessary since the code as it is would give 0 e_change for lines without 'E' in them, but I'll leave this in just to be safe
                if "E" in line and ("X" in line or "Y" in line or "Z" in line):

                    # Check if this extrusion is actually pushing material out of the nozzle
                    # Need to pay attention to retractions, and how far filament is pulled out of the nozzle at times
                    # Thus, if e_change becomes negative, we want to keep it negative until there are enough positive changes to bring it back
                    extrusion_dif = new_e - last_e
                    if e_change <= 0:
                        e_change = e_change + extrusion_dif
                    else:
                        e_change = extrusion_dif

#                     print(e_change)

                    # Only record these movements if we're actually extruding
                    if e_change > 0.0:
                        did_extrude = True

                        #Check if we're starting a new layer, and start adding to a new layer if so
                        if new_z != last_extruded_z:
                            if printcore == 0:
                                one_layers.append(one_lines)
                                one_lines = []
                                one_new_line = []
                            else:
                                two_layers.append(two_lines)
                                two_lines = []
                                two_new_line = []
                        if printcore == 0:
                            one_new_line.append([new_x, new_y, new_z])
                        else:
                            two_new_line.append([new_x, new_y, new_z])
                        last_extruded_z = new_z

                #Not extruding on this move, so now start a new line/maybe point(s) if there are multiple non-extrusion moves
                if not did_extrude:
                    if printcore == 0:
                        if len(one_new_line) > 1:
                            one_lines.append(one_new_line)
                        one_new_line = []
                        one_new_line.append([new_x, new_y, new_z])
                    else:
                        if len(two_new_line) > 1:
                            two_lines.append(two_new_line)
                        two_new_line = []
                        two_new_line.append([new_x, new_y, new_z])

                current_x = new_x
                current_y = new_y
                current_z = new_z


    #Removing first layer, since our first command will go to the starting Z coordinate and thus add an empty list to layers
    one_layers = one_layers[1:]
    two_layers = two_layers[1:]

    file.close()

    return one_layers, two_layers

def build_mesh_from_points(points, line_height):
    '''
    Given a list of points defining a two-dimensional line, will build a mesh of
    a 3D representation of the line by essentially extruding it upwards.
    points is the list of points to build a mesh off of
    line_height is the height of the line
    Returns a list of points for the new 3D shape, as well as a list of indices
    into this list, defining the triangles of the generated mesh.
    '''
    point_count = len(points)
    second_layer = np.array(points) + np.array([0,0,line_height])
    indices = []

    #Front Face
    indices += [[0,point_count+1,1],[0,point_count,point_count+1]]

    #Back Face
    indices += [[point_count-2,point_count-1,(point_count*2)-1],[point_count-2,(point_count*2)-1,(point_count*2)-2]]

    i=0
    while i < point_count-2:
        #Bottom layer
        indices += [[i,i+1,i+3],[i,i+3,i+2]]
        #Top Layer
        indices += [[point_count+i,point_count+i+3,point_count+i+1],[point_count+i,point_count+i+2,point_count+i+3]]
        #Right Layer
        indices += [[i,i+2,i+point_count],[i+2,i+2+point_count,i+point_count]]
        #Left Layer
        indices += [[i+1,i+1+point_count,i+3],[i+3,i+1+point_count,i+3+point_count]]
        i+=2

    points += list(second_layer)
    return points, indices
