import utils
import pyrender
from io import BytesIO
import skimage.io as io
import numpy as np
import trimesh
import requests
import matplotlib.pyplot as plt

class WatchDog():

    def __init__(self, camera_pos, camera_rot, camera_fov, aspect_ratio, nozzle_width, main_color, secondary_color=[0.87,0.84,0.67]):
        '''
        camera_pos should be a list of 3 values, defining x, y, and z.
        camera_fov and aspect ratio should be single floating point numbers
        nozzle_width should be a single number defining the width in mm of the nozzle
        TODO: add multiple-nozzle-width functionality
        main_color is the color of the material being extruder from hotend 0,
        while secondary_color is the color of the material being extruded from hotend 1.
        As such, secondary_color's default is a PVA-ish color.
        '''
        self.camera_pos = camera_pos
        self.camera_fov = camera_fov
        self.camera_rot = camera_rot
        self.aspect_ratio = aspect_ratio
        self.nozzle_width = nozzle_width
        self.main_color = main_color
        self.secondary_color = secondary_color
        self.lights = []
        self.layers = []
        self.secondary_layers = []

    def read_layers_from_file(self, filename):
        #18 for nozzle offset because we're assuming we're on an Ultimaker 3 here
        self.layers, self.secondary_layers = utils.parse_gcode_file(filename, 18)

    def set_layers(self, layers, secondary_layers=[]):
        self.layers = layers
        if secondary_layers != []:
            self.secondary_layers = secondary_layers

    def add_light(self, light_pos, light_color, light_intensity, light_range):
        translation = np.array([
           [1.0,  0.0, 0.0, light_pos[0]],
           [0.0,  1.0, 0.0, light_pos[1]],
           [0.0,  0.0, 1.0, light_pos[2]],
           [0.0,  0.0, 0.0, 1.0],
        ])
        self.lights.append([pyrender.PointLight(color=light_color, intensity=light_intensity, range=light_range), translation])

    def build_object_mesh(self, main_layers, secondary_layers, height):
        '''
        Builds a mesh from the given layers. main_layers will typically the main object
        printed, while secondary_layers will typically the support material for that object.
        height is the number of layers to render from the bottom up.
        If the print is a single material print, give an empty list for the respective variable
        '''

        vertices = []
        indices = []

        if main_layers != []:
            for layer in main_layers[:height]:
                for line in layer:
                    if len(line) > 1:
                        normal_points = utils.get_corner_normals(line, 0.2)
                        new_vertices, new_indices = utils.build_mesh_from_points(normal_points, 0.2)
                        new_indices = list(np.array(new_indices) + len(vertices))
                        vertices += new_vertices
                        indices += new_indices

        main_object_vertices = len(vertices)


        if secondary_layers != []:
            for layer in secondary_layers[:height]:
                for line in layer:
                    if len(line) > 1:
                        normal_points = utils.get_corner_normals(line, 0.2)
                        new_vertices, new_indices = utils.build_mesh_from_points(normal_points, 0.2)
                        new_indices = list(np.array(new_indices) + len(vertices))
                        vertices += new_vertices
                        indices += new_indices

        secondary_vertices = len(vertices) - main_object_vertices

        colors = [self.main_color] * main_object_vertices + [self.secondary_color] * secondary_vertices

        return trimesh.Trimesh(vertices=vertices, faces=indices, vertex_colors=colors)

    def render_mesh(self, mesh, width, height):
        # for i in range(20):
        scene = pyrender.Scene()

        #Adding the mesh into the scene
        scene.add(pyrender.Mesh.from_trimesh(mesh))

        #Adding the camera into the scene
        camera = pyrender.PerspectiveCamera(yfov=self.camera_fov, aspectRatio=self.aspect_ratio)

        pitch_cos = np.cos(self.camera_rot[0])
        pitch_sin = np.sin(self.camera_rot[0])

        roll_cos = np.cos(self.camera_rot[1])
        roll_sin = np.sin(self.camera_rot[1])

        yaw_cos = np.cos(self.camera_rot[2])
        yaw_sin = np.sin(self.camera_rot[2])

        pitch = np.mat([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, pitch_cos,-pitch_sin, 0.0],
            [0.0, pitch_sin, pitch_cos, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        roll = np.mat([
            [roll_cos, 0.0, roll_sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-roll_sin, 0.0, roll_cos, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        yaw = np.mat([
            [yaw_cos, -yaw_sin, 0.0, 0.0],
            [yaw_sin, yaw_cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        camera_transform = np.mat([
            [1.0, 0.0, 0.0, self.camera_pos[0]],
            [0.0, 1.0, 0.0, self.camera_pos[1]],
            [0.0, 0.0, 1.0, self.camera_pos[2]],
            [0.0, 0.0, 0.0, 1.0]
        ])

        s = np.sqrt(2)/2
        camera_pose = np.mat([
           [1.0, 0.0,   0.0,   0.0],
           [0.0,  1.0, 0.0, 0.0],
           [0.0,  0.0,   1.0,   0.0],
           [0.0,  0.0, 0.0, 1.0],
        ])

        #Multiply yaw*roll*pitch
        camera_rotation = yaw*roll*pitch
        camera_pose = camera_transform*camera_rotation*camera_pose#*camera_z_rot*camera_y_rot*camera_x_rot


        scene.add(camera, pose=np.array(camera_pose))

        #Adding lights into the scene
        for light in self.lights:
            scene.add(light[0], pose=light[1])

        #Render the scene and return the resultant image
        r = pyrender.OffscreenRenderer(width, height)
        color, depth = r.render(scene)
            # io.imsave(str(i)+"_test.png", color)
        return color

    def test_compare_images(self, real, rendered):
        new_image = 3*(np.array(real)//4) + np.array(rendered)//4
        io.imsave("new.png", new_image)
        # plt.imshow(new_image)
        # plt.show()

    def compare_images(self, real, rendered, empty_value, threshold):
        """
        real is the real image taken
        rendered is the image we've rendered to compare to it
        empty_value should the value used for transparent pixels in rendered
        """


        real = np.array(real)
        rendered = np.array(rendered)

        difference = real - rendered

        total_pixels_compared = 0
        pixels_below_threshold = 0

        for i in range(difference.shape[0]):
            for j in range(difference.shape[1]):
                if np.array_equal(real[i,j], empty_value):
                    continue

                total_pixels_compared += 1
                if np.mean(difference[i,j]) < threshold:
                    pixels_below_threshold += 1

        # mask = (real == empty_value)
        #
        # difference = real - rendered
        # difference = difference + (mask * threshold)
        #
        # pixel_count = difference.shape[0]*difference.shape[1]
        # comparison_pixel_count = pixel_count - np.sum(mask)

        return pixels_below_threshold / total_pixels_compared
