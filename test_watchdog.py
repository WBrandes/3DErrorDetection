from watchdog import WatchDog

def main():

    real_im = io.imread("test_pictures/36.jpg")

    # Camera pos, FOV, aspect ratio, nozzle width, RGB color
    watchdog = WatchDog([273.7, -5, 15.0], [1.123, 0.05, 0.85], np.pi/3.0, 1.0, 0.4, [0.0,0.0,1.0])
    watchdog.add_light([163.7, 110, 47.65], [1.0, 1.0, 1.0], 5000.0, 1000.0)
    watchdog.read_layers_from_file("test_pictures/test_timelapse_eartube.gcode")
    mesh = watchdog.build_object_mesh(watchdog.layers, [], 36)
    rendered_im = watchdog.render_mesh(mesh, real_im.shape[1], real_im.shape[0])
    watchdog.test_compare_images(real_im, rendered_im)


if __name__ == "__main__":
    main()
