from collections import namedtuple

Reading = namedtuple('Reading', ['bearing', 'size', 'r', 'g', 'b', 'a'])

def one_click_process(opencv_image):
    def unwrap_image(donut):
        # TODO(buckbaskin): take a donut image and convert it to an unwrapped 360
        # degree image
        return donut

    def panorama_to_readings(panorama):
        # TODO(buckbaskin): given a panorama, find blobs using opencv and convert
        #   their position in the image to a reading (bearing, size, color)
        list_of_readings = []
        list_of_readings.append(Reading(0,1,0,0,0,0))
        return list_of_readings

    wrapped = opencv_image
    unwrapped = unwrap_image(wrapped)
    return panorama_to_readings(unwrapped)