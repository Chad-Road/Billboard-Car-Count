"""This code has been striped and optimized to work on computers with limited
power and processing capabilities (e.g. Raspberry Pi, Beagleboard, ODROID, etc.)

This assumes you are streaming from a live video source, otherwise it's better
to save the given file and run the object detection on the given machine

Note on Cameras:
Most USB cameras (webcams or similar) should work out-of-the-box on most 
single board computers, however, cameras that plug directly into the 
SBC may require picamera (rapsberry pi) or a similar libraryto get them 
to work properly.

CPU Power Note: 
While this has been optimized to use as little processing power as possible,
there are still many SBCs or microcontrollers with far too little poweer to have
usable object detection"""


import pandas as pd
import cv2 as cv
import math

# OpenCV automatically detect connected webcams and gives them an index number
# If you have multiple cameras connected you may have to change
video_capture_device_index = 0
video_capture = cv.VideoCapture(video_capture_device_index)

# Test to make sure that opencv has connected to your camera
# This will likely either work very easily or be virtually impossible
# You should check 
if video_capture:
    print("Successfully connected to live camera")
else:
    print("Cannot connect to live camera")
    exit()


total_frames_count = video_capture.get(cv.CAP_PROP_FRAME_COUNT)
frames_per_second = video_capture.get(cv.CAP_PROP_FPS)

cars_list = [["x_cord", "y_cord"]]
previous_centroid_x_cord = 0
previous_centroid_y_cord = 0

frame_number = 0
total_car_count = 0
number_is_close_tolerance = 5


# Line which if a centroid crosses, it is counted as a car
count_line_height = 340

# Sets minimum and maximum size to calculate contour centroid
minimum_contour_size = 200
maximum_contour_size = 1000

# The numbers in set are property identifiers 3 = width, 4 = height
# This sets video height and width to predetermined height and width
# Not usually needed, but sometimes required for certain cameras/SBCs
#     to work properly
video_height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
video_width = video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
video_capture.set(3, video_width)
video_capture.set(4, video_height)


# 0x7634706d is used to force opencv to use mp4 due to compatibility issues on certiain systems
# Note: Most systems will require the 'isColor' parameter to match output color settings
#    (e.g. color output "isColor" = True, grayscale output "isColor" = False)
video_result = cv.VideoWriter('video_result_V2.mp4', 0x7634706d, frames_per_second, (video_width, video_height), isColor=True)

# This will have the largest affect on performance
# In most cases MOG2 is the best option, but low power systems may require 'Frame Difference' instead
background_subtractor = cv.createBackgroundSubtractorMOG2()

while True:
    return_val, captured_image = video_capture.read()

    if return_val:

        # Changes image to grayscale
        gray_image = cv.cvtColor(captured_image, cv.COLOR_BGR2GRAY)

        # Slight blur to eliminate some nosie
        grayed_blured_image = cv.blur(gray_image, (5, 5))

        # Sets input file, specificies no output, and sets learning rate to 0.01
        foreground_mask = background_subtractor.apply(grayed_blured_image, None, .01)

        # Second blur after mask to limit noise in mask
        foreground_blured_image = cv.blur(foreground_mask, (12, 12))

        # Slight dialation 
        dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        dilation = cv.dilate(foreground_blured_image ,None, dilation_kernel, iterations=1)

        erosion_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
        erosion = cv.erode(dilation, None, erosion_kernel, iterations=7)

        # Removes grays and returns black and white (binary) image
        return_val, binary_image = cv.threshold(erosion, 1, 255, cv.THRESH_BINARY)

        # Finds the countours of the final output binary image
        contours, hierarchy = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            drawn_contours = cv.drawContours(captured_image, [cnt], -1, (255, 0, 0), 3, lineType=cv.LINE_4)

            # Create bounding boxes around contours
            x_cord, y_cord, width, height = cv.boundingRect(cnt)
            bounding_boxes = cv.rectangle(captured_image, (x_cord, y_cord), (x_cord + width, y_cord + height), 
                                            (0, 255, 0), 3)

        # Line which if a centroid crosses, it is counted as a car
        count_line = cv.line(captured_image, (0, count_line_height), (640, count_line_height), (0, 0, 255), 6)

        # Loop to cycle through all contours
        for i in range(len(contours)):

            # Calculate contour area of current contour in loop
            contour_area = cv.contourArea(contours[i])

            # Tests if contour is correct size and heirachy tests if contour is contained in another contour
            if (contour_area > minimum_contour_size and 
                contour_area < maximum_contour_size and
                hierarchy[0, i, 3]):

                # Get all moment for current contour
                current_contour = contours[i]
                current_moment = cv.moments(current_contour)

                # Standard formula to calculate centroid of a figure
                current_centroid_x_cord = int(current_moment['m10'] / current_moment['m00'])
                current_centroid_y_cord = int(current_moment['m01'] / current_moment['m00'])

                # Tests if the current contour has past the count line on the bottom of the screen
                # math.isclose is used to test if current centroid is close to another centroid in previous frame
                # Used to avoid double counting
                if (current_centroid_y_cord > count_line_height
                    and not math.isclose(current_centroid_x_cord, previous_centroid_x_cord, abs_tol=10)
                    and not math.isclose(current_centroid_y_cord, previous_centroid_y_cord, abs_tol=10)):

                    cars_list.append([current_centroid_x_cord, current_centroid_y_cord])
                    total_car_count += 1

                # After above test sets new previous centroid before loop restarts
                previous_centroid_x_cord = current_centroid_x_cord
                previous_centroid_y_cord = current_centroid_y_cord
    
        # Create blue rectangle on top left of screen with car count, also counts frame number on bottom left
        cv.rectangle(captured_image, (0, 0), (140, 20), (255, 0, 0), -1)
        cv.putText(captured_image, "Total Cars = " + str(total_car_count), (3, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv.putText(captured_image, "Frame number: " + str(frame_number), (3, 350), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)

        # Writes final image to file
        video_result.write(captured_image)

        frame_number += 1

        # cv.imshow('grayed blured', binary_image)

        # k = cv.waitKey(int(1000/frames_per_second)) & 0xff
        # if k == 27:
        #     break

    else:
        break

cars_df = pd.DataFrame(cars_list)
cars_df.to_csv("cars_df")

video_result.release()
cv.destroyAllWindows()








