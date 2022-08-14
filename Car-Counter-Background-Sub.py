import pandas as pd
import numpy as np
import cv2 as cv
import math

traffic_video_one = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_one.mp4"
traffic_video_two = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_two.mp4"
traffic_video_three = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_three.mp4"
traffic_video_four = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_four.mp4"

video_capture = cv.VideoCapture(traffic_video_four)

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
video_height = 360
video_width = 640
video_capture.set(3, video_width)
video_capture.set(4, video_height)

video_result = cv.VideoWriter('video_result_V2.mp4', 0x7634706d, 15.0, (video_width, video_height), isColor=True)

background_subtractor = cv.createBackgroundSubtractorKNN()

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








