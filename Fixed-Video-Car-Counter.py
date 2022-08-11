import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image

# car_haar_cascade = "C:\\Coding\\GitHub\\Billboard-Car-Count\\car_detection_haar.xml"
traffic_video_one = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_one.mp4"
traffic_video_two = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_two.mp4"
traffic_video_three = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_three.mp4"
traffic_video_four = "C:\\Coding\\GitHub\\Billboard-Car-Count\\media\\traffic_four.mp4"

video_capture = cv.VideoCapture(traffic_video_four)

total_frames_count = video_capture.get(cv.CAP_PROP_FRAME_COUNT)
frames_per_second = video_capture.get(cv.CAP_PROP_FPS)

# The numbers in set are property identifiers 3 = width, 4 = height
# This sets video height and width to predetermined height and width
video_height = 360
video_width = 640
video_capture.set(3, video_width)
video_capture.set(4, video_height)

carscrosseddown = 0  # keeps track of cars that crossed down

frame_number = 0
cars_crossed_line = 0
total_cars_seen = 0

car_ids = []
car_ids_crossed = []

car_df = pd.DataFrame(index=range(int(total_frames_count)))
car_df.index.name = "Frame Number"


#--- This is inlcluded to able to easily compare results with pre-build car haar cascade ---#
###############################################################################################
# video_result = cv.VideoWriter('video_result.mp4', 0x7634706d, 15.0, (video_width, video_height))

# while True:
#     return_val, captured_image = video_capture.read()

#     if (type(captured_image) == type(None)):
#         break

#     to_grayscale = cv.cvtColor(captured_image, cv.COLOR_BGR2GRAY)

#     car_detected = car_cascade_classifier.detectMultiScale(to_grayscale, 1.1, 2)

#     for (x_cord, y_cord, width, height) in car_detected:
#         cv.rectangle(captured_image, (x_cord, y_cord), (x_cord+width, y_cord+height), (0, 255, 255), 2)

#     video_result.write(captured_image)
# video_result.release()
# cv.destroyAllWindows()
########################################################################################################


video_result = cv.VideoWriter('video_result.mp4', 0x7634706d, 15.0, (video_width, video_height), isColor=True)
# video_result = cv.VideoWriter('video_result.mp4', 0x7634706d, 15.0, (video_width, video_height), isColor=False)

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

            # Extra processing to get crisp countour lines around vehicles
            hull = cv.convexHull(cnt)
            epsilon = 0.1 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            x_cord, y_cord, width, height = cv.boundingRect(cnt)
            bounding_boxes = cv.rectangle(captured_image, (x_cord, y_cord), (x_cord + width, y_cord + height), (0, 255, 0), 3)

        # Line after which to start plotting centroids of contours
        start_line_height = 240
        start_line = cv.line(captured_image, (0, start_line_height), (640, start_line_height), (0, 100, 255), 2)

        # Line which if a centroid crosses, it is counted as a car
        count_line_height = 340
        count_line = cv.line(captured_image, (0, count_line_height), (640, count_line_height), (0, 0, 255), 6)

        # Sets minimum size for counting of contour
        minimum_contour_size = 40

        # Sets maximum contour size, necessary for early stages of video while contours are computed
        maximum_contour_size = 1500

        # x and y locations of contour centroids for the current frame
        x_cord_centroids = np.zeros(len(contours))
        y_cord_centroids = np.zeros(len(contours))


        # Iterates through all the identified contours in the current frame
        for i in range(len(contours)):

            # Singling out contours that are not contained within other contours
            if hierarchy[0, i, 3]:

                contour_area = cv.contourArea(contours[i])

                # Identifying countours that meet our countour size requirements
                if maximum_contour_size > contour_area and contour_area > minimum_contour_size:

                    # Single out contour and identify its moment (centroid)
                    current_contour = contours[i]
                    centroid_moment = cv.moments(current_contour)

                    # Starndard formula to find centroids in a given figure
                    contour_x_cord = int(centroid_moment['m10'] / centroid_moment['m00'])
                    contour_y_cord = int(centroid_moment['m01'] / centroid_moment['m00'])

                    # Only add centroids that have passed given y coordinate
                    if contour_y_cord < start_line_height:

                        x_cord_centroids[i] = contour_x_cord
                        y_cord_centroids[i] = contour_y_cord


        x_cord_centroids = x_cord_centroids[x_cord_centroids != 0]
        y_cord_centroids = y_cord_centroids[y_cord_centroids != 0]



        minimum_x_index_list = []
        minimum_y_index_list = []

        maximum_centroid_movement = 50

        if len(x_cord_centroids):

            if not car_ids:

                for i in range(len(x_cord_centroids)):

                    car_ids.append(i)
                    car_df[str(car_ids[i])] = ""

                    car_df.at[int(frame_number), str(car_ids[i])] = [x_cord_centroids[i], y_cord_centroids[i]]

                    total_cars_seen = car_ids[i] + 1

            else:

                change_in_x_cord = np.zeros((len(x_cord_centroids), len(car_ids)))
                change_in_y_cord = np.zeros((len(y_cord_centroids), len(car_ids)))

                for i in range(len(x_cord_centroids)):

                    for j in range(len(car_ids)):

                        previous_frame_centroid = car_df.iloc[int(frame_number - 1)][str(car_ids[j])]

                        current_frame_centroid = np.array([x_cord_centroids[i], y_cord_centroids[i]])

                        if not previous_frame_centroid:
                            continue

                        else:

                            change_in_x_cord[i, j] = previous_frame_centroid[0] - current_frame_centroid[0]
                            change_in_y_cord[i, j] = previous_frame_centroid[1] - current_frame_centroid[1]

                for j in range(len(car_ids)):

                    sum_of_changes = np.abs(change_in_x_cord[:, j]) + np.abs(change_in_y_cord[:, j])

                    car_ids_with_min_index = np.argmin(np.abs(sum_of_changes))

                    minimum_x_index = car_ids_with_min_index
                    minimum_y_index = car_ids_with_min_index

                    minimum_change_x_index = change_in_x_cord[minimum_x_index, j]
                    minimum_change_y_index = change_in_y_cord[minimum_y_index, j]

                    if minimum_change_x_index == 0 and minimum_change_y_index == 0 and np.all(change_in_x_cord[:, j]== 0) and np.all(change_in_y_cord[:, j] ==0):
                        continue

                    else:

                        if np.abs(minimum_change_x_index) < maximum_centroid_movement and np.abs(minimum_change_y_index) < maximum_centroid_movement:

                            car_df.at[int(frame_number), str(car_ids[j])] = [x_cord_centroids[minimum_x_index], y_cord_centroids[minimum_y_index]]
                            minimum_x_index_list.append(minimum_x_index)
                            minimum_y_index_list.append(minimum_y_index)

                for i in range(len(x_cord_centroids)):

                    if i not in minimum_x_index_list and minimum_y_index_list:

                        car_df[str(total_cars_seen)] = ""
                        total_cars_seen += 1
                        temp_total_cars = total_cars_seen -1
                        car_ids.append(temp_total_cars)
                        car_df.at[int(frame_number), str(temp_total_cars)] = [x_cord_centroids[i], y_cord_centroids[i]]


        cars_on_frame = 0
        current_cars_index = []

        for i in range(len(car_ids)):
            
            if car_df.at[int(frame_number), str(car_ids[i])] != "":

                cars_on_frame += 1
                current_cars_index.append(i)

        for i in range(cars_on_frame):

            

            current_frame_car_centroid = car_df.iloc[int(frame_number)][str(car_ids[current_cars_index[i]])]

            previous_frame_car_centroid = car_df.iloc[int(frame_number -1)][str(car_ids[current_cars_index[i]])]

            if current_frame_car_centroid:

                cv.drawMarker(captured_image, (int(current_frame_car_centroid[0]), int(current_frame_car_centroid[1])), (0, 0, 255), cv.MARKER_STAR, markerSize=5,
                thickness=1, line_type=cv.LINE_AA)

                if previous_frame_car_centroid:

                    x_cord_starting = previous_frame_car_centroid[0] - maximum_centroid_movement
                    y_cord_starting = previous_frame_car_centroid[1] - maximum_centroid_movement

                    x_width = previous_frame_car_centroid[0] + maximum_centroid_movement
                    y_width = previous_frame_car_centroid[1] + maximum_centroid_movement

                    if (previous_frame_car_centroid[1] <= count_line_height and
                        current_frame_car_centroid[1] >= count_line_height and
                        car_ids[current_cars_index[i]] not in car_ids_crossed):
                        
                        print("thing happened")

                        carscrosseddown += 1

                        cv.line(captured_image, (0, count_line_height), (720, count_line_height), (0, 0, 125), 5)
                        car_ids_crossed.append(current_cars_index[i])

        # print("rectangle is executed")
        cv.rectangle(captured_image, (0, 0), (140, 20), (255, 0, 0), -1)
        cv.putText(captured_image, "Total Cars = " + str(len(car_ids)), (3, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv.putText(captured_image, "Frame number: " + str(frame_number), (3, 350), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)

        # cv.imshow('grayed blured', binary_image)


        video_result.write(captured_image)

        frame_number += 1

        # k = cv.waitKey(int(1000/frames_per_second)) & 0xff
        # if k == 27:
        #     break

    else:
        break

video_result.release()
cv.destroyAllWindows()








