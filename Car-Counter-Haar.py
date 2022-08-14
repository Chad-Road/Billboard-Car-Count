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

# The numbers in set are property identifiers 3 = width, 4 = height
# This sets video height and width to predetermined height and width
video_height = 360
video_width = 640
video_capture.set(3, video_width)
video_capture.set(4, video_height)

# 0x7634706d is used to force opencv to use mp4 due to compatibility issues on certiain systems
# Most users can use   VideoWriter_fourcc(*'MP4V')  for mp4
video_result = cv.VideoWriter('video_result_haar.mp4', 0x7634706d, 15.0, (video_width, video_height))

car_haar_cascade = "C:\\Coding\\GitHub\\Billboard-Car-Count\\car_detection_haar.xml"
car_cascade_classifier = cv.CascadeClassifier(car_haar_cascade)


while True:
    return_value, captured_image = video_capture.read()

    if (type(captured_image) == type(None)):
        break

    # Convert image to grayscale, generally required for haar cascade
    to_grayscale = cv.cvtColor(captured_image, cv.COLOR_BGR2GRAY)

    car_detected = car_cascade_classifier.detectMultiScale(to_grayscale, 1.1, 2)

    for (x_cord, y_cord, width, height) in car_detected:
        cv.rectangle(captured_image, (x_cord, y_cord), (x_cord+width, y_cord+height), (0, 255, 255), 2)

    video_result.write(captured_image)
video_result.release()
cv.destroyAllWindows()
