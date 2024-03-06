import cv2  # Import the OpenCV library for computer vision tasks.
import time  # Import the time library for time-related tasks.

video = cv2.VideoCapture(0)  # Start video capture from the default camera (usually the webcam).

first_frame = None  # Initialize the variable to store the first frame for background subtraction.
frame_update_time = 10  # Time in seconds after which the first frame will be updated
first_frame_set_time = None  # Capture the current time when the first frame is set

while True:  # Start an infinite loop to continuously capture frames.
    check, frame = video.read()  # Read a frame from the video capture object.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale to reduce complexity.
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Apply Gaussian blur to smooth the frame, reducing noise and detail.

    if first_frame is None:  # Check if the first frame has not been captured yet.
        first_frame = gray  # Set the current frame as the first frame.
        first_frame_set_time = time.time()  # Update the time when the first frame is set
        continue  # Skip the rest of the loop to update the first_frame with the first captured frame.

    time_elapsed = time.time() - first_frame_set_time  # Calculate the time elapsed since the first frame was set

    # If the time elapsed exceeds the frame update time, reset the first frame
    if time_elapsed > frame_update_time:
        first_frame = gray
        first_frame_set_time = time.time()  # Reset the timer for the first frame update

    delta_frame = cv2.absdiff(first_frame, gray)  # Calculate the absolute difference between the first frame and the current frame.
    threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]  # Apply thresholding to highlight significant differences.
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=1)  # Dilate the thresholded frame to fill in gaps.

    (cntr, _) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the thresholded frame.

    for contour in cntr:  # Loop through each contour found.
        if cv2.contourArea(contour) < 1000:  # If the contour area is less than 1000, ignore it.
            continue
        (x, y, w, h) = cv2.boundingRect(contour)  # Get the bounding box for the contour.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle around the detected motion.

    cv2.imshow("AI_Detector", frame)  # Display the frame with detected motion in a window.
    key = cv2.waitKey(1)  # Wait for 1 millisecond and check if a key is pressed.
    if key == ord('q'):  # If the 'q' key is pressed, break the loop.
        break

video.release()  # Release the video capture object.
cv2.destroyAllWindows()  # Close all OpenCV windows.
