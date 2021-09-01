import cv2

# Integer for the webcam to be used or the string URL of the IP camera
VIDEO_STREAM = 0
# Location of the classifier
CLASSIFIER_LOCATION = "haarcascade_frontalface_default.xml"

def showCapture():
    # Initiate the video capture
    cam = cv2.VideoCapture(VIDEO_STREAM)
    # Initiate the cascade
    faceCascade = cv2.CascadeClassifier(CLASSIFIER_LOCATION)

    while cam.isOpened():
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Multi classify the grayscale version of the current frame and get the coordinates of the bounding rectangles
        faces = faceCascade.detectMultiScale(gray, 10, 5)

        # Draw the rectangles
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, w + h), (0, 255, 0), 2)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        cv2.imshow("Detected faces", frame)

showCapture()