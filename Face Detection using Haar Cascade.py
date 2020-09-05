"""Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images."""


"""Read the documentation over Face Detection using Haar Cascade Classifier
Visit:- https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html"""

"""Haar Cascade is an older approach to detect any face over an image but the modern approach is by use of CNN"""

"""Haar Cascade basically has some rectangles of some images such as eyes, nose, mouth, ears which is traversed through the entire image and features of both the matrices (rectangle, and actual image's) values are matched. These rectangles are called as KERNELS"""

# Import CV2
import cv2

# capture image input
cap = cv2.VideoCapture(0)  # By deafult the id for system webcam is 0


# Using Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


# Reading an frame from video
while True:
    ret, frame = cap.read()
    # ret refers to boolean True = read, False = Not captured image/frame due to some fault
    
    
    if(ret == False):
        continue
        
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # (image_frame, scaling factor, No. of neighbors)
    """
    What does scaling factor defines? :- Let say the classifier is trained over an image of 400X400 but our image size is 600X400 thus
    to implement the classifier over our image we need to shrink the image size thus scaling factor shrinks the image at a factor of 1.3 
    as the value given.
    
    What does No.of Neighbors define? :- It tells that how many neighbors the rectangle(kernel) should look for since it is implemented
    using KNN
    
    face_cascade.detectMultiScale() function returns the left upper corner coordinate of the image as 'x' and 'y' and width 'w' and height 'h' as (x, y, w, h) of the face assuming as of a rectangle
    
                                             -------w-------
                                        (x, y)_____________
                                           | |             |
                                           | |             |
                                           h |             |
                                           | |             |
                                           | |             |
                                           | |_____________|
                                                           (x+w, y+h)
     """
    

    
    
    # drawing the face rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        """"cv2.rectangle is a method to draw a rectangle which takes parameter as
        (image/stream name, up left coordinate, bottom right coordinate, color of boundary in BGR, line thickness) """
   
    
    # showing the stream
    cv2.imshow("Video Frame", frame)
    
    
    # wait for user input to stop capturing
    key_pressed = cv2.waitKey(1) & 0xFF
    
    
    # until 'q' is pressed the stram will not stop (not even if you press the close button)
    if(key_pressed == ord('q')):
        break
        
cap.release()
cv2.destroyAllWindows()
    