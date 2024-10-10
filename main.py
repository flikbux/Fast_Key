#test
import cv2
import numpy as np
from PIL import Image as im

# Read the original image
img = cv2.imread('test.jpg')
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
#cv2.imshow('Sobel X', sobelx)
#cv2.waitKey(0)
#cv2.imshow('Sobel Y', sobely)
#cv2.waitKey(0)
#cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
#cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
data = im.fromarray(edges)
data.save('gfg_dummy_pic.png')
for i in range(len(edges)):
    for j in range(len(edges[i])):
        print(edges[i][j], end=' ')
cv2.destroyAllWindows()
#----------------------- partie detect circle
edgepic = cv2.imread('gfg_dummy_pic.png')
if edges is not None:
    # Detect circles using Hough Transform
    detected_circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,  # Corrected method
        dp=1,  # Inverse ratio of the accumulator resolution to the image resolution
        minDist=10,  # Minimum distance between detected centers
        param1=100,  # Upper threshold for the Canny edge detector
        param2=50,  # Accumulator threshold for the circle centers at the detection stage
        minRadius=10,  # Minimum circle radius
        maxRadius=0   # Maximum circle radius
    )

    # If some circles are detected, draw them
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))  # Round to integers
        print("Detected circles: ", detected_circles)

        for pt in detected_circles[0, :]:  # Loop through the detected circles
            a, b, r = pt[0], pt[1], pt[2]  # Extract circle parameters

            # Draw the circumference of the circle.
            cv2.circle(edgepic, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(edgepic, (a, b), 1, (0, 0, 255), 3)

        # Display the result
        cv2.imshow("Detected Circle", edgepic)
        cv2.waitKey(0)

cv2.destroyAllWindows()