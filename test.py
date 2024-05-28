import cv2
import numpy as np
import glob
"""
# Function to find intersection point of two lines
def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x, y = np.linalg.solve(A, b)
    return int(x), int(y)

#file_path = "/home/psxls7/catkin_ws/src/robotic_view_capture/view_capture_scripts/logs/turntable_calibration/"
#image_paths = glob.glob(file_path + '*.png')

image_paths = ["/home/psxls7/Downloads/sudoku.png"]

images = [cv2.imread(path) for path in image_paths]

# Process each image
for idx, img in enumerate(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(f"Image {idx+1}", img)
    cv2.waitKey(0)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Draw lines on the image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show image with detected lines
    cv2.imshow(f"Image {idx+1}", img)
    cv2.waitKey(0)

# Calculate center of rotation using intersection of detected lines
if len(images) > 1:
    lines1 = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    lines2 = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines1 is not None and lines2 is not None:
        line1 = lines1[0][0]
        line2 = lines2[0][0]
        center = intersection(line1, line2)
        print("Center of rotation:", center)

cv2.waitKey(0)
cv2.destroyAllWindows()"""

import sys
import cv2
import numpy as np
import glob
import math

def main(argv):
    
    default_file = "/home/psxls7/catkin_ws/src/robotic_view_capture/view_capture_scripts/logs/turntable_calibration/turntable_pos_0.png"
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    dst = cv2.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv2.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])