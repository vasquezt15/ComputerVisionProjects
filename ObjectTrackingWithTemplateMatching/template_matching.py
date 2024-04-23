import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

NUMBER_OBJECTS = 3

# define a video capture object
vid = cv.VideoCapture(0)

ret, frame = None, None
ROIs = []
labels =[]
#The following loop will prompt the user to take a picture as many time as there are objects to 
while(True):
        ret, frame = vid.read()
        # Display the resulting frame
        cv.imshow('frame', frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
img = frame
while (len(ROIs) < NUMBER_OBJECTS):
    assert img is not None, "file could not be read, check with os.path.exists()"
    #img_copy = img.copy()
    ROIs.append(cv.selectROI("select the area", img))
    cv.destroyAllWindows()

labels.append(input("Enter the label of the first object: "))
labels.append(input("Enter the label of the second object: "))
labels.append(input("Enter the label of the thrid object: "))
# Crop image
templates = [img[int(r[1]):int(r[1]+r[3]), 
                int(r[0]):int(r[0]+r[2])] for r in ROIs]

ws = []
hs = []

for template in templates:
    assert template is not None, "file could not be read, check with os.path.exists()"
    ws.append(template.shape[1])
    hs.append(template.shape[0])

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# methods = [methods[0]]
# ret2, frame2 = vid.read()
# print(ret2)

values = []

for meth in methods:
    vals = []
    while(True):

        # Capture the video frame by frame
        ret, frame = vid.read()

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        img = frame.copy()
        method = eval(meth)

        # Apply template Matching
        res = [cv.matchTemplate(img,template,method) for template in templates]

        min_val_1, max_val_1, min_loc_1, max_loc_1 = cv.minMaxLoc(res[0])
        min_val_2, max_val_2, min_loc_2, max_loc_2 = cv.minMaxLoc(res[1])
        min_val_3, max_val_3, min_loc_3, max_loc_3 = cv.minMaxLoc(res[2])

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left_1 = min_loc_1
            top_left_2 = min_loc_2
            top_left_3 = min_loc_3
        else:
            top_left_1 = max_loc_1
            top_left_2 = max_loc_2
            top_left_3 = max_loc_3
        
        bottom_right_1 = (top_left_1[0] + ws[0], top_left_1[1] + hs[0])
        bottom_right_2 = (top_left_2[0] + ws[1], top_left_2[1] + hs[1])
        bottom_right_3 = (top_left_3[0] + ws[2], top_left_3[1] + hs[2])
        cv.putText(img, meth, (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # threshold = 0.8

        #The lower the number the better
        if method == cv.TM_CCOEFF:
            #threshold = 4251663
            threshold = .8
        if method == cv.TM_CCOEFF_NORMED:
            threshold = 0.677
        if method == cv.TM_CCORR:
            #threshold = 143577735
            threshold = 0.98
        if method == cv.TM_CCORR_NORMED:
            threshold = 0.978
        if method == cv.TM_SQDIFF:
            threshold = 0.057#3203056
        if method == cv.TM_SQDIFF_NORMED:
            threshold = 0.057

        if (res[0][top_left_1[1]][top_left_1[0]] >= threshold):
            cv.rectangle(img,top_left_1, bottom_right_1, 255, 2)
            cv.putText(img, labels[0], (top_left_1[0], top_left_1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if (res[1][top_left_2[1]][top_left_2[0]] >= threshold):
            cv.rectangle(img,top_left_2, bottom_right_2, 255, 2)
            cv.putText(img, labels[1], (top_left_2[0], top_left_2[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if (res[2][top_left_3[1]][top_left_3[0]] >= threshold):
            cv.rectangle(img,top_left_3, bottom_right_3, 255, 2)
            cv.putText(img, labels[2], (top_left_3[0], top_left_3[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.imshow("Cropped image", img)
        vals.append(res[0][top_left_1[1]][top_left_1[0]])
    values.append(vals)

# for i in range(len(methods)):
#     print(i)
#     print("max:")
#     print(max(values[i]))
#     print("min:")
#     print(min(values[i]))
#     print("sum:")
#     print(sum(values[i])/len(values[i]))

# t_0 = 4251663
# t_1 = 0.677
# t_2 = 143577735
# t_3 = 0.978
# t_4 = 3203056
# t_5 = 0.057

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()