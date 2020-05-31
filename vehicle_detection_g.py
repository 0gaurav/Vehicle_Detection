import cv2
import numpy as np
import matplotlib.pylab as plt


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

fourcc=cv2.VideoWriter_fourcc(* 'XVID')
out=cv2.VideoWriter('output.avi',fourcc,24.0,(854,480))
cap = cv2.VideoCapture('input.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
ret, frame1 = cap.read()
print(frame1.shape)
ret, frame2 = cap.read()
while cap.isOpened():
	
    fgmask = fgbg.apply(frame1)
    fgmask2 = fgbg.apply(frame2)

    diff = cv2.absdiff(fgmask, fgmask2)

    kernel = np.ones((13,13), np.uint8)
    _, thresh = cv2.threshold(diff, 107, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, kernel, iterations=1)   

    height = diff.shape[0]
    width = diff.shape[1]
    region_of_interest_vertices = [
        (0,290),
        (0, 450),
        (850,450),
        (850,290)
    ]
    cropped_image = region_of_interest(dilated, np.array([region_of_interest_vertices], np.int32),)

    no_of_vehicle = 0
    contours, _ = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h*w > 15000:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

            no_of_vehicle = no_of_vehicle+1

    cv2.putText(frame1, "Vehicle Detection Project  -GAURAV", (210,473), 3, 0.7, (0,0,0), 2)

    cv2.putText(frame1, "Vehicles Detected: " + str(no_of_vehicle), (10, 280), 3, 0.7, (230,0,0), 2)
    img = cv2.line(frame1, (0,290), (851, 290), (0,0,255), 2)
    img = cv2.line(frame1, (0,450), (851,450), (0,0,255), 2)

    out.write(frame1)
    
    cv2.imshow("Project", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(50) == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()
