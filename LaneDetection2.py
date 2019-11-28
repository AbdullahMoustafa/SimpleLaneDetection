import cv2 
import numpy as np 
import matplotlib.pyplot as plt

#canny algorithm
def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)   #convert image to gray to be one layer
    blur = cv2.GaussianBlur(gray,(5,5),0)  #To reduce noise in gray sacle image
    canny = cv2.Canny(blur,50,150)  #traces the strongest gradiant 0-255 where inputs are Lowest and Highest thershold
    return canny


def make_coordinates(image,line_parameters):      #to draw line from the bottom of the frame
    slope , intercept = line_parameters
    y1 = image.shape[0]
    y2 = int (y1*(3/5))

    x1 = int((y1-intercept) /slope)
    x2 = int((y2-intercept) /slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):    #review  #to get the average slope and intercept so we could draw a forward line
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope , intercept))
        else:
            right_fit.append((slope , intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)

    return np.array([left_line,right_line])

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height) , (1100,height) , (550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask ,polygons,255)
    masked_image = cv2.bitwise_and(image,mask) 
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1,y1), (x2,y2) , (255,0,0) , 10)
    return line_image 


#image = cv2.imread('test_image.png')
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2 , np.pi/180 ,100 , np.array([]) , minLineLength=40 , maxLineGap=5)   #review
#averaged_lines = average_slope_intercept(lane_image , lines)
#line_image = display_lines(lane_image, averaged_lines)

#combo_image = cv2.addWeighted(lane_image , 0.8 , line_image , 1 , 1)   #0.8 is used to make the first image a little bit darker
##0.8 is the % of the brightness from the orignal image
##1 will make the lines image more lighter
##1 is scaler gamma coeff

##plt.imshow(canny_image)   #to get the needed dimenions to set the polygon on interest 
##plt.show()
#cv2.imshow("result",combo_image)
#cv2.waitKey(0)


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()   #first retured value is boolen and the second one is the frame 
    
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2 , np.pi/180 ,100 , np.array([]) , minLineLength=40 , maxLineGap=5)   #review
    averaged_lines = average_slope_intercept(frame , lines)
    line_image = display_lines(frame, averaged_lines)

    combo_image = cv2.addWeighted(frame , 0.8 , line_image , 1 , 1)
    cv2.imshow("result",combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'): #to leave 1msec between each frame in the video
        break
cap.release()
cv2.distoryAllWindows()