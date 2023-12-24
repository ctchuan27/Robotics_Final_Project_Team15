#!/usr/bin/env python

import rclpy
import cv2
import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *
import time


### <use for centroid finding> start
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

#function 1
def calculate_line_equation(x0, y0, theta):
    m = math.tan(theta)
    b = y0 - m * x0
    return m, b

#function 2
def find_ratio(img_path):
    image = cv2.imread(img_path)
    height, width, color = image.shape
    center_point = [width/2, height/2]
    image_gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray =  cv2.GaussianBlur(image_gray,(5,5),cv2.BORDER_DEFAULT)
    ret, image_binary = cv2.threshold(image_gray, 210, 255, cv2.THRESH_BINARY)
    #-----------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    #-----------------------------------------------------------
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    block_num = 0
    cx = 0
    cy = 0
    all_x = []
    all_y = []
    #print(cnts)
    for c in cnts:
        print("in")
        M = cv2.moments(cnts[block_num])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #print(x_out, y_out)
	    #block_num=block_num+1
        all_x.append(cx)
        all_y.append(cy)
        block_num=block_num+1
#-----------------------------------------------------------
    return all_x, all_y, center_point

#function 3
def centroid_finder(img_path):
    image = cv2.imread(img_path)
    height, width, color = image.shape
    center_point = [width/2, height/2]
    print(f"size : width = {width} px, height = {height} px")
    image_gray =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray =  cv2.GaussianBlur(image_gray,(5,5),0)
    ret, image_binary = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow('1234',image_binary)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #-----------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    #-----------------------------------------------------------
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    block_num = 0
    all_x = []
    all_y = []
    all_angle = []
    for c in cnts:
        M = cv2.moments(cnts[block_num])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f"*block {block_num+1}")
        print(f" centroid : ({cx}, {cy})")
        principal_angle = 0.5 * np.arctan2(2 * M['mu11'], -1*M['mu02'] + M['mu20'])
        print(f" principal angle : {principal_angle*180/np.pi}")
        m, b = calculate_line_equation(cx, cy, principal_angle)
        x1 = cx-400
        y1 = int(m * x1 + b)
        x2 = cx+400
        y2 = int(m * x2 + b)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(image,(cx,cy),2,(0,0,255),2)
        cv2.putText(image, "centroid = ( "+str(cx)+" , "+str(cy)+" )", (cx-20, cy+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "principle angle = "+str(-1*principal_angle*180/np.pi), (cx-90, cy+40), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1, cv2.LINE_AA)
        all_x.append(cx)
        all_y.append(cy)
        callibrate_angle = -1*principal_angle*180/np.pi
        all_angle.append(callibrate_angle)
        block_num=block_num+1
#-----------------------------------------------------------
    plt.imshow(image)
    # img_data = img_plot.get_array()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.savefig('processed_image.jpg')
    plt.show()
    # cv2.imshow('first',img_data)
    # cv2.imwrite('image4.jpg',er7_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return all_x, all_y, all_angle, center_point
### <use for centroid finding> end

def centroid_finder_v2(img_path):
    img = cv2.imread(img_path)
    height, width, color = img.shape
    center_point = [width/2, height/2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(h, w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow('1234',thresh)
    #cv2.waitKey(0)
    contours, hierarchies = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(thresh.shape[:2], dtype='uint8')
    cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)

    cx_list = []
    cy_list = []
    angle_list = []
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cx_float = round(M['m10']/M['m00'], 4)
            cy_float = round(M['m01']/M['m00'], 4)       
            print(cx, cy)
        area = cv2.contourArea(c)
        if area < 1000:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        center = (int(rect[0][0]), int(rect[0][1])) 
        width = float(rect[1][0])
        height = float(rect[1][1])
        angle = float(rect[2])
        
            
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
        print(angle)
        label = "Principal angle: " + str(round(angle,4)) + " degrees"
        if angle != 0: # drop the center of the whole image
            cv2.line(img, (0, int(cy-cx*math.tan(-angle*math.pi/180))), (cx+1000, int(cy+1000*math.tan(-angle*math.pi/180))), (200, 0, 0), 3)
            gradient=math.tan(angle)
            angle_list.append(round(-angle, 4))
            if cx != 359 and cy != 239:
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                cx_list.append(cx_float)
                cy_list.append(cy_float)

    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    k = 0
    output_text = ""
    while k < len(angle_list) :
        output_text = output_text + f"Center {k+1}: (" + str(cx_list[k]) + "," + str(cy_list[k])+f")\n Principal angle {k+1}: " + str(angle_list[k]) + "  degree\n"
        k = k + 1
    cv2.imshow('1234',result)

    cv2.waitKey(1)

    cv2.destroyAllWindows()
    return cx_list, cy_list, angle_list, center_point




# arm client
def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()
    return None

# gripper client
def set_io(state):
    gripper_node = rclpy.create_node('gripper')
    gripper_cli = gripper_node.create_client(SetIO, 'set_io')

    while not gripper_cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not availabe, waiting again...')
    
    io_cmd = SetIO.Request()
    io_cmd.module = 1
    io_cmd.type = 1
    io_cmd.pin = 0
    io_cmd.state = state
    gripper_cli.call_async(io_cmd)
    gripper_node.destroy_node()
    return None

def ratio_show(args=None):
    print("ratio finder start")
    rclpy.init(args=args)
    script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'
    init_coordinate = "240.00, 140, 500, -180.00, 0.0, 135.00"
    send_script("PTP(\"CPP\","+init_coordinate+",100,200,0,false)")
    send_script("Vision_DoJob(job1)")
    time.sleep(5)
    observe_x = 240-328.21
    observe_y = 140-330.94
    all_x, all_y, center_point = find_ratio("init_camview.jpg")
    #center_x = center_point[0]
    #center_y = center_point[1]
    print(all_x[0],all_y[0],all_x[1],all_y[1])
    #x_offset = (x_out - center_x)
    #y_offset = (y_out - center_y)
    x_offset = (all_x[0] - all_x[1])
    y_offset = (all_y[0] - all_y[1])
    #print(x_offset,y_offset)
    arm_x_offset = -x_offset*math.cos(45/180*np.pi)+y_offset*math.cos(45/180*np.pi)
    arm_y_offset = x_offset*math.cos(45/180*np.pi)+y_offset*math.cos(45/180*np.pi)
    #print(arm_x_offset,arm_y_offset)
    ratio1 = observe_x / arm_x_offset
    ratio2 = observe_y / arm_y_offset
    print(f"ratio x : {ratio1}, ratio y : {ratio2}")
    pixel_length_ratio = (ratio1 + ratio2) / 2
    print(f"average ratio : {pixel_length_ratio}")

    set_io(1.0) # 1.0: close gripper, 0.0: open gripper
    # set_io(0.0)
    rclpy.shutdown()
    return None

def main(args=None):

    rclpy.init(args=args)

    #--- move command by joint angle ---#
    script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'

    #--- move command by end effector's pose (x,y,z,a,b,c) ---#
    #targetP1 = "398.97, -122.27, 748.26, -179.62, 0.25, 90.12"s

    #Initial camera position for taking image (Please do not change the values)
    #For right arm: targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    #For left  arm: targetP1 = "350.00, 350, 730, -180.00, 0.0, 135.00"

    target = ["78.00, 219, 306, 179.00, 34.0, 146.00",
                "381.00, 293, 348, -148.00, 5.0, 118.00",
                "429.00, -33, 357, -140.00, 7.0, 9.00",
                "463.00, 169, 296, -143.00, 0.0, 89.00",]
    target_v1 = ["200.00, 80, 500, -180.00, 0.0, 135.00",
                "200.00, 140, 500, -180.00, 0.0, 135.00",
                "240.00, 80, 500, -180.00, 0.0, 135.00",
                "240.00, 140, 500, -180.00, 0.0, 135.00",
                "220.00, 210, 500, -180.00, 0.0, 135.00"]
    init_coordinate = "270.00, 290, 500, -180.00, 0.0, 135.00"
    #script1 = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
    #script2 = "PTP(\"CPP\","+targetP2+",100,200,0,false)"
    #send_script(script1)
    #send_script(script2)

# What does Vision_DoJob do? Try to use it...
# -------------------------------   ------------------
    # for i in range(4):
    #     send_script("PTP(\"CPP\","+target[i]+",100,200,0,false)")
    #     if i == 0:
    #         time.sleep(10) 
    #     else:
    #         time.sleep(10)
    #     send_script("Vision_DoJob(job1)")
    #     #cv2.waitKey(1)
    #     print(i)
    #     image = cv2.imread("image.jpg")
    #     cv2.imwrite(f'image{i}.jpg',image)
    #send_script("PTP(\"CPP\","+init_coordinate+",100,200,0,false)")
    #send_script("Vision_DoJob(job1)")
    #time.sleep(10)

    #image = cv2.imread("image.jpg")
    #cv2.imwrite(f'image{8}.jpg',image)
    # send_script("Vision_DoJob(job1)")
    # cv2.waitKey(1)
    #cv2.destroyAllWindows
    send_script("PTP(\"CPP\","+init_coordinate+",100,200,0,false)")
    #time.sleep(6)
    send_script("Vision_DoJob(job1)")
    time.sleep(15)
    pixel_length_ratio = 0.325493559564327
    all_x, all_y, all_angle, center_point = centroid_finder_v2("init_camview.jpg")
    #print(len(all_x))
    for idx in range(len(all_x)):
        print("in")
        set_io(0.0)
        x = all_x[idx]
        y = all_y[idx]
        angle_offset = all_angle[idx]
        center_x = center_point[0]
        center_y = center_point[1]
        x_offset = (x - center_x) * pixel_length_ratio
        y_offset = (y - center_y) * pixel_length_ratio
        arm_x_offset = x_offset*math.cos(45/180*np.pi)-y_offset*math.cos(45/180*np.pi)+51.5+3.8
        arm_y_offset = -x_offset*math.cos(45/180*np.pi)-y_offset*math.cos(45/180*np.pi)+63
        final_angle = 135+90-angle_offset
        if final_angle >= 180:
            final_angle = final_angle - 360
        elif final_angle < -180:
            final_angle = final_angle + 360
        print(arm_x_offset, arm_y_offset, final_angle)
        block_point = f"{270.00+arm_x_offset}, {290+arm_y_offset}, 200, -180.00, 0.0, {final_angle}"
        down = f"{270.00+arm_x_offset}, {290+arm_y_offset}, 110, -180.00, 0.0, {final_angle}"
        #print(block_point)
        #print("ready to move")
        send_script("PTP(\"CPP\","+block_point+",100,200,0,false)")
        send_script("PTP(\"CPP\","+down+",100,200,0,false)")
        #send_script("Vision_DoJob(job1)")
        destination = ["270.00, 290, 110, -180.00, 0.0, 135.00",
                "270.00, 290, 140, -180.00, 0.0, 135.00",
                "270.00, 290, 170, -180.00, 0.0, 135.00",]
        pause_point = "270.00, 290, 350, -180.00, 0.0, 135.00"
        set_io(1.0)
        send_script("PTP(\"CPP\","+block_point+",100,200,0,false)")
        send_script("PTP(\"CPP\","+destination[idx]+",100,200,0,false)")
        set_io(0.0)
        send_script("PTP(\"CPP\","+pause_point+",100,200,0,false)")
        
        #time.sleep(15)
#--------------------------------------------------
    
    #set_io(1.0) # 1.0: close gripper, 0.0: open gripper
    # set_io(0.0)
    
    return None

#ratio_show()

if __name__ == '__main__':
    
    main()
    rclpy.shutdown()


    
