#!/usr/bin/env python

import rclpy
import cv2
import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *
import time

#from portrait_capturer import PortraitCapturer
from pathlib import Path

### <use for centroid finding> start
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

#signature
def writing_path():
    gap = 1
    shift = 0
    A = [[[0,0],[5,10],[10,0]],[[2.5,5],[7.5,5]]]
    B = [[[0,0],[0,10],[7,10],[7,5],[0,5],[10,5],[10,0],[0,0]]]
    C = [[[10,0],[0,0],[0,10],[10,10]]]
    D = [[[0,0],[10,5],[0,10],[0,0]]]
    E = [[[0,0],[0,10],[10,10]],[[0,5],[10,5]],[[0,0],[10,0]]]
    F = [[[0,0],[0,10],[10,10]],[[0,5],[5,5]]]
    G = [[[10,10],[0,10],[0,0],[10,0],[10,5]]]
    H = [[[0,0],[0,10]],[[0,5],[10,5]],[[10,10],[10,0]]]
    I = [[[2.5,10],[7.5,10]],[[5,10],[5,0]],[[2.5,0],[7.5,0]]]
    J = [[[2.5,10],[7.5,10]],[[5,10],[5,0],[2.5,0]]]
    K = [[[0,0],[0,10]],[[10,10],[0,5],[10,0]]]
    L = [[[0,10],[0,0],[10,0]]]
    M = [[[0,0],[0,10],[5,0],[10,10],[10,0]]]
    N = [[[0,0],[0,10],[10,0],[10,10]]]
    O = [[[0,0],[0,10],[10,10],[10,0],[0,0]]]
    P = [[[0,0],[0,10],[10,10],[10,5],[0,5]]]
    Q = [[[0,0],[0,10],[10,10],[10,0],[0,0]],[[5,5],[10,0]]]
    R = [[[0,0],[0,10],[10,10],[10,5],[0,5],[10,0]]]
    S = [[[10,10],[0,10],[0,5],[10,5],[10,0],[0,0]]]
    T = [[[0,10],[10,10]],[[5,10],[5,0]]]
    U = [[[0,10],[0,0],[10,0],[10,10]]]
    V = [[[0,10],[5,0],[10,10]]]
    W = [[[0,10],[0,0],[5,5],[10,0],[10,10]]]
    X = [[[0,10],[10,0]],[[0,0],[10,10]]]
    Y = [[[0,10],[5,5],[10,10]],[[5,5],[5,0]]]
    Z = [[[0,10],[10,10],[0,0],[10,0]]]

    ok = 'Hello~, please enter the letter you want to show:'
    ok = input("Hello~, please enter the letter you want to show:\n")
    ok =ok.upper()
    to_write = list(ok)
    print(to_write)

    path = []
    gap = 3
    shift = 0

    for num, i in enumerate(to_write):
        shift = num*(10+gap)
        if(i == 'A'):
            A_copy = list(list((x + shift, y) for x, y in subpath) for subpath in A)
            path.extend(A_copy)
        elif(i == 'B'):
            B_copy = list(list((x + shift, y) for x, y in subpath) for subpath in B)
            path.extend(B_copy)
        elif(i == 'C'):
            C_copy = list(list((x + shift, y) for x, y in subpath) for subpath in C)
            path.extend(C_copy)
        elif(i == 'D'):
            D_copy = list(list((x + shift, y) for x, y in subpath) for subpath in D)
            path.extend(D_copy)
        elif(i == 'E'):
            E_copy = list(list((x + shift, y) for x, y in subpath) for subpath in E)
            path.extend(E_copy)
        elif(i == 'F'):
            F_copy = list(list((x + shift, y) for x, y in subpath) for subpath in F)
            path.extend(F_copy)
        elif(i == 'G'):
            G_copy = list(list((x + shift, y) for x, y in subpath) for subpath in G)
            path.extend(G_copy)
        #elif(i == 'H'):
            #path = path+H
        elif(i == 'H'):
            H_copy = list(list((x + shift, y) for x, y in subpath) for subpath in H)
            path.extend(H_copy)
        elif(i == 'I'):
            I_copy = list(list((x + shift, y) for x, y in subpath) for subpath in I)
            path.extend(I_copy)
        elif(i == 'J'):
            J_copy = list(list((x + shift, y) for x, y in subpath) for subpath in J)
            path.extend(Jcopy)
        elif(i == 'K'):
            K_copy = list(list((x + shift, y) for x, y in subpath) for subpath in K)
            path.extend(K_copy)
        elif(i == 'L'):
            L_copy = list(list((x + shift, y) for x, y in subpath) for subpath in L)
            path.extend(L_copy)
        elif(i == 'M'):
            M_copy = list(list((x + shift, y) for x, y in subpath) for subpath in M)
            path.extend(M_copy)
        elif i == 'N':
            N_copy = list(list((x + shift, y) for x, y in subpath) for subpath in N)
            path.extend(N_copy)
        elif i == 'O':
            O_copy = list(list((x + shift, y) for x, y in subpath) for subpath in O)
            path.extend(O_copy)
        elif i == 'P':
            P_copy = list(list((x + shift, y) for x, y in subpath) for subpath in P)
            path.extend(P_copy)
        elif i == 'Q':
            Q_copy = list(list((x + shift, y) for x, y in subpath) for subpath in Q)
            path.extend(Q_copy)
        elif i == 'R':
            R_copy = list(list((x + shift, y) for x, y in subpath) for subpath in R)
            path.extend(R_copy)
        elif i == 'S':
            S_copy = list(list((x + shift, y) for x, y in subpath) for subpath in S)
            path.extend(S_copy)
        elif i == 'T':
            T_copy = list(list((x + shift, y) for x, y in subpath) for subpath in T)
            path.extend(T_copy)
        elif i == 'U':
            U_copy = list(list((x + shift, y) for x, y in subpath) for subpath in U)
            path.extend(U_copy)
        elif i == 'V':
            V_copy = list(list((x + shift, y) for x, y in subpath) for subpath in V)
            path.extend(V_copy)
        elif i == 'W':
            W_copy = list(list((x + shift, y) for x, y in subpath) for subpath in W)
            path.extend(W_copy)
        elif i == 'X':
            X_copy = list(list((x + shift, y) for x, y in subpath) for subpath in X)
            path.extend(X_copy)
        elif i == 'Y':
            Y_copy = list(list((x + shift, y) for x, y in subpath) for subpath in Y)
            path.extend(Y_copy)
        elif i == 'Z':
            Z_copy = list(list((x + shift, y) for x, y in subpath) for subpath in Z)
            path.extend(Z_copy)
        elif i == ' ':
            shift = shift+5
    return(path)




#photo catch
class PortraitCapturer:
    def __init__(self, camera_id, portrait_folder):
        self.camera_id = camera_id
        self.portrait_folder = Path(portrait_folder)
        self.portrait_folder.mkdir(parents=True, exist_ok=True)

    def capture_portrait(self):
        print(f">>>> Capturing portrait")
        cv2.namedWindow('portrait', cv2.WINDOW_FULLSCREEN)

        # define video capture
        cap = cv2.VideoCapture(self.camera_id)
        while True:
            # capture video frame by frame
            ret, frame = cap.read()
            if ret:
                frame_viz = frame.copy()
                frame_viz = cv2.putText(frame_viz, "Press X to capture portrait", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
                frame_viz = cv2.putText(frame_viz, "Press Q to exit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('x'):
                    cv2.imwrite(str(self.portrait_folder / 'portrait.png'), frame)
                    print('Portrait saved!')
                elif key == ord('q'):
                    break
                cv2.imshow('portrait', frame_viz)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


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
    #print(h, w)
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
            #print(cx, cy)
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




from PIL import Image,ImageGrab




def photo_to_patch(row, column, original_image):
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    height, width = original_image.shape[:2]
    row_step = (int)(height/row)
    column_step = (int)(width/column)
    img = original_image[0:row_step*row, 0:column_step*column]
    #fig = plt.figure()
    #fig, ax = plt.subplots(row,column)
    count = 1
    all_img = []
    for i in range(row):
        for j in range(column):
            #print(i, j)
            tmp_img = original_image[(i*row_step):(i*row_step+row_step), (j*column_step):(j*column_step+column_step)]
            #cv2.imshow('1',tmp_img)
            #plt.subplot(row,column,count)
            #plt.imshow(tmp_img)
            #plt.title(title,fontsize=8)
            #cv2.imwrite(f'{i}{j}.jpg',tmp_img)
            #cv2.destroyAllWindows()
            all_img.append(tmp_img)
            count = count + 1
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('lichen_sep.jpg')

    return all_img

def process_image_patch(original_image, index, row, column, current_row, current_column, row_step, column_step, pixel_length_ratio): # Douglas-peucker approximation
    #print(i, j, row_step, column_step)
    # Convert to black and white threshold map
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    (thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    professor1 = cv2.Canny(gray, 30, 100)
    (cnts,_) = cv2.findContours(professor1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    # Convert bw image back to colored so that red, green and blue contour lines are visible, draw contours
    #modified_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    #contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(modified_image, contours, -1, (255, 0, 0), 2)
    # ret,professor1 = cv2.threshold(professor,100,255,0)
    patch_pos = index+1
    thick =3
    if (patch_pos <= row*thick) or (patch_pos >= (row*column-row*thick)):
        threshold = 0.15
    elif (patch_pos%row <= thick) or (patch_pos%row >= row-thick+1):
        threshold = 0.15
    else:
        threshold = 0.02
    #print(patch_pos, threshold)
    height, width = original_image.shape[:2]
    #print(height, width)
    #lll = np.zeros([int(250),int(200),3],dtype=np.uint8)
    lll = np.zeros([int(height),int(width),3],dtype=np.uint8)
    lll.fill(255)
    j = 1
    num = 0
    output_coor = []
    #pixel_length_ratio = 0.325493559564327
    #pixel_length_ratio = 0.6

    if cnts != ():
        for cnt_num in range(len(cnts)):
            num = np.sum(len(cnts[cnt_num])) + num
        if (num > 10) and (num <= 20):
            delete = num - 10
            threshold = threshold + delete*0.01
        elif (num > 20) and (num <= 30):
            delete = num - 10
            threshold = threshold + delete*0.001
        elif num > 30:
            delete = num - 10
            threshold = threshold + delete*0.001
        else:
            delete = 0
            #threshold = 0.005
        #print(num, delete,threshold)
        #print(f"patch{i}*{j}")
        pre_x = 0
        pre_y = 0
        for object in cnts:
            epsilon = threshold*cv2.arcLength(object,True)
            #print(f"no.{j}")
            #print(i)
            #print(j)
            #if (j % (num//(delete + 1))) != 0 or (delete == 0):
            approx = cv2.approxPolyDP(object,epsilon,True)

            for idx in range(len(approx)):
                #print(i)
                #print(approx[idx])
                x = approx[idx][0][0] + current_column*column_step
                y = approx[idx][0][1] + current_row*row_step
                output_coor.append((x,y))
                #print(x,y)
                #print(column_step,row_step, pixel_length_ratio)
                x_offset = x * pixel_length_ratio
                y_offset = y * pixel_length_ratio
                #arm_x_offset = x_offset*
                #arm_y_offset = y_offset
                arm_x_offset = x_offset*math.cos(45/180*np.pi)-y_offset*math.cos(45/180*np.pi)
                arm_y_offset = x_offset*math.cos(45/180*np.pi)+y_offset*math.cos(45/180*np.pi)
                init_x = 442
                init_y = 203
                block_point = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 230, -180.00, 0.0, 135"
                block_point_up = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 240, -180.00, 0.0, 135"
                block_point_orig = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 400, -180.00, 0.0, 135"
#411 474
                #print(arm_x_offset)
                #print(arm_y_offset)
                #print(x_offset, y_offset)
                #print(arm_x_offset, arm_y_offset)
                #print("do")
                #send_script("PTP(\"CPP\","+block_point_up+",100,200,0,false)")
                send_script("PTP(\"CPP\","+block_point+",100,200,0,false)")
                #send_script("PTP(\"CPP\","+down+",100,200,0,false)")
                #send_script("Vision_DoJob(job1)")
            #print(approx)
            image = cv2.drawContours(lll,approx,-1,(0,86,173),3)
            j = j + 1
            up = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 250, -180.00, 0.0, 135"
            send_script("PTP(\"CPP\","+up+",100,200,0,false)")
        #print(j)
    else:
        #check_first = check_first + 1
        image=lll
    return image

def process_image_big(original_image):  # Douglas-peucker approximation
    # Convert to black and white threshold map
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    (thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    professor1 = cv2.Canny(gray, 50, 150)
    (cnts,_) = cv2.findContours(professor1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    height, width = original_image.shape[:2]
    lll = np.zeros([int(height),int(width),3],dtype=np.uint8)
    lll.fill(255)
    pixel_length_ratio = 0.6
    output_coor = []
    if cnts != ():
        for i in cnts:
            epsilon = 0.005*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
            #print("==========")
            #print(approx)
            #print(len(approx))
            #print(approx[0])
            #print(approx[0][0])
            #print(approx[0][0][0])
            #print(approx[0][1][0])
            for idx in range(len(approx)):
                #print(i)
                #print(approx[idx])
                x = approx[idx][0][0]
                y = approx[idx][0][1]
                output_coor.append((x,y))
                #print(output_coor)
                x_offset = x * pixel_length_ratio
                y_offset = y * pixel_length_ratio
                arm_x_offset = -x_offset
                arm_y_offset = y_offset
                #arm_x_offset = x_offset*math.cos(45/180*np.pi)-y_offset*math.cos(45/180*np.pi)
                #arm_y_offset = x_offset*math.cos(45/180*np.pi)+y_offset*math.cos(45/180*np.pi)
                block_point = f"{250.00+arm_x_offset}, {140+arm_y_offset}, 225, -180.00, 0.0, 135"
                block_point_up = f"{250.00+arm_x_offset}, {140+arm_y_offset}, 235, -180.00, 0.0, 135"
                #print(arm_x_offset)
                #print(arm_y_offset)
                #print(x_offset)
                #print(y_offset)
                #print(arm_x_offset, arm_y_offset)
                #print("do")
                #send_script("PTP(\"CPP\","+block_point_up+",100,200,0,false)")
                #send_script("PTP(\"CPP\","+block_point+",100,200,0,false)")
                #send_script("PTP(\"CPP\","+down+",100,200,0,false)")
                #send_script("Vision_DoJob(job1)")
            image = cv2.drawContours(lll,approx,-1,(0,86,173),3)
        up = f"{250.00+arm_x_offset}, {140+arm_y_offset}, 235, -180.00, 0.0, 135"
        #send_script("PTP(\"CPP\","+up+",100,200,0,false)")
    else:
        image=lll
    return image


def screen_record(img, all_img, row, column, args=None):
    #while(True):
        #screen = np.array(ImageGrab.grab(bbox=(100, 240, 750, 600)))
    #rclpy.init(args=args)
    #set_io(1.0)
    script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'
    height, width = img.shape[:2]
    big_size = max(height, width)
    ratio = 160 / big_size
    row_step = (int)(height/row)
    column_step = (int)(width/column)
    white = np.zeros([(height-height%row),(width-width%column),3],dtype=np.uint8)
    white.fill(255)
    #or i in range(len(all_img)):
    count = 0
    for i in range(row):
        if (i%2) == 0:
            for j in range(column):
                #print(f"i:{i},j:{j},count:{count}")
                image = process_image_patch(all_img[count], count, row, column, i, j, row_step, column_step,ratio)
                white[(i*row_step):(i*row_step+row_step), (j*column_step):(j*column_step+column_step)] = image
                #plt.subplot(row,column,count+1)

                #plt.imshow(image)
                #plt.xticks([])
                #plt.yticks([])
                #print(count)
                count = count + 1

        else :
            count = count - 1 + column
            for j in range(column-1,-1,-1):
                #print(f"i:{i},j:{j},count:{count}")
                image = process_image_patch(all_img[count], count, row, column, i, j, row_step, column_step,ratio)
                white[(i*row_step):(i*row_step+row_step), (j*column_step):(j*column_step+column_step)] = image
                #plt.subplot(row,column,count+1)

                #plt.imshow(image)
                #plt.xticks([])
                #plt.yticks([])
                #print(count)
                count = count - 1
            count = count + 1 + column
    #plt.plot()
    plt.subplot(2,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.title("simplify result")
    #plt.title("split into small patches")
    plt.imshow(white)

    #plt.tight_layout()
    #plt.show()
    #print("original")


    '''
        cv2.imshow('window', lll)
        if cv2.waitKey(25) & 0xFF == ord('q'):

            cv2.imwrite('windows.png',lll)
            cv2.destroyAllWindows()
            break
    '''
 # or img[:] = 255
#img = cv2.imread("lichen_v2.jpg")

def sig_write(sig):

    pixel_length_ratio = 1
    init_x = 365
    init_y = 489
    init = f"{init_x}, {init_y}, 300, -180.00, 0.0, 135"
    send_script("PTP(\"CPP\","+init+",100,200,0,false)")
    for each_up in sig:
        for point in each_up:
            x = point[0]
            y = point[1]

            x_offset = x * pixel_length_ratio
            y_offset = y * pixel_length_ratio
            arm_x_offset = -(x_offset*math.cos(45/180*np.pi)-y_offset*math.cos(45/180*np.pi))
            arm_y_offset = -(x_offset*math.cos(45/180*np.pi)+y_offset*math.cos(45/180*np.pi))
            block_point = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 230, -180.00, 0.0, 135"
            block_point_up = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 240, -180.00, 0.0, 135"
            block_point_orig = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 400, -180.00, 0.0, 135"
            send_script("PTP(\"CPP\","+block_point+",100,200,0,false)")
            up = f"{init_x+arm_x_offset}, {init_y+arm_y_offset}, 250, -180.00, 0.0, 135"
        send_script("PTP(\"CPP\","+up+",100,200,0,false)")
        #print(j)

def operate(args=None):
    rclpy.init(args=args)
    set_io(1.0)
    sig = writing_path()
    camera_id = 0
    portrait_folder = 'photo2cartoon/out'
    image_taker = PortraitCapturer(camera_id, portrait_folder)
    image_taker.capture_portrait()
    img = cv2.imread("photo2cartoon/out/0000.png")
    row = 10
    column = 10

    #lll = np.zeros([250,200,3],dtype=np.uint8)
    #lll.fill(255) # or img[:] = 255

    all_img = photo_to_patch(row, column, img)
    screen_record(img, all_img, row, column)

    image = process_image_big(img)
    sig_write(sig)
    plt.subplot(2,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.title("big photo")
    plt.imshow(image)
    plt.savefig('result2.jpg')
    plt.show()


    set_io(1.0)


    rclpy.shutdown()

operate()





