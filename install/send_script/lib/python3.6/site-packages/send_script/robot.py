#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
from PIL import Image,ImageGrab
import cv2
import matplotlib.pyplot as plt




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
            plt.subplot(row,column,count)
            plt.imshow(tmp_img)
            #plt.title(title,fontsize=8)
            #cv2.imwrite(f'{i}{j}.jpg',tmp_img)
            #cv2.destroyAllWindows()
            all_img.append(tmp_img)
            count = count + 1
    plt.tight_layout()
    plt.show()
    plt.savefig('lichen_sep.jpg')

    return all_img

def process_image4(original_image, index, row, column):  # Douglas-peucker approximation
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
        for i in cnts:
            epsilon = threshold*cv2.arcLength(i,True)
            #print(f"no.{j}")
            #print(i)
            #print(j)
            #if (j % (num//(delete + 1))) != 0 or (delete == 0):
            approx = cv2.approxPolyDP(i,epsilon,True)
            #print(approx)
            image = cv2.drawContours(lll,approx,-1,(0,86,173),3)
            j = j + 1
        #print(j)
    else:
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
    
    if cnts != ():
        for i in cnts:
            epsilon = 0.005*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
            #print("==========")
            #print(approx)
            #print(approx[0])
            #print(approx[0][0])
            #print(approx[0][0][0])
            #print(approx[0][1][0])
            for i in range(len(approx)):
                disire_x = approx[i][0][0]
                disire_y = approx[i][0][1]
                print(disire_x,disire_y)
            image = cv2.drawContours(lll,approx,-1,(0,86,173),3)
    else:
        image=lll
    return image


def screen_record(img, all_img, row, column):
    #while(True):
        #screen = np.array(ImageGrab.grab(bbox=(100, 240, 750, 600)))
    height, width = img.shape[:2]
    row_step = (int)(height/row)
    column_step = (int)(width/column)
    white = np.zeros([(height-height%row),(width-width%column),3],dtype=np.uint8)
    white.fill(255)
    #or i in range(len(all_img)):
    count = 0
    for i in range(row):
        for j in range(column):
            image = process_image4(all_img[count], count, row, column)
            white[(i*row_step):(i*row_step+row_step), (j*column_step):(j*column_step+column_step)] = image
            #plt.subplot(row,column,count+1)
            
            #plt.imshow(image)
            #plt.xticks([])
            #plt.yticks([])
            count = count + 1
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
img = cv2.imread("lichen_v2.jpg")
row = 10
column = 10
#lll = np.zeros([250,200,3],dtype=np.uint8)
#lll.fill(255) # or img[:] = 255
all_img = photo_to_patch(row, column, img)
screen_record(img, all_img, row, column)

image = process_image_big(img)
plt.subplot(2,2,1)
plt.xticks([])
plt.yticks([])
plt.title("big photo")
plt.imshow(image)
plt.savefig('result2.jpg')




