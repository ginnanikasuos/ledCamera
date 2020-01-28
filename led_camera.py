import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import cnn_model

mxpix = 255
green = (0,mxpix,0)
red = (mxpix,0,0)
blue = (0,0,mxpix)
white = (mxpix,mxpix,mxpix)

LABELS = ["red_upper","red_side","red_bottom","green_upper","green_side","green_bottom"]
model_name = "./phots-model-light.hdf5"

class ledCamera():
    def __init__(self,camera=True):
        self.image = None
        self.thresh = 200
        self.max_pixel = mxpix
        self.pic_size = 25
        self.shape = 32
        self.color = 3
        self.in_shape = (self.shape,self.shape,self.color)
        self.cap = None
        if camera:
            self.result = self.search_capture()
        else:
            self.result = False
            
        self.model = cnn_model.get_model(self.in_shape, len(LABELS))
        self.model.load_weights(model_name)
    
    def __del__(self):
        if self.result != False:
            self.cap.release()
    
    def search_capture(self):
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(i,cap.get(3),cap.get(4),cap.get(5))
                if cap.get(3) == 1280.0 and cap.get(4) == 960.0 and cap.get(5) == 25.0:
                    cap.set(cv2.CAP_PROP_FPS,10)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                    print("set camera :",i)
                    break
                cap.release()
            else:
                break
        else:
            return False
        
        self.cap = cap
        return True

    def find_led(self,cp_img):
        image = cp_img
        hsv = cv2.cvtColor(cp_img, cv2.COLOR_BGR2HSV_FULL)
        gry_img = hsv[:,:,2]
        #gry_img = cv2.cvtColor(cp_img,cv2.COLOR_BGR2GRAY)
        ret,new_img = cv2.threshold(gry_img,self.thresh,self.max_pixel,cv2.THRESH_BINARY)
        #plt.imshow(new_img)
        #plt.show()
        contours, hierarchy = cv2.findContours(
                        new_img,
                        cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_NONE
                        )
        for i in range(len(contours)):
            con = contours[i]
            hrc = hierarchy[0][i]
            mu = cv2.moments(con)
            area = mu["m00"]
            rect = cv2.minAreaRect(con)
            ((x,y),(w,h),r) = rect
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if w > h:
                length = w
            else:
                length = h
                
            #print(area,hrc,length)

            if 1000 < area and hrc[3] == -1 and 200 < length and length < 500:
                cx,cy = int(mu["m10"]/area),int(mu["m01"]/area)
                idx,per = self.led_distinction(cp_img,(cx,cy))
                if idx == False:
                    idx = 0
                    per = 0
                x0 = cx - 10
                x1 = cx + 10
                y0 = cy - 10
                y1 = cy + 10
                cv2.rectangle(cp_img,(x0,y0),(x1,y1),green,thickness=2)
                text = "{0}:{1}%({2},{3})r:{4:.2f}".format(LABELS[idx],per,cx,cy,r)
                cv2.putText(cp_img,text,(50,50),cv2.FONT_HERSHEY_PLAIN, 1.5,white,1,cv2.LINE_AA)
                cv2.rectangle(cp_img,(x0,y0),(x1,y1),red,thickness=2)
                cv2.drawContours(cp_img,[box],0,red,2)

                return (cx,cy),cp_img
        return False,new_img

    def led_distinction(self,img,center):
        cx,cy = center
        #print(center)
        x0 = cx - self.pic_size
        y0 = cy - self.pic_size
        x1 = cx + self.pic_size
        y1 = cy + self.pic_size
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > img.shape[1]:
            x1 = img.shape[1]
        if y1 > img.shape[0]:
            y1 = img.shape[0]
        cut_img = img[y0:y1,x0:x1]
        if cut_img.shape[0] == 0 or cut_img.shape[1] == 0:
            return (False,0)
        
        cut_img = cv2.resize(cut_img,(self.shape,self.shape))
        cut_img = cut_img.reshape(-1,self.shape,self.shape,self.color)
        cut_img = cut_img / 255
        pre = self.model.predict([cut_img])[0]
        idx = pre.argmax()
        per = int(pre[idx] * 100)
        #print(idx,per)

        return (idx, per)
    
    def led_count(self,img):
        # フィルタ
        #kernel = np.ones((5,5),np.float32)/25
        #filter_img = cv2.filter2D(img,-1,kernel)
        # 明度のみをグレイスケールに使用

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) 
        #h = hsv_img[:,:,0]
        #s = hsv_img[:,:,1]
        v = hsv_img[:,:,2]
        gry_img = v

        # 通常のグレイスケール
        #gry_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 青重視
        #gry_img = filter_img[:,:,1]

        # Otsu法
        thresh,bin_img = cv2.threshold(gry_img,0,mxpix,cv2.THRESH_OTSU)
        #ret,bin_img = cv2.threshold(gry_img,thresh,max_pixel,cv2.THRESH_BINARY)
        #print(thresh)

        # ノイズフィルタ
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations=2)
        # 背景
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        # 背景からの距離
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        # 前景
        ret,sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

        '''
        plt.imshow(img)
        plt.show()
        plt.imshow(gry_img,cmap='gray')
        plt.show()
        plt.imshow(bin_img,cmap='gray')
        plt.show()
        plt.imshow(sure_bg,cmap='gray')
        plt.show()
        plt.imshow(dist_transform,cmap='gray')
        plt.show()

        plt.imshow(sure_fg,cmap='gray')
        plt.show()
        '''

        contours, hierarchy = cv2.findContours(
                            sure_fg.astype(np.uint8),
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_NONE
                            )
        count = 0
        ret_center = []
        size = self.pic_size
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 2
        thickness = 2
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            if cx <= size:
                continue
            if cx >= img.shape[1] - size:
                continue
            if cy <= size:
                continue
            if cy >= img.shape[0] - size:
                continue
                
            ret_center.append((cx,cy))
            
            x0 = cx - size
            x1 = cx + size
            y0 = cy - size
            y1 = cy + size
            cv2.rectangle(img,(x0,y0),(x1,y1),(0,255,0),thickness=2)
            count += 1
            cv2.putText(img,"{}".format(count),(cx,cy),font,font_size,green,thickness,cv2.LINE_AA)
            '''
            cut_img = img[y0:y1,x0:x1]
            cut_img = cv2.resize(cut_img,(32,32))
            plt.imshow(cut_img)
            plt.show()
            '''
        
        cv2.putText(img,"{}".format(count),(10,40),font,font_size,blue,thickness,cv2.LINE_AA)
            
        return ret_center,img
    
    def release(self):
        self.cap.release()
       
    def camera_view(self):
        while True:
            # camera input capture
            ret, frame = self.cap.read()
            if ret == False:
                print("camera false")
                self.release()
                return False
            
            # find led
            center,cap_img = self.find_led(frame)
            # image show
            cv2.imshow("frame",cap_img)
            if cv2.waitKey(1) == 13: break

        self.cap.release()
        cv2.destroyAllWindows()