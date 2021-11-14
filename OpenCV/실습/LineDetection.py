import sys
import math
import cv2, os
import numpy as np
from glob import glob

class LineDetection:

    def __init__(self, Path):
        
        self.path = Path # 나중에는 video에서 image로 받아야함.

        # Loads an image
        self.src_v1 = cv2.imread(cv2.samples.findFile(self.path))#, cv2.IMREAD_GRAYSCALE)
        self.src_v2 = self.src_v1.copy()
        self.src_v3 = self.src_v2.copy()
        self.edge = cv2.Canny(self.src_v1, 50, 200, None, 3 ) # Canny edge detector 
        self.color_edge = cv2.cvtColor(self.edge, cv2.COLOR_GRAY2BGR)

        print('shape', self.src_v1.shape) # (480, 640) = (높이, 너비)
        self.center_x = self.src_v1.shape[1] // 2 # 실수 좌표 설정하면 오류 가능성있음
        self.center_y = self.src_v1.shape[0] // 2  
    
    def hough_transform(self):
        lines_p = cv2.HoughLinesP(self.edge, 1, np.pi / 180, 50, None, 120 , 10)
        line_vertical = []
        line_horizontal = []

        if lines_p is not None:
            for i in range(0, len(lines_p)):
                l = lines_p[i][0] # 해당 직선을 지나는 두 점의 좌표 

                abs_v = abs(l[0] - l[2]) # x가 비슷하다면 수직 직선을 긋는다.
                abs_h = abs(l[1] - l[3]) # y가 비슷하다면 수평 직선을 긋는다.

                if abs_v <= 10:  
                    l[1] = 0
                    l[3] = 500
            
                    line_vertical.append([[l[0], l[1]], [l[2], l[3]]]) # (n, 2, 2)
                    cv2.line(self.color_edge, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
            
                elif abs_h <= 10:
                    l[0] = 0
                    l[2] = 500
                
                    line_horizontal.append([[l[0], l[1]], [l[2], l[3]]])
                    cv2.line(self.color_edge, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
        
        return line_vertical, line_horizontal
    
    def calculate_points(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        px = (x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)
        py= (x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)
        p = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

        x = px / p
        y = py / p

        return x, y 

    def get_intersection_point(self, lines_v, lines_h):
        points = []

        for line_v in lines_v:
            for line_h in lines_h:
                x, y = self.calculate_points(line_v[0], line_v[1], line_h[0], line_h[1])
                x = round(x)
                y = round(y)

                points.append([x, y])
        
        return points

    def draw_circle(self, is_points):
        d1, d2, d3, d4 = 1000, 1000, 1000, 1000

        for x, y in is_points:
            d = round(math.sqrt((self.center_x-x)**2 + (self.center_y-y)**2))
        
            
            if x - self.center_x > 0 and self.center_y - y > 0:
                if d < d1:
                    d1 = d
                    p1 = [x, y]
                
            elif x - self.center_x < 0 and self.center_y - y > 0:
                if d < d2:
                    d2 = d
                    p2 = [x, y]
                
            elif x - self.center_x < 0 and self.center_y - y < 0:
                if d < d3:
                    d3 = d
                    p3 = [x, y]
                
            elif x - self.center_x > 0 and self.center_y - y < 0:
                if d < d4:
                    d4 = d
                    p4 = [x, y]

            cv2.circle(self.color_edge, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(self.src_v2, (x, y), 5, (0, 0, 255), -1)

        try:
            print('1사분면 최소 거리 : {}, (x, y) = ({}, {})'.format(d1, p1[0], p1[1]))
            cv2.circle(self.color_edge, (p1[0], p1[1]), 5, (0, 255, 255), -1)
            cv2.circle(self.src_v2, (p1[0], p1[1]), 5, (0, 255, 255), -1)

            print('2사분면 최소 거리 : {}, (x, y) = ({}, {})'.format(d2, p2[0], p2[1]))
            cv2.circle(self.color_edge, (p2[0], p2[1]), 5, (0, 255, 255), -1)
            cv2.circle(self.src_v2, (p2[0], p2[1]), 5, (0, 255, 255), -1)

            print('3사분면 최소 거리 : {}, (x, y) = ({}, {})'.format(d3, p3[0], p3[1]))
            cv2.circle(self.color_edge, (p3[0], p3[1]), 5, (0, 255, 255), -1)
            cv2.circle(self.src_v2, (p3[0], p3[1]), 5, (0, 255, 255), -1)

            print('4사분면 최소 거리 : {}, (x, y) = ({}, {})'.format(d4, p4[0], p4[1]))
            cv2.circle(self.color_edge, (p4[0], p4[1]), 5, (0, 255, 255), -1)
            cv2.circle(self.src_v2, (p4[0], p4[1]), 5, (0, 255, 255), -1)
                
            cv2.rectangle(self.src_v3, (p2[0], p2[1]), (p4[0], p4[1]), (255,0,255), 3)
        
        except UnboundLocalError:
            print('사분면의 최소거리 중 일부분이 존재하지 않습니다.')
            p1 = None
            p2 = None
            p3 = None
            p4 = None
        return p2, p4
    
    def crop(self, p2, p4):
        x0, y0 = p2[0], p2[1]
        x1, y1 = p4[0], p4[1]

        self.crop_img = self.src_v1[y0:y1, x0:x1]
        
    def show_image(self):
        cv2.imshow("color_edge", self.color_edge)
        cv2.imshow("original + intersection_points", self.src_v2)
        cv2.imshow("original + ROI BOX", self.src_v3)
        cv2.imshow("crop_img", self.crop_img)

        cv2.waitKey()
    
    def start_detection(self):

        line_vertical, line_horizontal = self.hough_transform()
        is_points = self.get_intersection_point(line_vertical, line_horizontal)

        p2, p4 = self.draw_circle(is_points)
        if p2 == None:
            return 0
        self.crop(p2, p4)

        fHead = self.path.split('\\')[-2] # ./train_img, train과 realT와의 분리를 위해 
        fHead = fHead.split('_')[0] # ./train
        fHead = fHead.split('/')[-1] # train

        fName = './' + fHead + '_crop/' + self.path.split('\\')[-1]
        # cv2.imwrite(fName, self.crop_img)
        print('crop_img({}, {})를 {}에 저장하였습니다.'.format(p4[0]-p2[0], p4[1]-p2[1], fName))
        # self.show_image()

path = './train_img_g/'
img_path = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.jpg'))]
n = 1
for path in img_path:
    print('path :', path)
    test = LineDetection(path)
    if test.start_detection() != 0:
        test.show_image()
    n += 1
    
    # if n == 20:
    #     break
print('train_img ------> train_crop')

# test = LineDetection()
# test.start_detection()
# test.show_image()

        
         

