import cv2
import numpy as np

"""用于显示图片"""
def cv_show(name,img):
    img=cv2.resize(img,(500,500))
    img=cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

"""图像尺寸变换函数"""
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

"""寻找边界点的函数"""
def Order_points(pts):
    #一共四个坐标点
    rect=np.zeros((4,2),dtype='float32')

    #按顺序找到对应坐标，0，1,2,3分别是左上、右上、右下、左下
    #计算左上、右下
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]   #取S中最小值的索引
    rect[2]=pts[np.argmax(s)]   #取S中最大值的索引

    #计算右上和左下
    diff=np.diff(pts,axis=1)     #np.diff():矩阵中后一个元素减去前一个元素的差值,axis=1的情况,就是从后向前依次横着减
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]

    return  rect      #返回矫正后的四个顶点坐标

"""透视变换函数"""
def Four_point_transfrom(image,pts):       #输入原始图像和查找到的轮廓的顶点
    #获取输入的坐标点
    rect=Order_points(pts)
    (tl,tr,br,bl)=rect

    #计算输入的W和H值
    widthA = np.sqrt(((br[0] - bl[0]) **2 ) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    #变换后对应坐标位置
    dst=np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype='float32')

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))        #进行放射变换

    # 返回变换后结果
    return warped

"""图像增强并锐化:
    图像锐化主要影响图像中的低频分量，不影响图像中的高频分量。
    运用图像的基本加减运算 与 opencv 自带的高斯模糊函数求得
    对灰度图做高斯模糊
    mask = 原灰度图-模糊图
    锐化图= 原灰度图 + mask 
"""
def Strong_pic(input_image):
    # input_image=cv2.imread(input_image,cv2.IMREAD_GRAYSCALE)      #读取图片并将图片灰度化
    # input_image = cv2.imread(input_image)
    #图片进行高斯模糊
    # blur = cv2.GaussianBlur(input_image,(3,3),0)
    #创建掩码
    # backimage = cv2.bitwise_not(input_image, blur)
    # #锐化图
    # backimage = cv2.bitwise_not(input_image, backimage)

    # backimage = cv2.cvtColor(backimage,cv2.COLOR_GRAY2RGB)
    #图像增强
    out = 1.5 * input_image
    # 进行数据截断，大于255的值截断为255
    out[out > 255] = 255
    # 数据类型转换
    out = np.around(out)
    out = out.astype(np.uint8)

    #二值化
    # rec,thre=cv2.threshold(backimage,100,255, cv2.THRESH_TOZERO_INV )
    # thre=cv2.adaptiveThreshold(backimage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img = resize(out, height=500)  # 图片缩小
    cv2.imshow("img", img)
    # cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return   img

def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
"""透视变换，自动找到需要识别的矩形边界
1、还是要进行预处理识别轮廓边界
2、如果图片本身就是一个轮廓就不需要裁剪，所以要对轮廓进行判断后在进行裁剪
3、图像裁剪
4、图像透视变换进行校正
"""
def pic_process(input_img):
    # 读取输入
    image = imreadex(input_img)
    # 坐标也会相同变化
    ratio = image.shape[0] / 500.0
    orig = image.copy()     #赋值一张图片
    image=resize(orig,height=500)   #将图片的高度变为500，然后等比例缩放

    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     #图片灰度化
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)           #高斯滤波
    median = cv2.medianBlur(gray, 5)
    rec,threshold=cv2.threshold(median,127,255,cv2.THRESH_TOZERO_INV)
    # 展示预处理结果
    print("STEP 1: 边缘检测")
    cv2.imshow("Image", image)
    cv2.imshow("gray", threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #对边缘检测的图片做形态学处理
    kernel = np.ones((5, 5), np.uint8)    #定义核的大小
    dilate = cv2.dilate(gray, kernel,iterations=1)
    cv_show('closing',dilate)
    erode = cv2.erode(dilate, kernel, iterations=1)
    edged = cv2.Canny(erode, 75, 200)  # candy边缘检测
    cv_show('edged', edged)
    result = image.copy()
    """霍夫直线检测，，检测不到边缘"""
    # lines=cv2.HoughLinesP(edged,1.0,np.pi/180,100)
    # for x1, y1, x2, y2 in lines[0]:
    #     cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.imshow("gray", result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #轮廓检测：RETR_EXTERNAL:查找外轮廓
    """角点检测"""

    """漫水填充"""

    img,cnts,heri=cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]     #对轮廓按照面积大小进行排序
    """计算图片的面积"""

    #遍历轮廓
    for c in cnts:
        #计算轮廓相似，计算得到的四个边
        peri = cv2.arcLength(c, True)    #轮廓的周长
        # C表示输入的点集
        # 0.02*peri是表示精度的控制，表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        # True表示封闭的，
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 判断边界是否为我们想要的边界，1、边界有4个点、2、轮廓的周长大于500
        if len(approx) == 4 and  peri>1000:
            screenCnt = approx
            #没有画出轮廓
            cv2.drawContours(image.copy(), [screenCnt], -1, (0, 255, 0), 2)
            # cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)  # 绘制选择的轮廓
            # 展示结果
            print("STEP 2: 获取轮廓")
            cv2.imshow("Outline", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 透视转换（矩阵变换），screenCnt.reshape(4, 2) * ratio将坐标点还原到原始输入点
            warped = Four_point_transfrom(orig, screenCnt.reshape(4, 2) * ratio)
            warped = resize(warped, height=500)      #将仿射变换的图像进行缩放

            """图像扫描算法有问题，不能简单的进行二值化，而是应该图像增强"""
            # 二值处理
            # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            # threshold = cv2.threshold(warped, 98, 255, cv2.THRESH_BINARY)[1]
            strong=Strong_pic(warped)
            cv2.imwrite('scan.jpg', strong)
            # 展示结果
            print("STEP 3: 变换")
            cv2.imshow("Original", cv2.resize(orig,(500,500)))
            cv2.imshow("Scanned", cv2.resize(strong, (500,500)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """将原图像BGR格式转换成RGB，因为Image里面的图片格式是RGB，不转换就会发生颜色错乱"""
            image  = image[...,::-1]
            strong = strong[..., ::-1]
            break
        else:
             """手动裁剪然后，直接进行图像处理"""
        print('未识别到边界，手动裁剪~~~~')
        strong = Strong_pic(image)
        image = image[..., ::-1]
        strong = strong[..., ::-1]
        break

    return  image, strong     #将图片返回并显示

"""如果识别不是特别准确则手动矫正所选区域"""
def Manual_cutting(input_image):
    return

"""用于图片处理,一般是将裁剪的图片进行图像处理操作"""
def img_process(input_img):
    img=cv2.imread(input_img)    #读入图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #图像灰度化处理
    rec,threshold=cv2.threshold(img,127,255,cv2.THRESH_BINARY)    #二值化处理
    cv_show('pic',threshold)
    return threshold

if __name__=='__main__':
    Strong_pic('id_card3.jpg')



