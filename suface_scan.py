"""用TK编写的界面"""
import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import pic_scan
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np

#frame：框架控件；在屏幕上显示一个矩形区域，多用来作为容器
class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green": ("绿牌", "#55FF55"), "yello": ("黄牌", "#FFFF00"), "blue": ("蓝牌", "#6666FF")}

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("图片扫描")
        win.state("zoomed")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        # ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)

        from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20,  command=self.from_pic)  # from_pic是界面函数的入口，button是可以执行的函数
        from_vedio_ctl = ttk.Button(frame_right2, text="来自摄像头", width=20, command=self.from_vedio)

        self.image_ctl = ttk.Label(frame_left)          #显示原图
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)          # 显示扫描后的图
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='扫描结果：').grid(column=0, row=0, sticky=tk.W)     #标题的位置在0行0列的头部

        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_vedio_ctl.pack(anchor="se", pady="5")
        from_pic_ctl.pack(anchor="se", pady="5")
        # self.predictor = predict.CardPredictor()
        # self.predictor.train_svm()

    # 载入图片，转换为TK格式的图像
    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)  # 将数组转换成图像
        imgtk = ImageTk.PhotoImage(image=im)  # 保存PhotoImage对象
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def show_roi(self, r, roi, color):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.update_time = time.time()
            try:
                c = self.color_transform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except:
                self.color_ctl.configure(state='disabled')
        elif self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            self.color_ctl.configure(state='disabled')

    def from_vedio(self):
        if self.thread_run:
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                mBox.showwarning('警告', '摄像头打开失败！')
                self.camera = None
                return
        self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True
        # 读取图片文件
        
    # 从图片读取数据，需要修改
    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])

        if self.pic_path:
            # 执行数据读取和训练操作
            img_bgr ,scan_pic= pic_scan.pic_process(self.pic_path)    #得到处理后的图片，原图像的颜色发生改变？？？
            # img_bgr=cv2.resize(img_bgr,(int(img_bgr.shape[0]/8),int(img_bgr.shape[1]/8)))
            # scan_pic = cv2.resize(scan_pic,(int(scan_pic.shape[0]/8),int(scan_pic.shape[1]/8)))
            cv2.imshow('orig1',img_bgr)
            cv2.waitKey()
            cv2.destroyAllWindows()
            """显示扫描图片，位置怎么处理"""
            scan = Image.fromarray(scan_pic)  # 将掃描图片转换成数组
            self.imgtk_roi = ImageTk.PhotoImage(image=scan)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')     #配置图片

            """显示原图，位置怎么处理？？"""
            img_bgr = Image.fromarray(img_bgr)  # 将掃描图片转换成数组
            self.imgtk_roi2 = ImageTk.PhotoImage(image=img_bgr)
            self.image_ctl.configure(image=self.imgtk_roi2, state='enable')  # 配置图片
            # self.r_ctl.configure(text=str(r))          #配置文字显示
            self.update_time = time.time()


    # @staticmethod
    # def vedio_thread(self):
    #     self.thread_run = True
    #     predict_time = time.time()
    #     while self.thread_run:
    #         _, img_bgr = self.camera.read()
    #         self.imgtk = self.get_imgtk(img_bgr)
    #         self.image_ctl.configure(image=self.imgtk)
    #         if time.time() - predict_time > 2:
    #             r, roi, color = self.predictor.predict(img_bgr)
    #             self.show_roi(r, roi, color)
    #             predict_time = time.time()
    #     print("run end")


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()
    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()         #进入窗口循环

