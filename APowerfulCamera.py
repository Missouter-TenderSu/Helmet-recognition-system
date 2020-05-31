#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import time

from cv2.cv2 import (COLOR_BGR2GRAY, COLOR_BGR2RGB, VideoCapture, VideoWriter,
                     VideoWriter_fourcc, cvtColor, imread, imwrite)
from PyQt5.QtCore import QCoreApplication, QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QPushButton
from qimage2ndarray import array2qimage

from Ui_APowerfulCamera import Ui_Camera
#opencv_part

from PIL import Image
import mmcv
import numpy
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result

import numpy as np
config_file = mmcv.Config.fromfile('configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py')
checkpoint_file = 'work_dirs/epoch_4.pth'
model = init_detector(config_file, checkpoint_file)
CLASSES = ('person', 'hat')


class CamShow(QMainWindow,Ui_Camera):
    def __init__(self,parent=None): #ui部分
        super(CamShow,self).__init__(parent)
        self.setupUi(self)
        self.Mix()  #条和Box的值匹配
        self.Btn()  #按钮初始化
        self.Var()   #变量初始化
        self.Func() #控件功能集合
        self.Timer = QTimer()   #计时器
        self.Timer.timeout.connect(self.TimerOutFun)    #计时器

    def Mix(self):  #条和Box的值匹配    #ui部分
        self.RSld.valueChanged.connect(self.RBx.setValue)
        self.RBx.valueChanged.connect(self.RSld.setValue)
        self.GSld.valueChanged.connect(self.GBx.setValue)
        self.GBx.valueChanged.connect(self.GSld.setValue)
        self.BSld.valueChanged.connect(self.BBx.setValue)
        self.BBx.valueChanged.connect(self.BSld.setValue)

        self.BGSld.valueChanged.connect(self.BGBx.setValue)
        self.BGBx.valueChanged.connect(self.BGSld.setValue)
        self.ZYSld.valueChanged.connect(self.ZYBx.setValue)
        self.ZYBx.valueChanged.connect(self.ZYSld.setValue)
        self.LDSld.valueChanged.connect(self.LDBx.setValue)
        self.LDBx.valueChanged.connect(self.LDSld.setValue)
        self.DBSld.valueChanged.connect(self.DBBx.setValue)
        self.DBBx.valueChanged.connect(self.DBSld.setValue)

    def Btn(self):  #按钮初始化 #ui部分
        self.PrepCamera()
        self.Stop.setEnabled(False)
        self.Record.setEnabled(False)
        self.Die.setEnabled(False)
        self.RSld.setEnabled(False)
        self.RBx.setEnabled(False)
        self.GSld.setEnabled(False)
        self.GBx.setEnabled(False)
        self.BSld.setEnabled(False)
        self.BBx.setEnabled(False)
        self.BGSld.setEnabled(False)
        self.BGBx.setEnabled(False)
        self.ZYSld.setEnabled(False)
        self.ZYBx.setEnabled(False)
        self.LDSld.setEnabled(False)
        self.LDBx.setEnabled(False)
        self.DBSld.setEnabled(False)
        self.DBBx.setEnabled(False)
        self.quick.setEnabled(False)
        self.Local.setEnabled(False)
        self.Pic.setEnabled(False)
    
    def PrepCamera(self):   #准备一下照相机   #功能部分
        for ID in range(5000):
            self.camera=VideoCapture(ID)
            success, frame = self.camera.read()
            if success :
                break
        if not success:
            msg = QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
            buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
        return success
            

    def PrepVideo(self):    #准备一下检测视频
        self.camera=VideoCapture(self.RecordPath)
        success, frame = self.camera.read()
        if success :
            pass
        if not success:
            msg = QMessageBox.warning(self, u"Warning", u"文件不可读取,请重新选择路径",
            buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
        return success

    def PrepPicture(self):
        success=imread(self.RecordPath)
        if type(success)==type(None):
            success=False
        else:
            success=True
        if success :
            pass
        if not success:
            msg = QMessageBox.warning(self, u"Warning", u"图片不可读取,请重新选择路径",
            buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
        return success

    def Var(self):   #参数初始化 #ui部分
        self.RecordFlag=0
        self.RecordPath='D:/'
        self.LJTx.setText(self.RecordPath)
        self.Image_num=0
        self.R=1
        self.G=1
        self.B=1

        self.BGSld.setValue(self.camera.get(15))
        self.SetBG()
        self.ZYSld.setValue(self.camera.get(14))
        self.SetZY()
        self.LDSld.setValue(self.camera.get(10))
        self.SetLD()
        self.DBSld.setValue(self.camera.get(11))
        self.SetDB()
        self.Info.clear()

        self.temp=0
        self.count=0

    def Func(self):    #控件功能合集 #ui部分
        self.LJBut.clicked.connect(self.SetFilePath)
        self.Start.clicked.connect(self.StartCamera)
        self.Stop.clicked.connect(self.StopCamera)
        self.Record.clicked.connect(self.RecordCamera)
        self.Exit.clicked.connect(self.close)
        self.Die.stateChanged.connect(self.setDie)
        self.BGSld.valueChanged.connect(self.SetBG)
        self.ZYSld.valueChanged.connect(self.SetZY)
        self.LDSld.valueChanged.connect(self.SetLD)
        self.DBSld.valueChanged.connect(self.SetDB)
        self.RSld.valueChanged.connect(self.SetR)
        self.GSld.valueChanged.connect(self.SetG)
        self.BSld.valueChanged.connect(self.SetB)
        self.Local.stateChanged.connect(self.LocalFun)
        self.Pic.stateChanged.connect(self.PicFun)

    def StartCamera(self):  #界面按钮准备工作   #ui部分
        self.Local.setEnabled(True)
        self.Start.setEnabled(False)
        self.Stop.setEnabled(True)
        self.Record.setEnabled(True)
        self.Die.setEnabled(True)
        if self.Die.isChecked()==0:
            self.RSld.setEnabled(True)
            self.RBx.setEnabled(True)
            self.GSld.setEnabled(True)
            self.GBx.setEnabled(True)
            self.BSld.setEnabled(True)
            self.BBx.setEnabled(True)
        self.BGSld.setEnabled(True)
        self.BGBx.setEnabled(True)
        self.ZYSld.setEnabled(True)
        self.ZYBx.setEnabled(True)
        self.LDSld.setEnabled(True)
        self.LDBx.setEnabled(True)
        self.DBSld.setEnabled(True)
        self.DBBx.setEnabled(True)
        self.quick.setEnabled(True)
        self.Pic.setEnabled(True)
        self.Record.setText('录像')

        self.Timer.start(1)
        self.timelb=time.time()

    def StopCamera(self):   #暂停界面   #ui部分
        if self.Stop.text()=='暂停':
            self.Stop.setText('继续')
            self.Record.setText('保存')
            self.Timer.stop()
            self.camera.release()
        elif self.Stop.text()=='继续':
            if self.Local.isChecked():
                success=self.PrepVideo()
            elif self.Pic.isChecked():
                success=self.PrepPicture()
            else:
                success=self.PrepCamera()
            if not success:
                pass
            else:
                self.Stop.setText('暂停')
                self.Record.setText('录像')
                self.Timer.start(1)

    def TimerOutFun(self):  #读取图像    #功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分
        '''单帧图片功能'''
        time_interval = self.quick.value()
        zero=time.time()
        sec=1 #
        if self.Pic.isChecked():
            img=imread(self.RecordPath)
            success=True
        else:
            success,img=self.camera.read()  #读取图像！！！img为图像，如果要拿去加框框，可以在这里截胡
        if success: #如果成功读取
            self.Image = self.ColorAdjust(img)  #调整图像参数（RGB或者遗像）
            
            if self.Image_num % time_interval == 0:
                result = inference_detector(model, self.Image)
                self.temp=result
                self.count=show_result(self.Image, result,CLASSES,show=False,out_file="~/0.jpg")
                self.Image = imread("~/0.jpg")
                #img1_array.append(img)
                #imwrite("%d.jpg" % count, image)
                # print(count)
            elif self.Image_num!=1:
                result=self.temp
                self.count=show_result(self.Image, result,CLASSES,show=False,out_file="~/0.jpg")
                self.Image = imread("~/0.jpg")

            if self.count!=0 and self.Image_num % sec==0:
            #if 1==1 :
                Text1='有%d名员工未佩戴安全帽'%self.count
                self.Info_2.setHtml(QCoreApplication.translate("Camera", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\'; font-size:14pt; font-weight:600; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\"./warning.jpg\" /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:72; color:#ff0000;\">%s</span></p></body></html>")%(Text1))
            else:
                Text1='正常'
                self.Info_2.setHtml(QCoreApplication.translate("Camera", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\'; font-size:14pt; font-weight:600; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt;\">%s</span></p></body></html>")%Text1)
            
            self.DispImg()  #显示
            finish=time.time()
            self.Image_num+=1 #计算帧数
            if self.RecordFlag: #开始录像
                self.video_writer.write(self.Image)    #录像存入
            if self.Image_num%10==9:    #计算帧率
                frame_rate=10/(time.time()-self.timelb)
                Text1='延迟：'+str(round(1000*(finish-zero)/time_interval,1))+'  MS'
                Text2='帧率：'+str(round(frame_rate,1))+' FPS'
                self.Info.setHtml(QCoreApplication.translate("Camera", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">%s</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">%s</span></p></body></html>")%(Text1,Text2))
                self.timelb=time.time()
        elif self.Local.isChecked:#无法读取图像
            msg = QMessageBox.warning(self, u"Warning", u"视频已播放并检测完毕",
            buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
            self.StopCamera()
        else:
            msg = QMessageBox.warning(self, u"Warning", u"请检测相机是否被其他应用占用",
            buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
            self.StopCamera()
    def DispImg(self):#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分#功能部分

        #颜色调整
        img = cvtColor(self.Image, COLOR_BGR2RGB)
        #显示图像
        a=QPixmap(array2qimage(img)) 
        self.cam.setPixmap(a)
        self.cam.show()

    def setDie(self):   #ui部分     灰度模式按钮
        if self.Die.isChecked():
            self.RSld.setEnabled(False)
            self.RBx.setEnabled(False)
            self.GSld.setEnabled(False)
            self.GBx.setEnabled(False)
            self.BSld.setEnabled(False)
            self.BBx.setEnabled(False)
        else:
            self.RSld.setEnabled(True)
            self.RBx.setEnabled(True)
            self.GSld.setEnabled(True)
            self.GBx.setEnabled(True)
            self.BSld.setEnabled(True)
            self.BBx.setEnabled(True)

    def SetR(self): #ui部分 R
        R=self.RSld.value()
        self.R=R/255

    def SetG(self):#ui部分  G
        G=self.GSld.value()
        self.G=G/255

    def SetB(self):#ui部分 B
        B=self.BSld.value()
        self.B=B/255

    def ColorAdjust(self,img):  #调整颜色
        try:
            B=img[:,:,0]
            G=img[:,:,1]
            R=img[:,:,2]
            B=B*self.B
            G=G*self.G
            R=R*self.R
            img1=img
            img1[:,:,0]=B
            img1[:,:,1]=G
            img1[:,:,2]=R
            if self.Die.isChecked():
                img1 = cvtColor(img1, COLOR_BGR2GRAY)
            return img1
        except Exception as e:
            self.label.setText(str(e))

    def SetBG(self):    #调整曝光
        try:
            exposure_time_toset=self.BGSld.value()
            self.camera.set(22,exposure_time_toset)
            self.camera.set(10,exposure_time_toset)
        except Exception as e:
            self.label.setText(e)

    def SetZY(self):    #调整增益
        gain_toset=self.ZYSld.value()
        try:
            self.camera.set(22,gain_toset)
        except Exception as e:
            self.label.setText(e)

    def SetLD(self):    #调整亮度
        brightness_toset=self.LDSld.value()
        try:
            self.camera.set(10,brightness_toset)
        except Exception as e:
            self.label.setText(e)

    def SetDB(self):    #调整对比度
        contrast_toset=self.DBSld.value()
        try:
            self.camera.set(11,contrast_toset)
        except Exception as e:
            self.label.setText(e)

    def SetFilePath(self):  #选择保存路径
        if self.Local.isChecked() or self.Pic.isChecked():
            dirname = QFileDialog.getOpenFileName(self, "浏览", '.')[0]
        else:
            dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.LJTx.setText(dirname)
        if self.Local.isChecked() or self.Pic.isChecked():
            self.RecordPath=dirname
        else:
            self.RecordPath=dirname+'/'
            
    
    def LocalFun(self):
        if self.Local.isChecked():
            self.Pic.setEnabled(False)
            self.BGSld.setEnabled(False)
            self.BGBx.setEnabled(False)
            self.ZYSld.setEnabled(False)
            self.ZYBx.setEnabled(False)
            self.LDSld.setEnabled(False)
            self.LDBx.setEnabled(False)
            self.DBSld.setEnabled(False)
            self.DBBx.setEnabled(False)
            self.LJ.setText('检测路径')
            self.StopCamera()
            msg = QMessageBox.information(self, u"Tip", u"请选择文件路径,单击继续以开始检测",
            buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
        else:
            self.LJ.setText('保存路径')
            self.Pic.setEnabled(True)
            self.BGSld.setEnabled(True)
            self.BGBx.setEnabled(True)
            self.ZYSld.setEnabled(True)
            self.ZYBx.setEnabled(True)
            self.LDSld.setEnabled(True)
            self.LDBx.setEnabled(True)
            self.DBSld.setEnabled(True)
            self.DBBx.setEnabled(True)
            self.StopCamera()
    
    def PicFun(self):
        if self.Pic.isChecked():
            self.Local.setEnabled(False)
            self.BGSld.setEnabled(False)
            self.BGBx.setEnabled(False)
            self.ZYSld.setEnabled(False)
            self.ZYBx.setEnabled(False)
            self.LDSld.setEnabled(False)
            self.LDBx.setEnabled(False)
            self.DBSld.setEnabled(False)
            self.DBBx.setEnabled(False)
            self.LJ.setText('检测路径')
            self.StopCamera()
            msg = QMessageBox.information(self, u"Tip", u"请选择图片路径,单击继续以开始检测",
            buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
        else:
            self.LJ.setText('保存路径')
            self.Local.setEnabled(True)
            self.BGSld.setEnabled(True)
            self.BGBx.setEnabled(True)
            self.ZYSld.setEnabled(True)
            self.ZYBx.setEnabled(True)
            self.LDSld.setEnabled(True)
            self.LDBx.setEnabled(True)
            self.DBSld.setEnabled(True)
            self.DBBx.setEnabled(True)
            self.StopCamera()
    
    def RecordCamera(self):     #录像功能
        tag=self.Record.text()
        if tag=='保存':
            try:
                image_name=self.RecordPath+'image'+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.jpg'
                self.label.setText(image_name)
                imwrite(image_name, self.Image)
            except Exception as e:
                self.label.setText(e)
        elif tag=='录像':
            self.Record.setText('停止')
            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.avi'
            fps = 24/self.quick.value()
            size = (self.Image.shape[1],self.Image.shape[0])
            fourcc = VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_writer = VideoWriter(video_name, fourcc,self.camera.get(5), size)
            self.RecordFlag=1
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setText('录像中...')
            self.Stop.setEnabled(False)
            self.Exit.setEnabled(False)
        elif tag == '停止':
            self.Record.setText('录像')
            self.video_writer.release()
            self.RecordFlag = 0
            self.label.setText('录像已保存')
            self.Stop.setEnabled(True)
            self.Exit.setEnabled(True)
    def closeEvent(self, event):    #关闭事件
        ok = QPushButton()
        cacel = QPushButton()

        msg = QMessageBox(QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok,QMessageBox.ActionRole)
        msg.addButton(cacel, QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.camera.isOpened():
                self.camera.release()
            if self.Timer.isActive():
                self.Timer.stop()
            event.accept()
            
if __name__ == '__main__':      #毫无意义的主函数
    app = QApplication(sys.argv)
    ui=CamShow()
    ui.show()
    sys.exit(app.exec_())


'''
from PyQt5.QtGui import QBrush, QColor, QCursor, QFont, QPalette
from PyQt5.QtWidgets import QAbstractSpinBox, QCheckBox, QFrame, QLabel, QLineEdit, QPushButton, QSizePolicy, QSlider, QSpinBox, QStatusBar, QTabWidget, QTextEdit, QWidget
from PyQt5.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt

class Ui_Camera(object):
    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.m_flag=True
            self.m_Position=event.globalPos()-self.pos() #获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  #更改鼠标图标
            
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:  
            self.move(QMouseEvent.globalPos()-self.m_Position)#更改窗口位置
            QMouseEvent.accept()
            
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag=False
        self.setCursor(QCursor(Qt.ArrowCursor))
    def setupUi(self, Camera):
        self.setWindowFlags(Qt.FramelessWindowHint)  # 去掉标题栏的代码
        '''
