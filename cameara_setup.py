# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:22:36 2022

@author: Victus
"""

import cv2
import os
import numpy as np
import time
from pypylon import pylon
import datetime

tl_factory = pylon.TlFactory.GetInstance()
camera = pylon.InstantCamera() # tạo ra một instance để lấy và lưu giữ ảnh
camera.Attach(tl_factory.CreateFirstDevice()) # kiểm tra chỗ này

############################### Đọc video ################################3
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
print(camera.ExposureTime.GetValue())
# convert này dùng để chuyển định dạng sang opencv
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
print('Starting to acquire')

print("exposure time:", camera.ExposureTime.GetValue())
while camera.IsGrabbing():
    # timeout là thời gian lấy ảnh trở về > expose time( thời gian phơi gian)
    grab = camera.RetrieveResult(40000, pylon.TimeoutHandling_ThrowException)
    if grab.GrabSucceeded():
        start_time = time.time()
        #print("Grabing success")
        image = converter.Convert(grab)
        image = image.GetArray()
        resized = cv2.resize(image,(600,410)) 
        # không được đổi biến resized thành biến image vì biến image 
        #là chuỗi array đọc từ camera chính
        cv2.imshow('image', resized)
    else:
        break
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
cv2.destroyAllWindows()#
camera.Close()
grab.Release()
