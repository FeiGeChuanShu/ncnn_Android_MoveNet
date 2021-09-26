# ncnn_Android_MoveNet
Android MoveNet single human pose estimation by ncnn

this project is a ncnn Android demo for MoveNet, it depends on ncnn library and opencv.  
https://github.com/Tencent/ncnn  
https://github.com/nihui/opencv-mobile
## model support:  
1.movenet-singlepose-lightningv4(from [tfhub](https://tfhub.dev/google/movenet/singlepose/lightning/4))    
2.movenet-singlepose-thunderv4(from [tfhub](https://tfhub.dev/google/movenet/singlepose/thunder/4))  

## how to build and run
### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
* Open this project with Android Studio, build it and enjoy!
## result  
![](result.gif)  
## reference:  
https://github.com/nihui/ncnn-android-nanodet  
https://tfhub.dev/google/movenet/singlepose/lightning/4  
https://tfhub.dev/google/movenet/singlepose/thunder/4  
