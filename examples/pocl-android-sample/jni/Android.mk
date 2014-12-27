LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := poclVecAdd
LOCAL_C_INCLUDES := $(LOCAL_PATH)/libopencl-stub/include/
LOCAL_SRC_FILES := vectorAdd.cpp
LOCAL_CFLAGS   = -fPIC -O2
LOCAL_STATIC_LIBRARIES := OpenCL
LOCAL_LDLIBS := -ldl -llog
include $(BUILD_SHARED_LIBRARY)

include $(LOCAL_PATH)/libopencl-stub/Android.mk
