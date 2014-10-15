LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_LDLIBS := -ldl -llog
LOCAL_MODULE := poclVecAdd

LOCAL_C_INCLUDES +=                       \
    $(LOCAL_PATH)/libopencl-stub/include/

LOCAL_SRC_FILES :=              \
    vectorAdd.cpp               \
    libopencl-stub/libopencl.c

include $(BUILD_SHARED_LIBRARY)
