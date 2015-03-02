#!/bin/bash
#
# Build script for Android
#
#   Copyright (c) 2014 Krishnaraj R Bhat (krrishnarraj@gmail.com)
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
# Usage: build-arm.sh [release]
# default - builds debug version for quick testing
# release - builds release version with flto options. Much Much slower

PWD=`pwd`
I_AM=`id -un`
MY_GROUP=`id -gn`
ANDROID_TOOLCHAIN=/tmp/android-toolchain/

echo "NDK standalone toolchain setup..."
if [ ! -e $ANDROID_NDK/build/tools/make-standalone-toolchain.sh ]; then
    echo "Install Android NDK and set environment variable ANDROID_NDK to its root"
    return
fi
$ANDROID_NDK/build/tools/make-standalone-toolchain.sh \
				--toolchain=arm-linux-androideabi-4.9 \
				--arch=arm \
				--platform=android-16 \
				--install-dir=$ANDROID_TOOLCHAIN

INSTALL_PREFIX=/data/data/org.pocl.libs/files/
# Create directories for PREFIX, target location in android
if [ ! -e $INSTALL_PREFIX ]; then
    sudo mkdir -p $INSTALL_PREFIX
    sudo mkdir -p $INSTALL_PREFIX/lib/pkgconfig/
    sudo chown -R $I_AM:$MY_GROUP $INSTALL_PREFIX
    sudo chmod 755 -R $INSTALL_PREFIX
fi

# Prebuilt llvm that runson(android) -> target(android)
LLVM_HOST_ANDROID_TARGET_ANDROID=$PWD/pocl-android-prebuilts/arm/llvm/android
if [ ! -e $LLVM_HOST_ANDROID_TARGET_ANDROID/lib/libclangFrontend.a  ]; then
    echo "Build and place llvm(android) at " $LLVM_HOST_ANDROID_TARGET_ANDROID
    return
fi

if [ ! -e $ANDROID_TOOLCHAIN/sysroot/usr/lib/libclangFrontend.a  ]; then
echo "Copying llvm libs(android) to sysroot..."
cp -rf $LLVM_HOST_ANDROID_TARGET_ANDROID/* $ANDROID_TOOLCHAIN/sysroot/usr/
fi

# Prebuilt llvm that runon(x64) -> target(android)
LLVM_HOST_x64_TARGET_ANDROID=$PWD/pocl-android-prebuilts/arm/llvm/cross_compiler_for_android
if [ ! -e $LLVM_HOST_x64_TARGET_ANDROID/bin/clang ]; then
    echo "Build and place llvm runson(x64) -> target(android) at " $LLVM_HOST_x64_TARGET_ANDROID
    return
fi

if [ ! -e $ANDROID_TOOLCHAIN/sysroot/usr/bin/clang ]; then
echo "copying llvm(host) to sysroot...."
cp -rf $LLVM_HOST_x64_TARGET_ANDROID/* $ANDROID_TOOLCHAIN/sysroot/usr/
fi

PREBUILT_NCURSES=$PWD/pocl-android-prebuilts/arm/ncurses
if [ ! -e $PREBUILT_NCURSES/lib/libncurses.a ]; then
    echo "Build and place ncurses for android at " $PREBUILT_NCURSES
    return
fi
echo "copying ncurses to sysroot...."
cp -rf $PREBUILT_NCURSES/* $ANDROID_TOOLCHAIN/sysroot/usr/
ln -sf $ANDROID_TOOLCHAIN/sysroot/usr/lib/libncurses.a $ANDROID_TOOLCHAIN/sysroot/usr/lib/libcurses.a


PREBUILT_LTDL=$PWD/pocl-android-prebuilts/arm/libtool
if [ ! -e $PREBUILT_LTDL/lib/libltdl.a ]; then
    echo "Build and place libltdl for android at " $PREBUILT_LTDL
    return
fi
echo "copying ltdl to sysroot...."
cp -rf $PREBUILT_LTDL/* $ANDROID_TOOLCHAIN/sysroot/usr/

PREBUILT_HWLOC=$PWD/pocl-android-prebuilts/arm/hwloc
if [ ! -e $PREBUILT_HWLOC/lib/libhwloc.a ]; then
    echo "Build and place libhwloc for android at " $PREBUILT_HWLOC
    return
fi
echo "copying hwloc to sysroot...."
cp -rf $PREBUILT_HWLOC/* $ANDROID_TOOLCHAIN/sysroot/usr/

PREBUILT_BINUTILS=$PWD/pocl-android-prebuilts/arm/binutils
if [ ! -e $PREBUILT_BINUTILS/bin/ld ]; then
    echo "Build and place binutils for android at " $PREBUILT_BINUTILS
    return
fi
echo "copying ld to "$INSTALL_PREFIX
cp -rf $PREBUILT_BINUTILS/* $INSTALL_PREFIX/

ln -sf $ANDROID_TOOLCHAIN/sysroot/usr/lib/libc.so $ANDROID_TOOLCHAIN/sysroot/usr/lib/libpthread.so
ln -sf $ANDROID_TOOLCHAIN/sysroot/usr/lib/libc.so $ANDROID_TOOLCHAIN/sysroot/usr/lib/librt.so
ln -sf $ANDROID_TOOLCHAIN/sysroot/usr/include/GLES $ANDROID_TOOLCHAIN/sysroot/usr/include/GL
rm $ANDROID_TOOLCHAIN/sysroot/usr/lib/libstdc++.*

export PATH=$ANDROID_TOOLCHAIN/bin:$ANDROID_TOOLCHAIN/sysroot/usr/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANDROID_TOOLCHAIN/sysroot/usr/lib/
export HOST=arm-linux-androideabi
export PREFIX=$INSTALL_PREFIX
export SYSROOT=$ANDROID_TOOLCHAIN/sysroot/usr/
export TARGET_CPU="cortex-a9"

# flto option in gcc 4.8 eats all memory & eventually /tmp. Better to place tmp file in disk
export TMPDIR=$HOME/tmp/junk/
if [ ! -e $TMPDIR ]; then
    mkdir -p $TMPDIR
fi


#if [ ! -e $PWD/../configure ]; then
#    cd ..; ./autogen.sh; cd -
#fi

DEBUG_BUILD=1
if [ $# -gt 0 ]  && [ $1 = "release" ] ; then
    DEBUG_BUILD=0
fi

if [ $DEBUG_BUILD == 1 ] ; then
#CC="arm-linux-androideabi-gcc  -static-libstdc++ " CXX="arm-linux-androideabi-g++  -static-libstdc++ "  ac_cv_c_bigendian=no LLC_HOST_CPU=$TARGET_CPU HWLOC_CFLAGS="-I"$ANDROID_TOOLCHAIN"/sysroot/usr/include" HWLOC_LIBS="-L"$ANDROID_TOOLCHAIN"/sysroot/usr/lib -lhwloc" CFLAGS=" -Os " CPPFLAGS=" -Os " LDFLAGS=" "  SYSROOTDIR=$ANDROID_TOOLCHAIN/sysroot/ ../configure --prefix=$PREFIX --host=$HOST --disable-icd --with-sysroot=$ANDROID_TOOLCHAIN/sysroot/ --enable-debug --verbose

cmake -DCMAKE_TOOLCHAIN_FILE=androideabi.cmake -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_AR:FILEPATH=$HOST-gcc-ar -DCMAKE_RANLIB:FILEPATH=$HOST-gcc-ranlib -DCMAKE_CXX_FLAGS:STRING="-Os -ffunction-sections -fdata-sections -fno-lto" -DCMAKE_C_FLAGS:STRING="-Os -ffunction-sections -fdata-sections -fno-lto" -DCMAKE_EXE_LINKER_FLAGS:STRING='-fno-lto -fuse-linker-plugin -Wl,--gc-sections' -DCMAKE_MODULE_LINKER_FLAGS:STRING='-fno-lto -fuse-linker-plugin -Wl,--gc-sections'  -DCMAKE_SHARED_LINKER_FLAGS:STRING='-fno-lto -fuse-linker-plugin -Wl,--gc-sections' -DCMAKE_INSTALL_PREFIX:PATH=$PREFIX -DLLC_HOST_CPU=$TARGET_CPU ..
make -j4

else
#ac_cv_c_bigendian=no LLC_HOST_CPU=$TARGET_CPU HWLOC_CFLAGS="-I"$ANDROID_TOOLCHAIN"/sysroot/usr/include" HWLOC_LIBS="-L"$ANDROID_TOOLCHAIN"/sysroot/usr/lib -lhwloc" CFLAGS=" -ffunction-sections -fdata-sections -Os -flto " CPPFLAGS=" -ffunction-sections -fdata-sections -Os -flto " LDFLAGS=" -Wl,--gc-sections -flto " SYSROOTDIR=$ANDROID_TOOLCHAIN/sysroot/ ../configure --prefix=$PREFIX --host=$HOST --disable-icd --with-sysroot=$ANDROID_TOOLCHAIN/sysroot/

cmake -DCMAKE_TOOLCHAIN_FILE=androideabi.cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_AR:FILEPATH=$HOST-gcc-ar -DCMAKE_RANLIB:FILEPATH=$HOST-gcc-ranlib -DCMAKE_CXX_FLAGS:STRING="-Os -ffunction-sections -fdata-sections -flto" -DCMAKE_C_FLAGS:STRING="-Os -ffunction-sections -fdata-sections -flto" -DCMAKE_EXE_LINKER_FLAGS:STRING='-flto -fuse-linker-plugin -Wl,--gc-sections' -DCMAKE_MODULE_LINKER_FLAGS:STRING='-flto -fuse-linker-plugin -Wl,--gc-sections'  -DCMAKE_SHARED_LINKER_FLAGS:STRING='-flto -fuse-linker-plugin -Wl,--gc-sections' -DCMAKE_INSTALL_PREFIX:PATH=$PREFIX -DLLC_HOST_CPU=$TARGET_CPU ..
make

fi

make install

# Copy license files to install folder
cp -f $ANDROID_TOOLCHAIN/sysroot/usr/share/LICENSE* $INSTALL_PREFIX/share/
cp -f ../LICENSE $INSTALL_PREFIX/share/LICENSE.pocl

echo -e "\n\nBuild completed...\nBuilt files are at "$PREFIX"\n"

