FROM library/archlinux:latest@sha256:5906892165fc79b4e282e36f24802528bcee2bd2896982ad771045341e15db5c
ARG GIT_COMMIT=main
LABEL git-commit=$GIT_COMMIT vendor=pocl distro=Arch version=1.0

RUN pacman -Sy
RUN pacman --noconfirm -S gcc patch hwloc cmake git pkg-config make ninja ocl-icd clang llvm llvm-libs clinfo opencl-headers python3

RUN cd /home ; git clone https://github.com/pocl/pocl.git ; cd /home/pocl ; git checkout $GIT_COMMIT

RUN cd /home/pocl ; git pull ; mkdir b ; cd b; cmake -DCMAKE_INSTALL_PREFIX=/usr -G Ninja  ..
RUN cd /home/pocl/b ; ninja

ENV OCL_ICD_VENDORS=/home/pocl/b/ocl-vendors
ENV POCL_BUILDING=1
CMD cd /home/pocl/b ; clinfo ;  ctest -j4 --output-on-failure -L internal
