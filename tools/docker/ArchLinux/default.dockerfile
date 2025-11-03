FROM library/archlinux:latest@sha256:c136b06a4f786b84c1cc0d2494fabdf9be8811d15051cd4404deb5c3dc0b2e57
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
