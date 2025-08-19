FROM library/archlinux:latest@sha256:104d24b4464e89a16566d3e68ce0e2707aa15258c690ee9bef755930e8bc1c2d
ARG GIT_COMMIT=main
LABEL git-commit=$GIT_COMMIT vendor=pocl distro=Arch version=1.0

RUN pacman -Sy
RUN pacman --noconfirm -S gcc patch hwloc cmake git pkg-config make ninja ocl-icd clang llvm llvm-libs clinfo opencl-headers python3

RUN cd /home ; git clone https://github.com/pocl/pocl.git ; cd /home/pocl ; git checkout $GIT_COMMIT

RUN cd /home/pocl ; mkdir b ; cd b; \
   cmake -DCMAKE_INSTALL_PREFIX=/usr \
         -DKERNELLIB_HOST_CPU_VARIANTS=distro \
         -DPOCL_ICD_ABSOLUTE_PATH=OFF \
         -G Ninja ..
RUN cd /home/pocl/b ; ninja
RUN cd /home/pocl/b ; ninja install

CMD cd /home/pocl/b ; rm CTestCustom.cmake ; clinfo ;  ctest -j4 --output-on-failure -L internal
