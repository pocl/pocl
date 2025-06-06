FROM amd64/ubuntu:24.04@sha256:042ed8226ebeefec04028d26d1ab54e121782d811db96c8030762684e47bb626

ARG GIT_COMMIT=main
ARG GH_PR
ARG GH_SLUG=pocl/pocl
ARG LLVM_VERSION=17

LABEL git-commit=$GIT_COMMIT vendor=pocl distro=Ubuntu version=1.0

ENV TERM=dumb
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y

RUN apt install -y tzdata
RUN apt install -y build-essential ocl-icd-libopencl1 cmake git pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} libclang-cpp${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-dev make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev dialog apt-utils

RUN cd /home ; git clone https://github.com/$GH_SLUG.git ; cd /home/pocl ; git checkout $GIT_COMMIT
RUN cd /home/pocl ; test -z "$GH_PR" || (git fetch origin +refs/pull/$GH_PR/merge && git checkout -qf FETCH_HEAD) && :
RUN cd /home/pocl ; mkdir b ; cd b; \
   cmake -DCMAKE_INSTALL_PREFIX=/usr \
         -DKERNELLIB_HOST_CPU_VARIANTS=distro \
         -DPOCL_ICD_ABSOLUTE_PATH=OFF \
         -G Ninja ..
RUN cd /home/pocl/b ; ninja install
# removing this picks up PoCL from the system install, not the build dir
RUN cd /home/pocl/b ; rm -f CTestCustom.cmake
CMD cd /home/pocl/b ; ctest -j4 --output-on-failure -L internal
