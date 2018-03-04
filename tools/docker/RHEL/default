FROM centos:latest
ARG GIT_COMMIT=master
LABEL git-commit=$GIT_COMMIT vendor=pocl distro=Centos7 version=1.0

RUN yum upgrade -y

RUN rpm --rebuilddb

RUN yum install -y epel-release

RUN yum install -y gcc gcc-c++ hwloc-devel hwloc-libs cmake git-core pkgconfig make ninja-build ocl-icd ocl-icd-devel patch redhat-rpm-config findutils libtool-ltdl libtool-ltdl-devel

RUN rpm --rebuilddb

RUN yum install -y wget

RUN cd /etc/yum.repos.d/ ; wget https://copr.fedorainfracloud.org/coprs/alonid/llvm-5.0.0/repo/epel-7/alonid-llvm-5.0.0-epel-7.repo

RUN yum install -y llvm-5.0.0 llvm-5.0.0-devel llvm-5.0.0-libs clang-5.0.0 clang-5.0.0-devel clang-5.0.0-libs clang-5.0.0-tools-extra

RUN cd /home ; git clone https://github.com/pocl/pocl.git ; cd /home/pocl ; git checkout $GIT_COMMIT

RUN cd /home/pocl ; mkdir b ; cd b; cmake -G Ninja -DWITH_LLVM_CONFIG=/opt/llvm-5.0.0/bin/llvm-config ..

RUN cd /home/pocl/b ; ninja-build

CMD cd /home/pocl/b ; ctest -j4 --output-on-failure -L internal
