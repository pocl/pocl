#cmakedefine HAVE_LTTNG_UST

#cmakedefine HAVE_LINUX_VSOCK_H

#cmakedefine QUEUE_PROFILING

#cmakedefine FORKING

#cmakedefine ENABLE_RDMA
#cmakedefine RDMA_USE_SVM
#if !defined(ENABLE_RDMA) && defined(RDMA_USE_SVM)
#error RDMA_USE_SVM requires RDMA to be enabled
#endif
