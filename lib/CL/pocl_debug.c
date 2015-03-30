#include "pocl_debug.h"

#ifdef POCL_DEBUG_MESSAGES
int pocl_debug_messages;

#ifdef HAVE_CLOCK_GETTIME

  #if !defined(_MSC_VER) && !defined(__MINGW32__)

    #include <time.h>
    #include <stdio.h>

    void pocl_debug_print_header(const char* func, unsigned line) {
        struct tm t;
        long tm_nanosec;
        struct timespec timespec;

        clock_gettime(CLOCK_REALTIME, &timespec);
        tm_nanosec = timespec.tv_nsec;
        gmtime_r(&timespec.tv_sec, &t);
        fprintf(stderr,
            "[%04i-%02i-%02i %02i:%02i:%02i.%09li] POCL: "
            "in fn %s at line %u:\n", (t.tm_year + 1900),
            t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min,
            t.tm_sec, tm_nanosec,  func, line);
    }

    void pocl_debug_init_time() {
    }

  #else

    #include <windows.h>
    #include <stdio.h>

    void pocl_debug_print_header(const char* func, unsigned line) {
        SYSTEMTIME st;
        FILETIME t;
        unsigned long st_nanosec;
        GetSystemTimeAsFileTime(&t);
        FileTimeToSystemTime(&t, &st);
        st_nanosec = (t.dwLowDateTime % 10000000) * 100;

        fprintf(stderr,
            "[%04u-%02u-%02u %02u:%02u:%02u.%09lu] POCL: "
            "in fn %s at line %u:\n",
            (unsigned int)st.wYear, (unsigned int)st.wMonth,
            (unsigned int)st.wDay, (unsigned int)st.wHour,
            (unsigned int)st.wMinute, (unsigned int)st.wSecond,
            (unsigned long)st_nanosec, func, line);
    }

    void pocl_debug_init_time() {
    }

  #endif


#endif

#endif
