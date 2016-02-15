#include "pocl_debug.h"
#include "pocl_timing.h"

#ifdef POCL_DEBUG_MESSAGES
int pocl_debug_messages;
int stderr_is_a_tty;


  #if !defined(_MSC_VER) && !defined(__MINGW32__)

    #include <time.h>
    #include <stdio.h>

    void pocl_debug_print_header(const char* func, unsigned line) {

        int year, mon, day, hour, min, sec, nanosec;
        pocl_gettimereal(&year, &mon, &day, &hour, &min, &sec, &nanosec);

        const char* formatstring;
        if (stderr_is_a_tty)
          formatstring = POCL_COLOR_BLUE
              "[%04i-%02i-%02i %02i:%02i:%02i.%09li] "
              POCL_COLOR_RESET "POCL: in fn"
              POCL_COLOR_CYAN " %s "
              POCL_COLOR_RESET "at line %u:\n";
        else
          formatstring = "[%04i-%02i-%02i %02i:%02i:%02i.%09i] "
              "POCL: in fn %s at line %u:\n";
        fprintf(stderr,
            formatstring, year, mon, day, hour, min,
            sec, nanosec, func, line);
    }

    void pocl_debug_measure_start(uint64_t *start) {
      if (!pocl_debug_messages)
        return;
      *start = pocl_gettimemono_ns();
    }

    void pocl_debug_print_duration(const char* func, unsigned line,
                                   const char* msg, uint64_t nanosecs)
    {
      if (!pocl_debug_messages)
        return;
      const char* formatstring;
      if (stderr_is_a_tty)
        formatstring = "      >>>  " POCL_COLOR_MAGENTA "     %3" PRIu64
                       ".%03" PRIu64 " " POCL_COLOR_RESET " %s    %s\n";
      else
        formatstring = "      >>>       %3" PRIu64 ".%03"
                       PRIu64 "  %s    %s\n";

      uint64_t nsec = nanosecs % 1000000000;
      uint64_t sec = nanosecs / 1000000000;
      uint64_t a, b;

      if ((sec == 0) && (nsec < 1000))
        {
          b = nsec % 1000;
          if (stderr_is_a_tty)
            formatstring = "      >>>      " POCL_COLOR_MAGENTA
                    "     %3" PRIu64 " " POCL_COLOR_RESET " ns    %s\n";
          else
            formatstring = "      >>>           %3" PRIu64 "  ns    %s\n";
          POCL_MSG_PRINT2(func, line, formatstring, b, msg);
        }
      else if ((sec == 0) && (nsec < 1000000))
        {
          a = nsec / 1000;
          b = nsec % 1000;
          POCL_MSG_PRINT2(func, line, formatstring, a, b, "us", msg);
        }
      else if (sec == 0)
        {
          a = nsec / 1000000;
          b = (nsec % 1000000) / 1000;
          POCL_MSG_PRINT2(func, line, formatstring, a, b, "ms", msg);
        }
      else
          POCL_MSG_PRINT2(func, line, formatstring, sec, nsec, "s", msg);

    }

    void pocl_debug_measure_finish(uint64_t *start, uint64_t *finish,
                                   const char* msg,
                                   const char* func,
                                   unsigned line) {
      if (!pocl_debug_messages)
        return;
      *finish = pocl_gettimemono_ns();
      pocl_debug_print_duration(func, line, msg, (*finish - *start) );
    }

  #else

/* Doesn't work, haven't been able to get it working.
 * Needs someone with experience in Win programming. */

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


  #endif



#endif
