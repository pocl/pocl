#include "pocl_debug.h"
#include "pocl_timing.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdarg.h>

#ifdef POCL_DEBUG_MESSAGES

uint64_t pocl_debug_messages_filter; /* Bitfield */
int stderr_is_a_tty;

static pthread_mutex_t console_mutex = PTHREAD_MUTEX_INITIALIZER;

  #if !defined(_MSC_VER) && !defined(__MINGW32__)

    /* The same as fprintf(stderr,..), except it's protected by mutex
     * helps to keep the debug output correct when >1 thread
     * is doing fprintf() */
    int
    pocl_fprintf_err (const char* format, ...)
    {
      pthread_mutex_lock (&console_mutex);

      va_list args;
      va_start (args, format);
      int res = vfprintf (stderr, format, args);
      va_end (args);

      pthread_mutex_unlock (&console_mutex);
      return res;
    }

    void
    pocl_debug_messages_setup (const char* debug)
    {
      pocl_debug_messages_filter = 0;
      if (strlen (debug) == 1)
        {
          if (debug[0] == '1')
            pocl_debug_messages_filter = POCL_DEBUG_FLAG_GENERAL
                                         | POCL_DEBUG_FLAG_WARNING
                                         | POCL_DEBUG_FLAG_ERROR;
          return;
        }
      /* else parse */
      char* tokenize = strdup (debug);
      char* ptr = NULL;
      ptr = strtok (tokenize, ",");

      while (ptr != NULL)
      {
        if (strncmp (ptr, "general", 7) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_GENERAL;
        else if (strncmp (ptr, "event", 5) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_EVENTS;
        else if (strncmp (ptr, "cache", 5) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_CACHE;
        else if (strncmp (ptr, "llvm", 4) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_LLVM;
        else if (strncmp (ptr, "refc", 4) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_REFCOUNTS;
        else if (strncmp (ptr, "lock", 4) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_LOCKING;
        else if (strncmp (ptr, "cuda", 4) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_CUDA;
        else if (strncmp (ptr, "warn", 4) == 0)
          pocl_debug_messages_filter |= (POCL_DEBUG_FLAG_WARNING | POCL_DEBUG_FLAG_ERROR);
        else if (strncmp (ptr, "hsa", 3) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_HSA;
        else if (strncmp (ptr, "tce", 3) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_TCE;
        else if (strncmp (ptr, "mem", 3) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_MEMORY;
        else if (strncmp (ptr, "tim", 3) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_TIMING;
        else if (strncmp (ptr, "all", 3) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_ALL;
        else if (strncmp (ptr, "err", 3) == 0)
          pocl_debug_messages_filter |= POCL_DEBUG_FLAG_ERROR;
        else
          POCL_MSG_WARN ("Unknown token in POCL_DEBUG env var: %s", ptr);

        ptr = strtok (NULL,",");
      }

      free (tokenize);
      pocl_fprintf_err ("** Final POCL_DEBUG flags: %" PRIX64 " \n",
                        pocl_debug_messages_filter);
    }

    void
    pocl_debug_print_header (const char* func, unsigned line,
                             const char *filter, int filter_type)
    {

        int year, mon, day, hour, min, sec, nanosec;
        pocl_gettimereal(&year, &mon, &day, &hour, &min, &sec, &nanosec);

        const char *filter_type_str;
        const char *formatstring;

        if (filter_type == POCL_FILTER_TYPE_ERR)
          filter_type_str = (stderr_is_a_tty ? POCL_COLOR_RED : " *** ERROR *** ");
        else if (filter_type == POCL_FILTER_TYPE_WARN)
          filter_type_str = (stderr_is_a_tty ? POCL_COLOR_YELLOW : " *** WARNING *** ");
        else if (filter_type == POCL_FILTER_TYPE_INFO)
          filter_type_str = (stderr_is_a_tty ? POCL_COLOR_GREEN : " *** INFO *** ");
        else
          filter_type_str = (stderr_is_a_tty ? POCL_COLOR_GREEN : " *** UNKNOWN *** ");

        if (stderr_is_a_tty)
          formatstring = POCL_COLOR_BLUE
              "[%04i-%02i-%02i %02i:%02i:%02i.%09li]"
              POCL_COLOR_RESET "POCL: in fn %s "
              POCL_COLOR_RESET "at line %u:\n %s | %9s | ";
        else
          formatstring = "[%04i-%02i-%02i %02i:%02i:%02i.%09i] "
              "POCL: in fn %s at line %u:\n %s | %9s | ";

        pocl_fprintf_err (formatstring, year, mon, day, hour, min, sec,
                          nanosec, func, line, filter_type_str, filter);
    }

    void pocl_debug_measure_start(uint64_t *start) {
      *start = pocl_gettimemono_ns();
    }

    #define PRINT_DURATION(func, line, ...)                                   \
      do                                                                      \
        {                                                                     \
          pocl_debug_print_header (func, line,                                \
                                   "TIMING", POCL_FILTER_TYPE_INFO);          \
          pocl_fprintf_err (__VA_ARGS__);                                     \
        }                                                                     \
      while (0)


    void pocl_debug_print_duration(const char* func, unsigned line,
                                   const char* msg, uint64_t nanosecs)
    {
      if (!(pocl_debug_messages_filter & POCL_DEBUG_FLAG_TIMING))
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
          PRINT_DURATION (func, line, formatstring, b, msg);
        }
      else if ((sec == 0) && (nsec < 1000000))
        {
          a = nsec / 1000;
          b = nsec % 1000;
          PRINT_DURATION (func, line, formatstring, a, b, "us", msg);
        }
      else if (sec == 0)
        {
          a = nsec / 1000000;
          b = (nsec % 1000000) / 1000;
          PRINT_DURATION (func, line, formatstring, a, b, "ms", msg);
        }
      else
          PRINT_DURATION (func, line, formatstring, sec, nsec, "s", msg);

    }



    void pocl_debug_measure_finish(uint64_t *start, uint64_t *finish,
                                   const char* msg,
                                   const char* func,
                                   unsigned line) {
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
