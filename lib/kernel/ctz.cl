/* OpenCL built-in library: ctz()

   Copyright (c) 2022 Michal Babej / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "templates.h"



#if __has_builtin(__builtin_ctz)

/* These implementations return 8*sizeof(TYPE) when the input is 0 */

/* __builtin_ctz() is undefined for 0 */

#define __builtin_ctz0hh(n)                    \
  ({ char __n=(n); __n==0 ? 8 : __builtin_ctzs(__n); })
#define __builtin_ctz0h(n)                                     \
  ({ short __n=(n); __n==0 ? 16 : __builtin_ctzs(__n); })
#define __builtin_ctz0(n)                            \
  ({ int __n=(n); __n==0 ? 32 : __builtin_ctz(__n); })
#define __builtin_ctz0l(n)                                     \
  ({ long __n=(n); __n==0 ? 64 : __builtin_ctzl(__n); })

#else  /* !__has_builtin(__builtin_ctz) */

/* These implementations return 8*sizeof(TYPE) when the input is 0 */

/* These explicit implementations are taken from
   <http://aggregate.org/MAGIC/>:

   @techreport{magicalgorithms,
   author={Henry Gordon Dietz},
   title={{The Aggregate Magic Algorithms}},
   institution={University of Kentucky},
   howpublished={Aggregate.Org online technical report},
   date={2013-03-25},
   URL={http://aggregate.org/MAGIC/}
   }
*/

#define __builtin_ctz0hh(n)                     \
  ({                                            \
    char __n=(n);                               \
    return popcount ((__n & -__n) - 1);         \
  })

#define __builtin_ctz0h(n)                      \
  ({                                            \
    short __n=(n);                              \
    return popcount ((__n & -__n) - 1);         \
  })

#define __builtin_ctz0(n)                       \
  ({                                            \
    int __n=(n);                                \
    return popcount ((__n & -__n) - 1);         \
  })

#define __builtin_ctz0l(n)                      \
  ({                                            \
    long __n=(n);                               \
    return popcount ((__n & -__n) - 1);         \
  })

#endif


#define __builtin_ctz0uhh(n) __builtin_ctz0hh(n)
#define __builtin_ctz0uh(n)  __builtin_ctz0h(n)
#define __builtin_ctz0u(n)   __builtin_ctz0(n)
#define __builtin_ctz0ul(n)  __builtin_ctz0l(n)





#define ctz0 ctz
DEFINE_BUILTIN_G_G(ctz0)
#undef ctz0
