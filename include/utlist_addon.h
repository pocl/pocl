/*
Copyright (c) 2019-2024 PoCL Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/

#ifndef UTLISTADDON_H
#define UTLISTADDON_H

#define LL_COMPUTE_LENGTH(head, el, dst)                                      \
  do                                                                          \
    {                                                                         \
      dst = 0;                                                                \
      LL_FOREACH (head, el)                                                   \
      {                                                                       \
        ++dst;                                                                \
      }                                                                       \
    }                                                                         \
  while (0)

#define LL_CONCAT_ATOMIC(head1, head2)                                               \
  do                                                                          \
    {                                                                         \
      LDECLTYPE (head1) _tmp;                                                 \
      if (head1)                                                              \
        {                                                                     \
          _tmp = (head1);                                                     \
          while (!__sync_bool_compare_and_swap (&(_tmp->next), NULL, head2))  \
            {                                                                 \
              _tmp = _tmp->next;                                              \
            }                                                                 \
        }                                                                     \
      else                                                                    \
        {                                                                     \
          (head1) = (head2);                                                  \
        }                                                                     \
    }                                                                         \
  while (0)

#define LL_APPEND_ATOMIC(head, add)                                                  \
  do                                                                          \
    {                                                                         \
      LDECLTYPE (head) _tmp;                                                  \
      (add)->next = NULL;                                                     \
      if (head)                                                               \
        {                                                                     \
          _tmp = (head);                                                      \
          while (!__sync_bool_compare_and_swap (&(_tmp->next), NULL, add))    \
            {                                                                 \
              _tmp = _tmp->next;                                              \
            }                                                                 \
        }                                                                     \
      else                                                                    \
        {                                                                     \
          (head) = (add);                                                     \
        }                                                                     \
    }                                                                         \
  while (0)

#define LL_FOREACH_ATOMIC(head, el)                                                  \
  for (el = head; el; el = __atomic_load_n (&(el->next), __ATOMIC_SEQ_CST))

#endif