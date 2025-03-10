/* pocl_tree_sitter_utils.h - Utils to use tree sitter to fill out a
 * pocl_argument_info struct.

   Copyright (c) 2024 Robin Bijl / Tampere University

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

#ifndef TREE_SITTER_TEST_PROGRAM_TREE_SITTER_POCL_TREE_SITTER_UTILS_H
#define TREE_SITTER_TEST_PROGRAM_TREE_SITTER_POCL_TREE_SITTER_UTILS_H

#include "CL/cl.h"
#include "pocl_cl.h"
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-opencl.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Map a string representation of an argument type qualifier to a
 * CL_KERNEL_ARG_TYPE_*.
 *
 * \note str should only contain an argument type qualifier and nothing else.
 * \param str [in]: string representation, can start with '__'.
 * \return A valid arg type or CL_KERNEL_ARG_TYPE_NONE if no matches.
 */
cl_kernel_arg_type_qualifier pocl_str_to_type_qualifier (const char *str);

/**
 * Map a string representation of an argument access qualifier to a
 * CL_KERNEL_ARG_ACCESS_*.
 *
 * \note str should only contain an argument access qualifier and nothing
 * else. \param str [in]: string representation, can start with '__'. \return
 * A valid arg access or CL_KERNEL_ARG_ACCESS_NONE if no matches.
 */
cl_kernel_arg_access_qualifier
pocl_str_to_access_qualifier (const char *str);

/**
 * Map a string representation of an argument address qualifier to a
 * CL_KERNEL_ARG_ADDRESS_*.
 *
 * \note str should only contain an argument address qualifier and nothing
 * else. White spaces are oke. \param str [in]: string representation, can
 * start with '__'. \return A valid address type or
 * CL_KERNEL_ARG_ADDRESS_PRIVATE if no matches.
 */
cl_kernel_arg_address_qualifier
pocl_str_to_address_qualifier (const char *str);

/**
 * Creates a copy of the string that the tree sitter node was created of.
 *
 * \param source [in]: Original string parsed by tree sitter.
 * \param node [in]: Tree sitter node to copy contents from.
 * \return Copied null terminated string.
 */
char *pocl_ts_copy_node_contents (const char *source, TSNode node);

#define POCL_PRINT_TS_NODE(__source, __node)                                  \
do                                                                          \
  {                                                                         \
    char *string = pocl_ts_copy_node_contents (__source, __node);           \
    POCL_MSG_WARN ("node contents: %s\n", string);                          \
    free (string);                                                          \
  }                                                                         \
while (0)

/**
 * Searches through the tree for the node with the given kernel name and
 * returns the node with kernel arguments.
 *
 * \param source [in]: kernel string.
 * \param root_node [in]: The node (e.g. the root node) to search through.
 * \param kernel_name [in]: String of the kernel name to find.
 * \param status [out]: Return CL_SUCCESS if found, otherwise CL_FAILED.
 * \return The TSNode containing the kernel argument tree.
 */
TSNode pocl_ts_find_kernel_params (const char *source,
                                   TSNode root_node,
                                   const char *kernel_name,
                                   int32_t *status);

/**
 * Searches through the tree with the given query. The query should have two
 * captures, the first to match the key, the second to return.
 *
 * \param source [in]: String the tree was created from.
 * \param root_node [in]: The node (e.g. the root node) to search through.
 * \param query [in]: Used to find specific nodes in the tree. See tree
 * sitter documentation for info on how to write queries. \param key [in]:
 * String to match the first capture to. \param status [out]: Return
 * CL_SUCCESS if found, otherwise CL_FAILED. \return The TSNode containing
 * the kernel argument tree.
 */
TSNode pocl_ts_find_in_source (const char *source,
                               TSNode root_node,
                               const char *query,
                               const char *key,
                               int32_t *status);

int pocl_ts_map_node_to_arg_info (const char *source,
                                  TSNode node,
                                  pocl_argument_info *arg_info);

#ifdef __cplusplus
}
#endif

#endif // TREE_SITTER_TEST_PROGRAM_TREE_SITTER_POCL_TREE_SITTER_UTILS_H
