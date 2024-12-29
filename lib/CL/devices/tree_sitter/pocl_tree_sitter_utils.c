/* pocl_tree_sitter_utils.c - Utils to use tree sitter to fill out a
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

#include "pocl_tree_sitter_utils.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

/* These constants are taken from the generated parser and could change in the
 * future.
 */

/* things like global, local, etc */
#define TS_TYPE_ADDRESS_SPACE_QUALIFIER 202
/* for images to indicate read/write */
#define TS_TYPE_ACCESS_QUALIFIER 203
/* Appears when a variable is const */
#define TS_TYPE_TYPE_QUALIFIER 242
/* Things like int, float, etc */
#define TS_TYPE_SCALAR_TYPE 245
/* Things like int4, float8, etc */
#define TS_TYPE_VECTOR_TYPE 110
/* Appears when variable is a pointer */
#define TS_TYPE_POINTER_DECLARATOR 227
/* variable name */
#define TS_TYPE_IDENTIFIER 1

char *
pocl_ts_copy_node_contents (const char *source, TSNode node)
{

  assert (source != NULL);
  uint32_t start = ts_node_start_byte (node);
  uint32_t end = ts_node_end_byte (node);
  assert (end >= start);
  uint32_t size = end - start;

  char *ret = malloc (size + 1);
  strncpy (ret, source + start, size);
  ret[size] = 0;
  return ret;
}

cl_kernel_arg_type_qualifier
pocl_str_to_type_qualifier (const char *str)
{

  assert (str != NULL);
  if (strstr (str, "const") != NULL)
    return CL_KERNEL_ARG_TYPE_CONST;

  if (strstr (str, "restrict") != NULL)
    return CL_KERNEL_ARG_TYPE_RESTRICT;

  if (strstr (str, "volatile") != NULL)
    return CL_KERNEL_ARG_TYPE_VOLATILE;

  if (strstr (str, "pipe") != NULL)
    return CL_KERNEL_ARG_TYPE_PIPE;

  return CL_KERNEL_ARG_TYPE_NONE;
}

cl_kernel_arg_access_qualifier
pocl_str_to_access_qualifier (const char *str)
{

  assert (str != NULL);
  cl_kernel_arg_access_qualifier ret = CL_KERNEL_ARG_ACCESS_NONE;
  int read_access = strstr (str, "read") != NULL;
  int write_access = strstr (str, "write") != NULL;
  if (read_access)
    ret = CL_KERNEL_ARG_ACCESS_READ_ONLY;
  if (write_access)
    ret = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
  if (read_access && write_access)
    ret = CL_KERNEL_ARG_ACCESS_READ_WRITE;
  return ret;
}

cl_kernel_arg_address_qualifier
pocl_str_to_address_qualifier (const char *str)
{

  assert (str != NULL);
  if (strstr (str, "global") != NULL)
    return CL_KERNEL_ARG_ADDRESS_GLOBAL;

  if (strstr (str, "local") != NULL)
    return CL_KERNEL_ARG_ADDRESS_LOCAL;

  if (strstr (str, "constant") != NULL)
    return CL_KERNEL_ARG_ADDRESS_CONSTANT;

  if (strstr (str, "private") != NULL)
    return CL_KERNEL_ARG_ADDRESS_PRIVATE;

  return CL_KERNEL_ARG_ADDRESS_PRIVATE;
}

/**
 * Map a tree sitter node of an argument pointer to a pocl_argument_info.
 *
 * \param source [in]: Original string parsed by tree sitter.
 * \param node [in]: Tree sitter node to parse.
 * \param arg_info [out]: argument info to set mapped values to.
 */
int
pocl_ts_map_node_to_pointer (const char *source,
                             TSNode node,
                             pocl_argument_info *arg_info)
{
  assert (arg_info->type_name != NULL);
  /* +1 for '*' character */
  size_t new_type_size = strlen (arg_info->type_name) + 1;
  char *type_name = realloc (arg_info->type_name, new_type_size + 1);
  if (type_name == NULL)
    return CL_FAILED;
  strncat (type_name, "*", new_type_size);
  arg_info->type_name = type_name;

  arg_info->type = POCL_ARG_TYPE_POINTER;
  arg_info->type_size = sizeof (cl_mem);
  uint32_t node_count = ts_node_named_child_count (node);
  TSNode child;
  TSFieldId id;
  for (uint32_t i = 0; i < node_count; i++)
    {
      child = ts_node_named_child (node, i);
      id = ts_node_symbol (child);
      switch (id)
        {
        case TS_TYPE_IDENTIFIER:
          {
            arg_info->name = pocl_ts_copy_node_contents (source, child);
            break;
          }
        case TS_TYPE_TYPE_QUALIFIER:
          {
            /* Not parsing pointer type qualifier. */
            break;
          }
        default:
          {
            POCL_MSG_ERR ("Could not map pointer, unknown node id: %u\n", id);
            return CL_FAILED;
          }
        }
    }
  return CL_SUCCESS;
}

/**
 * Maps STR to a variable TYPE, possibly a vector.
 *
 * For example when TYPE is int,
 * and STR int8, this returns sizeof(cl_int8).
 */
#define RET_STR_TO_TYPE(STR, TYPE)                                            \
  do                                                                          \
    {                                                                         \
      char *end_ptr;                                                          \
      char *found_ptr = strstr (STR, #TYPE);                                  \
      if (found_ptr != NULL)                                                  \
        {                                                                     \
                                                                              \
          unsigned long str_size = strlen (#TYPE);                            \
          if (strlen (found_ptr) > str_size)                                  \
            {                                                                 \
              /* reset errno, strtoul only sets it on failure */              \
              errno = 0;                                                      \
              unsigned long vec_size                                          \
                = strtoul (found_ptr + str_size, &end_ptr, 10);               \
              if (!(errno == 0 && *end_ptr == '\0'))                          \
                {                                                             \
                  POCL_MSG_ERR ("Could not parse vector type of: %s\n", STR); \
                  return 0;                                                   \
                }                                                             \
                                                                              \
              switch (vec_size)                                               \
                {                                                             \
                case 2:                                                       \
                  return sizeof (cl_##TYPE##2);                               \
                case 3:                                                       \
                  return sizeof (cl_##TYPE##3);                               \
                case 4:                                                       \
                  return sizeof (cl_##TYPE##4);                               \
                case 8:                                                       \
                  return sizeof (cl_##TYPE##8);                               \
                case 16:                                                      \
                  return sizeof (cl_##TYPE##16);                              \
                default:                                                      \
                  {                                                           \
                    POCL_MSG_ERR ("Unknown vector size: %lu (%s)\n",          \
                                  vec_size, STR);                             \
                    return 0;                                                 \
                  }                                                           \
                }                                                             \
            }                                                                 \
          return sizeof (cl_##TYPE);                                          \
        }                                                                     \
    }                                                                         \
  while (0)

/**
 * Map a string representation of a OpenCL C type to the size of said type.
 *
 * \note str should only contain a type qualifier and nothing else.
 * White spaces are oke.
 * @param type_name [in]: String representation of OpenCL C type.
 * @return the size in bytes of the type or zero if it was not possible to
 * parse.
 */
unsigned int
pocl_map_type_name_to_size (const char *type_name)
{
  assert (type_name != NULL);

  RET_STR_TO_TYPE (type_name, int);
  RET_STR_TO_TYPE (type_name, float);
  RET_STR_TO_TYPE (type_name, char);
  RET_STR_TO_TYPE (type_name, long);
  RET_STR_TO_TYPE (type_name, double);
  RET_STR_TO_TYPE (type_name, half);
  RET_STR_TO_TYPE (type_name, short);

  char *found_ptr = strstr (type_name, "image");
  if (found_ptr != NULL)
    return sizeof (cl_mem);

  if (strncmp (type_name, "sampler", 7) == 0)
    return sizeof (cl_sampler);

  if (strncmp (type_name, "size_t", 6) == 0)
    return sizeof (size_t);

  /* TODO: Not sure if this is the right size */
  if (strncmp (type_name, "bool", 4) == 0)
    return sizeof (cl_bool);

  POCL_MSG_ERR ("Could not match %s to a type\n", type_name);
  return 0;
}

int
pocl_ts_map_node_to_arg_info (const char *source,
                              TSNode node,
                              pocl_argument_info *arg_info)
{

  arg_info->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
  arg_info->access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
  arg_info->type_qualifier = CL_KERNEL_ARG_TYPE_NONE;

  uint32_t node_count = ts_node_named_child_count (node);
  TSFieldId id = 0;
  char *node_text = NULL;

  for (uint32_t i = 0; i < node_count; i++)
    {
      TSNode child = ts_node_named_child (node, i);
      node_text = pocl_ts_copy_node_contents (source, child);
      id = ts_node_symbol (child);

      switch (id)
        {
        case TS_TYPE_ADDRESS_SPACE_QUALIFIER:
          {
            arg_info->address_qualifier
              = pocl_str_to_address_qualifier (node_text);
            break;
          }
        case TS_TYPE_ACCESS_QUALIFIER:
          {
            arg_info->type = POCL_ARG_TYPE_IMAGE;
            arg_info->type_size = sizeof (cl_mem);
            arg_info->access_qualifier
              = pocl_str_to_access_qualifier (node_text);
            break;
          }
        case TS_TYPE_TYPE_QUALIFIER:
          {
            arg_info->type_qualifier |= pocl_str_to_type_qualifier (node_text);
            break;
          }
        case TS_TYPE_VECTOR_TYPE:
        case TS_TYPE_SCALAR_TYPE:
          {
            arg_info->type_name = pocl_ts_copy_node_contents (source, child);
            if (strncmp (arg_info->type_name, "sampler_t", 9) == 0)
              {
                arg_info->type = POCL_ARG_TYPE_SAMPLER;
                arg_info->type_size = sizeof (cl_sampler);
              }
            else
              arg_info->type_size
                = pocl_map_type_name_to_size (arg_info->type_name);
            if (arg_info->type_size == 0)
              {
                POCL_MSG_ERR ("Could not determine type size.\n");
                goto ERROR;
              }
            break;
          }
        case TS_TYPE_POINTER_DECLARATOR:
          {
            int ret = pocl_ts_map_node_to_pointer (source, child, arg_info);
            if (ret != CL_SUCCESS)
              goto ERROR;
            break;
          }
        case TS_TYPE_IDENTIFIER:
          {
            arg_info->name = pocl_ts_copy_node_contents (source, child);
            break;
          }
        default:
          {
            POCL_MSG_ERR ("Unknown tree sitter type: %u\n", id);
            goto ERROR;
          }
        }
      free (node_text);
    }

  return CL_SUCCESS;

ERROR:
  free (node_text);
  return CL_KERNEL_ARG_INFO_NOT_AVAILABLE;
}

TSNode
pocl_ts_find_in_source (const char *source,
                        TSNode root_node,
                        const char *query,
                        const char *key,
                        int32_t *status)
{
  *status = CL_FAILED;
  size_t query_size = strlen (query);

  uint32_t error_offset = 0;
  TSQueryError error_type = 0;
  TSQuery *query_ts = ts_query_new (tree_sitter_opencl (), query, query_size,
                                    &error_offset, &error_type);

  TSNode ret = {};
  if (error_type != TSQueryErrorNone)
    {
      POCL_MSG_ERR ("failed to create query\n");
      POCL_MSG_ERR ("error from: \n%s \n", &(query[error_offset]));
      return ret;
    }
  if (ts_query_capture_count (query_ts) != 2)
    {
      POCL_MSG_ERR ("Tree-sitter query should have two captures.\n");
      return ret;
    }

  TSQueryCursor *cursor = ts_query_cursor_new ();
  ts_query_cursor_exec (cursor, query_ts, root_node);

  size_t key_len = strlen (key);
  TSQueryMatch match = {};
  while (ts_query_cursor_next_match (cursor, &match))
    {
      /* While the documentation describes predicates like #eq?, these are
       * usually implemented by the bindings and C API does not provide this,
       * Therefore, we write our own.
       */
      uint32_t source_offset = ts_node_start_byte (match.captures[0].node);
      if (strncmp (&source[source_offset], key, key_len) == 0)
        {
          ret = match.captures[1].node;
          *status = CL_SUCCESS;
          break;
        }
    }

  ts_query_delete (query_ts);
  ts_query_cursor_delete (cursor);
  return ret;
}

#define TS_KERNEL_QUERY                                                       \
  "(function_declarator \n"                                                   \
  "declarator: (identifier) @fn_name\n"                                       \
  "parameters: (parameter_list) @params \n"                                   \
  ")\n"

TSNode
pocl_ts_find_kernel_params (const char *source,
                            TSNode root_node,
                            const char *kernel_name,
                            int32_t *status)
{
  return pocl_ts_find_in_source (source, root_node, TS_KERNEL_QUERY,
                                 kernel_name, status);
}