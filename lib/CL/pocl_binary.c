/* OpenCL runtime library: pocl binary

   Copyright (c) 2016 pocl developers

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

#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "pocl_cl.h"
#include "pocl_binary.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"

#include <sys/stat.h>
#include <dirent.h>
#include <libgen.h>

#ifndef __APPLE__
  #include <endian.h>
#else
  #include <libkern/OSByteOrder.h>
  #define htole16(x) OSSwapHostToLittleInt16(x)
  #define le16toh(x) OSSwapLittleToHostInt16(x)
  #define htole32(x) OSSwapHostToLittleInt32(x)
  #define le32toh(x) OSSwapLittleToHostInt32(x)
  #define htole64(x) OSSwapHostToLittleInt64(x)
  #define le64toh(x) OSSwapLittleToHostInt64(x)
#endif

/* pocl binary identifier */
#define POCLCC_STRING_ID "poclbin"
#define POCLCC_STRING_ID_LENGTH 8
#define POCLCC_VERSION 1

/* pocl binary structures */

/* Note that structs are not 1:1 to what's serialized on-disk. In particular
 * 1) for integer values, endianness is forced to LITTLE_ENDIAN
 * 2) pointers in general are not written at all, rather reconstructed from data
 * 3) char* strings are written as: | uint32_t strlen | strlen bytes of content |
 * 4) files are written as two strings: | uint32_t | relative filename | uint32_t | content |
 */

typedef struct pocl_binary_kernel_s
{
  /* the first 3 fields are sizes in bytes of the data pieces that follow
   * (to allow quickly jumping between records in the serialized binary;
   * this is required to e.g. extract kernel metadata without having to
   * completely deserialize everything in the binary) */

  /* size of this entire struct serialized, including all data (binaries & arginfo sizes)
   * current offset in binary + struct_size = offset of the next pocl_binary_kernel */
  uint64_t struct_size;
  /* size of kernel cachedir content serialized */
  uint64_t binaries_size;
  /* size of "arginfo" array of structs serialized */
  uint32_t arginfo_size;

  /* size of the kernel_name string */
  uint32_t sizeof_kernel_name;
  /* kernel_name string */
  char *kernel_name;

  // number of kernel arguments
  uint32_t num_args;
  // number of kernel local variables
  uint32_t num_locals;

  /* arguments and argument metadata. Note that not everything is stored
   * in the serialized binary */
  struct pocl_argument *dyn_arguments;
  struct pocl_argument_info *arg_info;
} pocl_binary_kernel;

typedef struct pocl_binary_s
{
  /* file format "magic" marker */
  char pocl_id[POCLCC_STRING_ID_LENGTH];
  /* llvm triple + target hash */
  uint64_t device_id;
  /* binary format version */
  uint32_t version;
  /* number of kernels in the serialized pocl binary */
  uint32_t num_kernels;
  /* program->build_hash[device_i], required to restore files into pocl cache */
  SHA1_digest_t program_build_hash;
} pocl_binary;


#define TO_LE(x)                                \
  ((sizeof(x) == 8) ? htole64((uint64_t)x) :    \
  ((sizeof(x) == 4) ? htole32((uint32_t)x) :    \
  ((sizeof(x) == 2) ? htole16((uint16_t)x) :    \
  ((sizeof(x) == 1) ? (uint8_t)(x) : 0 ))))

#define FROM_LE(x)                              \
  ((sizeof(x) == 8) ? le64toh((uint64_t)x) :    \
  ((sizeof(x) == 4) ? le32toh((uint32_t)x) :    \
  ((sizeof(x) == 2) ? le16toh((uint16_t)x) :    \
  ((sizeof(x) == 1) ? (uint8_t)(x) : 0 ))))

/***********************************************************/

#define BUFFER_STORE(elem, type)                  \
  do                                              \
    {                                             \
      type b_s_tmp = (type) TO_LE ( (type) elem); \
      memcpy (buffer, &b_s_tmp, sizeof (type));   \
      buffer += sizeof (type);                    \
    }                                             \
  while(0)

#define BUFFER_READ(elem, type)                   \
  memcpy (&elem, buffer, sizeof (type));          \
  elem = (type) FROM_LE ( (type) elem);           \
  buffer += sizeof (type)

#define BUFFER_STORE_STR2(elem, len)              \
  do {                                            \
    BUFFER_STORE(len, uint32_t);                  \
    if (len)                                      \
      {                                           \
        memcpy(buffer, elem, len);                \
        buffer += len;                            \
      }                                           \
  } while (0)

#define BUFFER_READ_STR2(elem, len)               \
  do                                              \
    {                                             \
      BUFFER_READ(len, uint32_t);                 \
      if (len)                                    \
        {                                         \
          elem = malloc (len + 1);                \
          memcpy (elem, buffer, len);             \
          elem[len] = 0;                          \
          buffer += len;                          \
        }                                         \
    } while (0)

#define BUFFER_STORE_STR(elem)                    \
  do { uint32_t len = strlen(elem);               \
    BUFFER_STORE_STR2(elem, len); } while (0)

#define BUFFER_READ_STR(elem)                     \
  do { uint32_t len = 0;                          \
    BUFFER_READ_STR2(elem, len); } while (0)

#define ADD_STRLEN(else_b)                        \
    if (serialized)                               \
      {                                           \
        unsigned char* t = buffer+res;            \
        res += *(uint32_t*)t;                     \
        res += sizeof(uint32_t);                  \
      }                                           \
    else                                          \
      {                                           \
        res += sizeof(uint32_t);                  \
        res += else_b;                            \
      }

/***********************************************************/

static unsigned char*
read_header(pocl_binary *b, const unsigned char *buffer)
{
  memset(b, 0, sizeof(pocl_binary));
  memcpy(b->pocl_id, buffer, POCLCC_STRING_ID_LENGTH);
  buffer += POCLCC_STRING_ID_LENGTH;
  BUFFER_READ(b->device_id, uint64_t);
  BUFFER_READ(b->version, uint32_t);
  BUFFER_READ(b->num_kernels, uint32_t);
  memcpy(b->program_build_hash, buffer, sizeof(SHA1_digest_t));
  buffer += sizeof(SHA1_digest_t);
  return (unsigned char*)buffer;
}

#define FNV_OFFSET UINT64_C(0xcbf29ce484222325)
#define FNV_PRIME UINT64_C(0x100000001b3)
static uint64_t
pocl_binary_get_device_id(cl_device_id device)
{
  /* FNV-1A with whatever device returns
   * as its build hash string */
  uint64_t result = FNV_OFFSET;
  char *dev_hash = device->ops->build_hash(device);

  int i, length = strlen(dev_hash);
  for (i=0; i<length; i++)
    {
      result *= FNV_PRIME;
      result ^= dev_hash[i];
    }
  free(dev_hash);

  return result;
}

static unsigned char*
check_binary(cl_device_id device, const unsigned char *binary)
{
  pocl_binary b;
  unsigned char *p = read_header(&b, binary);
  if (b.version != POCLCC_VERSION)
    return NULL;
  if (strncmp(b.pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH))
    return NULL;
  if (pocl_binary_get_device_id(device) != b.device_id)
    return NULL;
  return p;
}

int
pocl_binary_check_binary(cl_device_id device, const unsigned char *binary)
{
  return (check_binary(device, binary) != NULL);
}

/*****************************************************************************/

void
pocl_binary_set_program_buildhash(cl_program program,
                                  unsigned device_i,
                                  const unsigned char *binary)
{
  pocl_binary b;
  read_header(&b, binary);
  memcpy(program->build_hash[device_i],
         b.program_build_hash, sizeof(SHA1_digest_t));
}

cl_uint
pocl_binary_get_kernel_count(unsigned char *binary)
{
  pocl_binary b;
  read_header(&b, binary);
  return b.num_kernels;
}

cl_int
pocl_binary_get_kernel_names(unsigned char *binary,
                             char **kernel_names,
                             size_t num_kernels)
{
  pocl_binary b;
  unsigned char *buffer = read_header(&b, binary);
  assert(num_kernels == b.num_kernels);

  uint64_t struct_size;

  unsigned char *orig_buffer;
  unsigned i, len;
  for (i=0; i < num_kernels; i++)
  {
    orig_buffer = buffer;
    BUFFER_READ(struct_size, uint64_t);
    // skip binaries_size & arginfo_size
    buffer += sizeof(uint64_t) + sizeof(uint32_t);
    BUFFER_READ_STR2(kernel_names[i], len);
    kernel_names[i][len] = 0;
    buffer = orig_buffer + struct_size;
  }
  return CL_SUCCESS;
}

/***********************************************************/

/* serializes a single file. */
static unsigned char*
serialize_file(char* path, size_t basedir_offset, unsigned char* buffer)
{
  char* content;
  uint64_t fsize;
  char* p = path + basedir_offset;
  BUFFER_STORE_STR(p);
  pocl_read_file(path, &content, &fsize);
  BUFFER_STORE_STR2(content, fsize);
  free(content);
  return buffer;
}

/* recursively serializes files/directories by calling
 * either itself (on directory), or serialize_file (on files) */
static unsigned char*
recursively_serialize_path (char* path,
                            size_t basedir_offset,
                            unsigned char* buffer)
{
  struct stat st;
  stat (path, &st);

  if (S_ISREG (st.st_mode))
    buffer = serialize_file (path, basedir_offset, buffer);

  if (S_ISDIR (st.st_mode))
    {
      DIR *d;
      struct dirent *entry;
      char subpath[POCL_FILENAME_LENGTH];

      strncpy (subpath, path, POCL_FILENAME_LENGTH);
      char* p = subpath + strlen(subpath);
      *p++ = '/';
      d = opendir (path);
      while ((entry = readdir (d)))
        {
          if (strcmp (entry->d_name, ".") == 0) continue;
          if (strcmp (entry->d_name, "..") == 0) continue;
          strcpy (p, entry->d_name);
          buffer =
            recursively_serialize_path (subpath, basedir_offset, buffer);
        }
      closedir (d);
    }

  return buffer;
}

/* serializes an entire pocl kernel cachedir. */
static unsigned char*
serialize_kernel_cachedir (cl_kernel kernel,
                           unsigned device_i,
                           unsigned char* buffer)
{
  cl_program program = kernel->program;
  char path[POCL_FILENAME_LENGTH];
  char basedir[POCL_FILENAME_LENGTH];

  pocl_cache_program_path (basedir, program, device_i);
  size_t basedir_len = strlen (basedir);

  pocl_cache_kernel_cachedir (path, program, device_i, kernel);
  POCL_MSG_PRINT_INFO ("Kernel %s: recur serializing cachedir %s\n",
                       kernel->name, path);
  buffer = recursively_serialize_path (path, basedir_len, buffer);

  return buffer;
}

/* serializes a single kernel */
static unsigned char*
pocl_binary_serialize_kernel_to_buffer(cl_kernel kernel,
                                       unsigned device_i,
                                       unsigned char *buf)
{
  unsigned char *buffer = buf;
  unsigned i;

  BUFFER_STORE(0, uint64_t); // struct_size
  BUFFER_STORE(0, uint64_t); // binaries_size
  BUFFER_STORE(0, uint32_t); // arginfo size
  uint32_t namelen = strlen(kernel->name);
  BUFFER_STORE_STR2(kernel->name, namelen);

  BUFFER_STORE(kernel->num_args, uint32_t);
  BUFFER_STORE(kernel->num_locals, uint32_t);

  for (i=0; i < (kernel->num_args + kernel->num_locals); i++)
    {
      BUFFER_STORE(kernel->dyn_arguments[i].size, uint64_t);
    }

  unsigned char *start = buffer;
  for (i=0; i < kernel->num_args; i++)
    {
      pocl_argument_info *ai = &kernel->arg_info[i];
      BUFFER_STORE(ai->access_qualifier, cl_kernel_arg_access_qualifier);
      BUFFER_STORE(ai->address_qualifier, cl_kernel_arg_address_qualifier);
      BUFFER_STORE(ai->type_qualifier, cl_kernel_arg_type_qualifier);
      BUFFER_STORE(ai->is_local, char);
      BUFFER_STORE(ai->is_set, char);
      BUFFER_STORE(ai->type, uint32_t);
      BUFFER_STORE_STR(ai->name);
      BUFFER_STORE_STR(ai->type_name);
    }

  uint32_t arginfo_size = buffer - start;

  unsigned char *end = serialize_kernel_cachedir (kernel, device_i, buffer);
  uint64_t binaries_size = end - buffer;

  /* write struct size properly */
  buffer = buf;
  uint64_t struct_size = end - buf;
  BUFFER_STORE(struct_size, uint64_t);
  BUFFER_STORE(binaries_size, uint64_t);
  BUFFER_STORE(arginfo_size, uint32_t);

  return end;
}

/**
 * Deserializes a single file from the binary to disk.
 *
 * Returns the number of bytes read.
 */
static size_t
deserialize_file (unsigned char* buffer,
                  char* basedir,
                  size_t offset)
{
  unsigned char* orig_buffer = buffer;
  size_t len;

  char* relpath = NULL;
  BUFFER_READ_STR2 (relpath, len);
  assert (len > 0);

  char* content = NULL;
  BUFFER_READ_STR2 (content, len);
  assert (len > 0);

  char *p = basedir + offset;
  strcpy (p, relpath);
  free (relpath);

  char *fullpath = basedir;
  if (pocl_exists (fullpath))
    goto RET;

  char* dir = strdup (basedir);
  char* dirpath = dirname (dir);
  if (!pocl_exists (dirpath))
    pocl_mkdir_p (dirpath);
  free (dir);

  pocl_write_file (fullpath, content, len, 0, 0);

RET:
  free (content);
  return (buffer - orig_buffer);
}

/* Deserializes all files of a single pocl kernel cachedir.  */
static unsigned char*
deserialize_kernel_cachedir (char* basedir, unsigned char* buffer, size_t bytes)
{
  size_t done = 0;
  size_t offset = strlen (basedir);

  while (done < bytes)
    {
      done += deserialize_file (buffer + done, basedir, offset);
    }
  assert(done == bytes);
  return (buffer + done);
}

/* Deserializes a single kernel.

   This has two modes of operation:

   1) if name_len and name_match are non-NULL, it only fills in pocl_binary_kernel
   with metadata (doesn't unpack files) and only if the name matches - used by
   pocl_binary_get_kernel_metadata()
   2) if name_len and name_match are NULL, unpacks kernel cachedir on disk, but
   does not set up kernel metadata of pocl_binary_kernel argument - used by
   pocl_binary_deserialize()
 */
static int
pocl_binary_deserialize_kernel_from_buffer (unsigned char **buf,
                                            pocl_binary_kernel *kernel,
                                            const char* name_match,
                                            size_t name_len,
                                            char* basedir)
{
  unsigned i;
  unsigned char *buffer = *buf;

  memset(kernel, 0, sizeof(pocl_binary_kernel));
  BUFFER_READ(kernel->struct_size, uint64_t);
  BUFFER_READ(kernel->binaries_size, uint64_t);
  BUFFER_READ(kernel->arginfo_size, uint32_t);
  BUFFER_READ_STR2(kernel->kernel_name, kernel->sizeof_kernel_name);
  BUFFER_READ(kernel->num_args, uint32_t);
  BUFFER_READ(kernel->num_locals, uint32_t);

  if (name_len > 0 && name_match)
    {
      *buf = *buf + kernel->struct_size;
      if (kernel->sizeof_kernel_name != name_len)
          return CL_INVALID_KERNEL_NAME;
      if (strncmp (kernel->kernel_name, name_match, kernel->sizeof_kernel_name))
          return CL_INVALID_KERNEL_NAME;

      kernel->dyn_arguments = calloc ((kernel->num_args + kernel->num_locals),
                                      sizeof(struct pocl_argument));
      POCL_RETURN_ERROR_COND ((!kernel->dyn_arguments), CL_OUT_OF_HOST_MEMORY);

      for (i=0; i < (kernel->num_args + kernel->num_locals); i++)
        {
          BUFFER_READ (kernel->dyn_arguments[i].size, uint64_t);
          kernel->dyn_arguments[i].value = NULL;
        }

      kernel->arg_info = calloc (kernel->num_args, sizeof (struct pocl_argument_info));
      POCL_RETURN_ERROR_COND ((!kernel->arg_info), CL_OUT_OF_HOST_MEMORY);

      for (i = 0; i < kernel->num_args; i++)
        {
          pocl_argument_info *ai = &kernel->arg_info[i];
          BUFFER_READ (ai->access_qualifier, cl_kernel_arg_access_qualifier);
          BUFFER_READ (ai->address_qualifier, cl_kernel_arg_address_qualifier);
          BUFFER_READ (ai->type_qualifier, cl_kernel_arg_type_qualifier);
          BUFFER_READ (ai->is_local, char);
          BUFFER_READ (ai->is_set, char);
          BUFFER_READ (ai->type, uint32_t);
          BUFFER_READ_STR (ai->name);
          BUFFER_READ_STR (ai->type_name);
        }
    }
  else
    {
      buffer += ((kernel->num_args + kernel->num_locals) * sizeof (uint64_t));
      buffer += kernel->arginfo_size;
      buffer =
        deserialize_kernel_cachedir (basedir, buffer, kernel->binaries_size);
    }

  *buf = buffer;
  return CL_SUCCESS;

}

/***********************************************************/

cl_int
pocl_binary_serialize(cl_program program, unsigned device_i, size_t *size)
{
  unsigned char *buffer = program->pocl_binaries[device_i];
  size_t sizeof_buffer = program->pocl_binary_sizes[device_i];
  unsigned char *end_of_buffer = buffer + sizeof_buffer;
  unsigned char *start = buffer;

  unsigned num_kernels = program->num_kernels;

  memcpy(buffer, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH);
  buffer += POCLCC_STRING_ID_LENGTH;
  BUFFER_STORE(pocl_binary_get_device_id(program->devices[device_i]), uint64_t);
  BUFFER_STORE(POCLCC_VERSION, uint32_t);
  BUFFER_STORE(num_kernels, uint32_t);
  memcpy(buffer, program->build_hash[device_i], sizeof(SHA1_digest_t));
  buffer += sizeof(SHA1_digest_t);

  assert(buffer < end_of_buffer);

  unsigned i;
  for (i=0; i < num_kernels; i++)
    {
      cl_kernel kernel = program->default_kernels[i];
      buffer = pocl_binary_serialize_kernel_to_buffer(kernel, device_i, buffer);
      assert(buffer <= end_of_buffer);
    }

  if (size)
    *size = (buffer - start);
  return CL_SUCCESS;
}

cl_int
pocl_binary_deserialize(cl_program program, unsigned device_i)
{
  unsigned char *buffer = program->pocl_binaries[device_i];
  size_t sizeof_buffer = program->pocl_binary_sizes[device_i];
  unsigned char *end_of_buffer = buffer + sizeof_buffer;

  pocl_binary b;
  buffer = read_header(&b, buffer);

  //assert(pocl_binary_check_binary_header(&b));
  assert (buffer < end_of_buffer);

  pocl_binary_kernel k;
  char basedir[POCL_FILENAME_LENGTH];

  unsigned i;
  for (i = 0; i < b.num_kernels; i++)
    {
      pocl_cache_program_path (basedir, program, device_i);
      if (pocl_binary_deserialize_kernel_from_buffer
          (&buffer, &k, 0, 0, basedir) != CL_SUCCESS)
        goto ERROR;
      assert (buffer <= end_of_buffer);
    }
  return CL_SUCCESS;

ERROR:
  return CL_OUT_OF_HOST_MEMORY;
}

#define MAX_BINARY_SIZE (256 << 20)

size_t
pocl_binary_sizeof_binary(cl_program program, unsigned device_i)
{
  if (program->pocl_binary_sizes[device_i])
    return program->pocl_binary_sizes[device_i];

  assert(program->pocl_binaries[device_i] == NULL);
  /* dumb solution, but
       * 1) it's simple,
       * 2) we'll likely need the binary itself soon anyway,
       * 3) memory is COW these days.. */
  size_t res = 0;
  unsigned char *temp_buf = malloc(MAX_BINARY_SIZE);
  program->pocl_binaries[device_i] = temp_buf;
  program->pocl_binary_sizes[device_i] = MAX_BINARY_SIZE;

  if (pocl_binary_serialize(program, device_i, &res) != CL_SUCCESS)
    {
      POCL_MEM_FREE(program->pocl_binaries[device_i]);
      program->pocl_binary_sizes[device_i] = 0;
      return 0;
    }

  program->pocl_binaries[device_i] = malloc(res);
  program->pocl_binary_sizes[device_i] = res;
  memcpy(program->pocl_binaries[device_i], temp_buf, res);
  free(temp_buf);
  return res;

}

/***********************************************************/

cl_int
pocl_binary_get_kernel_metadata (unsigned char *binary, const char *kernel_name,
                                cl_kernel kernel, cl_device_id device)
{
  assert (kernel_name);
  size_t name_len = strlen (kernel_name);

  int found = 0;
  pocl_binary b;
  memset(&b, 0, sizeof (pocl_binary));
  pocl_binary_kernel k;
  memset(&k, 0, sizeof (pocl_binary_kernel));

  unsigned char* buffer = read_header (&b, binary);

  POCL_RETURN_ERROR_ON ((!pocl_binary_check_binary (device, binary)),
                        CL_INVALID_PROGRAM,
                        "Deserialized a binary, but it doesn't seem to be "
                        "for this device.\n");

  unsigned j;
  assert (b.num_kernels > 0);
  for (j = 0; j < b.num_kernels; j++)
    {
      if (pocl_binary_deserialize_kernel_from_buffer (
            &buffer, &k, kernel_name, name_len, NULL) == CL_SUCCESS)
        {
          found = 1;
          break;
        }
    }

  POCL_RETURN_ERROR_ON ((!found), CL_INVALID_KERNEL_NAME, "Kernel not found\n");

  kernel->num_args = k.num_args;
  kernel->num_locals = k.num_locals;
  kernel->dyn_arguments = k.dyn_arguments;
  kernel->arg_info = k.arg_info;
  free (k.kernel_name);

  POCL_RETURN_ERROR_COND ((kernel->reqd_wg_size = calloc(3, sizeof(int))) == NULL,
                          CL_OUT_OF_HOST_MEMORY);

  return CL_SUCCESS;
}
