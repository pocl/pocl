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

/* pocl binary identifier */
#define POCLCC_STRING_ID "poclbin"
#define POCLCC_STRING_ID_LENGTH 8
/* changes for version 2: added program.bc right after header */
/* changes for version 3: added flush_denorms flag into header */
/* changes for version 4: kernel library is now linked into
                          program.bc, so older binaries may fail
                          to run with "undefined symbol" errors. */
/* changes for version 5: added program binary_type into header */
/* changes for version 6: added reqd_wg_size informations into
                          pocl_binary_kernel structure */
/* changes for version 7: removed dyn_arguments from storage, instead added
                          arg_info[i]->type_size; removed is_local and is_set
                          from storage, no need to store these (is_set makes
                          no sense in binary, and whether argument is local
                          is already in cl_kernel_arg_address_qualifier);
                          add has_arg_metadata & kernel attributes */
/* changes for version 8: added local_alignments after local_sizes */

#define FIRST_SUPPORTED_POCLCC_VERSION 6
#define POCLCC_VERSION 8

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

  /* number of kernel arguments */
  uint32_t num_args;
  /* number of kernel local variables */
  uint32_t num_locals;

  /* required work-group size */
  uint64_t reqd_wg_size[OPENCL_MAX_DIMENSION];

  uint64_t has_arg_metadata;

  uint32_t sizeof_attributes;
  char* attributes;

  /* arguments and argument metadata. Note that not everything is stored
   * in the serialized binary */
  size_t *local_sizes;
  size_t *local_alignments;
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
  /* various flags */
  uint64_t flags;
  /* program->build_hash[device_i], required to restore files into pocl cache */
  SHA1_digest_t program_build_hash;
} pocl_binary;

#define POCL_BINARY_FLAG_FLUSH_DENORMS (1 << 0)

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
  BUFFER_READ (b->flags, uint64_t);
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
  if (strncmp (b.pocl_id, POCLCC_STRING_ID, POCLCC_STRING_ID_LENGTH))
    {
      POCL_MSG_WARN ("File is not a pocl binary\n");
      return NULL;
    }
  if (b.version < FIRST_SUPPORTED_POCLCC_VERSION)
    {
      POCL_MSG_WARN ("PoclBinary version %i is not supported by "
                     "this pocl (the minimal is: %i)\n",
                     b.version, FIRST_SUPPORTED_POCLCC_VERSION);
      return NULL;
    }
  if (pocl_binary_get_device_id(device) != b.device_id)
    {
      POCL_MSG_WARN ("PoclBinary device id mismatch\n");
      return NULL;
    }
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
pocl_binary_get_kernel_count (cl_program program, unsigned device_i)
{
  unsigned char *binary = program->pocl_binaries[device_i];
  pocl_binary b;
  read_header(&b, binary);
  return b.num_kernels;
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

      strncpy (subpath, path, POCL_FILENAME_LENGTH-1);
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
serialize_kernel_cachedir (cl_program program,
                           const char* kernel_name,
                           unsigned device_i,
                           unsigned char* buffer)
{
  char path[POCL_FILENAME_LENGTH];
  char basedir[POCL_FILENAME_LENGTH];

  pocl_cache_program_path (basedir, program, device_i);
  size_t basedir_len = strlen (basedir);

  pocl_cache_kernel_cachedir (path, program, device_i, kernel_name);
  POCL_MSG_PRINT_INFO ("Kernel %s: recur serializing cachedir %s\n",
                       kernel_name, path);
  buffer = recursively_serialize_path (path, basedir_len, buffer);

  return buffer;
}

/* serializes a single kernel */
static unsigned char*
pocl_binary_serialize_kernel_to_buffer(cl_program program,
                                       pocl_kernel_metadata_t *meta,
                                       unsigned device_i,
                                       unsigned char *buf)
{
  unsigned char *buffer = buf;
  unsigned i;

  BUFFER_STORE(0, uint64_t); // struct_size
  BUFFER_STORE(0, uint64_t); // binaries_size
  BUFFER_STORE(0, uint32_t); // arginfo size
  uint32_t namelen = strlen (meta->name);
  BUFFER_STORE_STR2 (meta->name, namelen);
  BUFFER_STORE (meta->num_args, uint32_t);
  BUFFER_STORE (meta->num_locals, uint32_t);

  for (i = 0; i < OPENCL_MAX_DIMENSION; i++)
    {
      BUFFER_STORE (meta->reqd_wg_size[i], uint64_t);
    }

  for (i = 0; i < meta->num_locals; i++)
    {
      uint64_t temp = meta->local_sizes[i];
      BUFFER_STORE (temp, uint64_t);
      temp = meta->local_alignments[i];
      BUFFER_STORE (temp, uint64_t);
    }

  uint32_t attrlen = meta->attributes ? strlen (meta->attributes) : 0;
  BUFFER_STORE_STR2(meta->attributes, attrlen);
  BUFFER_STORE(meta->has_arg_metadata, uint64_t);

  /***********************************************************************/
  unsigned char *start = buffer;
  for (i = 0; i < meta->num_args; i++)
    {
      pocl_argument_info *ai = &meta->arg_info[i];
      BUFFER_STORE(ai->access_qualifier, cl_kernel_arg_access_qualifier);
      BUFFER_STORE(ai->address_qualifier, cl_kernel_arg_address_qualifier);
      BUFFER_STORE(ai->type_qualifier, cl_kernel_arg_type_qualifier);
      BUFFER_STORE(ai->type, uint32_t);
      BUFFER_STORE (ai->type_size, uint32_t);
      BUFFER_STORE_STR(ai->name);
      BUFFER_STORE_STR(ai->type_name);
    }
  /***********************************************************************/

  uint32_t arginfo_size = buffer - start;

  unsigned char *end
      = serialize_kernel_cachedir (program, meta->name, device_i, buffer);
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

   1) if kernel_index is >= 0 it only fills in pocl_binary_kernel
   with metadata (doesn't unpack files) and only if the name matches - used by
   pocl_binary_get_kernel_metadata()

   2) if name_len and name_match are NULL, unpacks kernel cachedir on disk, but
   does not set up kernel metadata of pocl_binary_kernel argument - used by
   pocl_binary_deserialize()
 */

static int
pocl_binary_deserialize_kernel_from_buffer (pocl_binary *b,
                                            unsigned char **buf,
                                            pocl_binary_kernel *kernel,
                                            pocl_kernel_metadata_t *meta,
                                            char *basedir)
{
  unsigned i;
  unsigned char *buffer = *buf;
  uint64_t *dynarg_sizes;

  memset(kernel, 0, sizeof(pocl_binary_kernel));
  BUFFER_READ(kernel->struct_size, uint64_t);
  BUFFER_READ(kernel->binaries_size, uint64_t);
  BUFFER_READ(kernel->arginfo_size, uint32_t);
  BUFFER_READ_STR2(kernel->kernel_name, kernel->sizeof_kernel_name);
  BUFFER_READ(kernel->num_args, uint32_t);
  BUFFER_READ(kernel->num_locals, uint32_t);

  dynarg_sizes = alloca (sizeof(uint64_t) * kernel->num_args);

  for (i = 0; i < OPENCL_MAX_DIMENSION; i++)
    {
      BUFFER_READ(kernel->reqd_wg_size[i], uint64_t);
    }

  if (meta)
    {
      if (b->version < 7)
        {
          for (i = 0; i < kernel->num_args; i++)
            {
              BUFFER_READ (dynarg_sizes[i], uint64_t);
            }
        }

      kernel->local_sizes = calloc (kernel->num_locals, sizeof (size_t));
      if (b->version >=8)
        {
          kernel->local_alignments = calloc (kernel->num_locals, sizeof (size_t));
        }
      for (i = 0; i < kernel->num_locals; i++)
        {
          uint64_t temp;
          BUFFER_READ (temp, uint64_t);
          kernel->local_sizes[i] = temp;
          if (b->version >=8)
            {
              BUFFER_READ (temp, uint64_t);
              kernel->local_alignments[i] = temp;
            }
        }

      if (b->version >= 7)
        {
          BUFFER_READ_STR2(kernel->attributes, kernel->sizeof_attributes);
          BUFFER_READ(kernel->has_arg_metadata, uint64_t);
        }
      else
        {
          kernel->attributes = NULL;
          kernel->has_arg_metadata = (-1);
        }

      meta->arg_info = calloc (kernel->num_args, sizeof (struct pocl_argument_info));
      POCL_RETURN_ERROR_COND ((!meta->arg_info), CL_OUT_OF_HOST_MEMORY);

      for (i = 0; i < kernel->num_args; i++)
        {
          pocl_argument_info *ai = &meta->arg_info[i];
          BUFFER_READ (ai->access_qualifier, cl_kernel_arg_access_qualifier);
          BUFFER_READ (ai->address_qualifier, cl_kernel_arg_address_qualifier);
          BUFFER_READ (ai->type_qualifier, cl_kernel_arg_type_qualifier);
          if (b->version < 7)
            {
              char t1, t2;
              BUFFER_READ (t1, char);
              BUFFER_READ (t2, char);
            }

          BUFFER_READ (ai->type, uint32_t);
          if (b->version >= 7)
            {
              BUFFER_READ (ai->type_size, uint32_t);
            }
          else
            {
              ai->type_size = dynarg_sizes[i];
            }
          BUFFER_READ_STR (ai->name);
          BUFFER_READ_STR (ai->type_name);
        }

    }
  else
    {
      /* skip the arg_info and all kernel metadata */
      buffer = *buf + (kernel->struct_size - kernel->binaries_size);
      deserialize_kernel_cachedir (basedir, buffer, kernel->binaries_size);
      POCL_MEM_FREE (kernel->kernel_name);
    }

  /* always skip to the next kernel */
  *buf = *buf + kernel->struct_size;
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
  uint64_t flags = 0;
  if (program->flush_denorms)
    flags |= POCL_BINARY_FLAG_FLUSH_DENORMS;
  flags |= (program->binary_type << 1);
  BUFFER_STORE (flags, uint64_t);
  memcpy(buffer, program->build_hash[device_i], sizeof(SHA1_digest_t));
  buffer += sizeof(SHA1_digest_t);

  assert(buffer < end_of_buffer);

  char basedir[POCL_FILENAME_LENGTH];
  pocl_cache_program_path (basedir, program, device_i);
  size_t basedir_len = strlen (basedir);
  char program_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_program_bc_path (program_bc_path, program, device_i);
  POCL_MSG_PRINT_INFO ("serializing program.bc: %s\n", program_bc_path);
  buffer = serialize_file (program_bc_path, basedir_len, buffer);

  unsigned i;
  for (i=0; i < num_kernels; i++)
    {
      buffer = pocl_binary_serialize_kernel_to_buffer
                 (program, &program->kernel_meta[i], device_i, buffer);
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
  program->flush_denorms = (b.flags & POCL_BINARY_FLAG_FLUSH_DENORMS);
  program->binary_type = (b.flags >> 1);

  //assert(pocl_binary_check_binary_header(&b));
  assert (buffer < end_of_buffer);

  char basedir[POCL_FILENAME_LENGTH];
  pocl_cache_program_path (basedir, program, device_i);
  size_t basedir_len = strlen (basedir);
  buffer += deserialize_file (buffer, basedir, basedir_len);

  pocl_binary_kernel k;
  unsigned i;
  for (i = 0; i < b.num_kernels; i++)
    {
      pocl_cache_program_path (basedir, program, device_i);
      if (pocl_binary_deserialize_kernel_from_buffer (&b, &buffer, &k, NULL, basedir)
          != CL_SUCCESS)
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
pocl_binary_get_kernels_metadata (cl_program program, unsigned device_i)
{
  unsigned char *binary = program->pocl_binaries[device_i];
  cl_device_id device = program->devices[device_i];

  pocl_binary b;
  memset(&b, 0, sizeof (pocl_binary));
  pocl_binary_kernel k;
  memset(&k, 0, sizeof (pocl_binary_kernel));

  unsigned char* buffer = read_header (&b, binary);
  POCL_RETURN_ERROR_ON ((!pocl_binary_check_binary (device, binary)),
                        CL_INVALID_PROGRAM,
                        "Deserialized a binary, but it doesn't seem to be "
                        "for this device.\n");
  size_t len;
  /* skip real path of program.bc */
  BUFFER_READ(len, uint32_t);
  assert (len > 0);
  buffer += len;

  /* skip content of program.bc */
  BUFFER_READ(len, uint32_t);
  assert (len > 0);
  buffer += len;

  unsigned j;
  assert (b.num_kernels > 0);
  assert (b.num_kernels == program->num_kernels);

  /* for each kernel, setup its metadata */
  for (j = 0; j < b.num_kernels; j++)
    {
      pocl_kernel_metadata_t *km = &program->kernel_meta[j];

      POCL_RETURN_ERROR_ON (pocl_binary_deserialize_kernel_from_buffer (
                                 &b, &buffer, &k, km, NULL),
                            CL_INVALID_PROGRAM,
                            "Can't deserialize kernel %u \n", j);

      km->num_args = k.num_args;
      km->num_locals = k.num_locals;
      km->local_sizes = k.local_sizes;
      km->local_alignments = k.local_alignments;
      km->attributes = k.attributes;
      km->has_arg_metadata = k.has_arg_metadata;
      km->name = k.kernel_name;
      km->data = (void **)calloc (program->num_devices, sizeof (void *));
      assert (km->name);

      unsigned l;
      for (l = 0; l < OPENCL_MAX_DIMENSION; l++)
        {
          km->reqd_wg_size[l] = k.reqd_wg_size[l];
        }
    }

  return CL_SUCCESS;
}
