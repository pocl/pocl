#if !defined(_MSC_VER) && !defined(__MINGW32__)
#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#define _BSD_SOURCE
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#else
#include <io.h>
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "pocl.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"

#ifdef __ANDROID__

int pocl_mkstemp(char *path);

#endif

/*****************************************************************************/

int
pocl_rm_rf(const char* path) 
{
  DIR *d = opendir(path);
  size_t path_len = strlen(path);
  int error = -1;
  
  if(d) 
    {
      struct dirent *p = readdir(d);
      error = 0;
      while (!error && p)
        {
          char *buf;
          if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, ".."))
            continue;
          
          size_t len = path_len + strlen(p->d_name) + 2;
          buf = malloc(len);
          buf[len] = '\0';
          if (buf)
            {
              struct stat statbuf;
              snprintf(buf, len, "%s/%s", path, p->d_name);
              
              if (!stat(buf, &statbuf))
                error = pocl_rm_rf(buf);
              else 
                error = remove(buf);
              
              free(buf);
            }
          p = readdir(d);
        }
      closedir(d);
      
      if (!error)
        remove(path);
    }
  return error;
}


int
pocl_mkdir_p (const char* path)
{
  int error;
  int errno_tmp;
  error = mkdir (path, S_IRWXU);
  if (error && errno == ENOENT)
    { // creates all needed directories recursively
      char *previous_path;
      int previous_path_length = strrchr (path, '/') - path;
      previous_path = malloc (previous_path_length + 1);
      strncpy (previous_path, path, previous_path_length);
      previous_path[previous_path_length] = '\0';
      pocl_mkdir_p ((const char*) previous_path);
      free (previous_path);
      error = mkdir (path, S_IRWXU);
    }
  else if (error && errno == EEXIST)
    error = 0;

  return error;
}

int
pocl_remove(const char* path) 
{
  return remove(path);
}

int
pocl_exists(const char* path) 
{
  return !access(path, R_OK);
}

int
pocl_filesize(const char* path, uint64_t* res) 
{
  FILE *f = fopen(path, "r");
  if (f == NULL)
    return -1;
  
  fseek(f, 0, SEEK_END);
  *res = ftell(f);
  
  fclose(f);
  return 0;
}

int 
pocl_touch_file(const char* path) 
{
  FILE *f = fopen(path, "w");
  if (f)
    {
      fclose(f);
      return 0;
    }
  return -1;
}
/****************************************************************************/

int
pocl_read_file(const char* path, char** content, uint64_t *filesize) 
{
  assert(content);
  assert(path);
  assert(filesize);
  
  *content = NULL;
  
  int errcode = pocl_filesize(path, filesize);
  ssize_t fsize = (ssize_t)(*filesize);
  if (!errcode) 
    {
      FILE *f = fopen(path, "r");
      
      *content = (char*)malloc(fsize+1);
      
      size_t rsize = fread(*content, 1, fsize, f);
      (*content)[rsize] = '\0';
      if (rsize < (size_t)fsize) 
        {
          errcode = -1;
          fclose(f);
        } 
      else 
        {
          if (fclose(f))
            errcode = -1;
        }
    }
  return errcode;
}



int 
pocl_write_file(const char *path, const char* content,
                uint64_t    count,
                int         append,
                int         dont_rewrite) 
{
  assert(path);
  assert(content);
  
  if (pocl_exists(path)) 
    {
      if (dont_rewrite) 
        {
          if (!append)
            return 0;
        } 
      else 
        {
          int res = pocl_remove(path);
          if (res)
            {
              POCL_MSG_ERR ("pocl_remove(%s) failed\n", path);
              return res;
            }
        }
    }
  
  FILE *f;
  if (append)
    f = fopen(path, "a");
  else
    f = fopen(path, "w");

  if (f == NULL)
    {
      POCL_MSG_ERR ("fopen(%s) failed\n", path);
      return -1;
    }

  if (fwrite (content, 1, (size_t)count, f) < (size_t)count)
    {
      POCL_MSG_ERR ("fwrite(%s) failed\n", path);
      return -1;
    }

  if (fflush (f))
    {
      POCL_MSG_ERR ("fflush() failed\n");
      return errno;
    }

#ifdef HAVE_FDATASYNC
  if (fdatasync (fileno (f)))
    {
      POCL_MSG_ERR ("fdatasync() failed\n");
      return errno;
    }
#elif defined(HAVE_FSYNC)
  if (fsync (fileno (f)))
    {
      POCL_MSG_ERR ("fsync() failed\n");
      return errno;
    }
#endif

  return fclose(f);
}

/****************************************************************************/

int pocl_rename(const char *oldpath, const char *newpath) {
  return rename (oldpath, newpath);
}

int
pocl_mk_tempname (char *output, const char *prefix, const char *suffix,
                  int *ret_fd)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
#error "making temporary files on Windows not implemented"
#elif defined(HAVE_MKOSTEMPS) || defined(HAVE_MKSTEMPS) || defined(__ANDROID__)
  /* using mkstemp() instead of tmpnam() has no real benefit
   * here, as we have to pass the filename to llvm,
   * but tmpnam() generates an annoying warning... */
  int fd;

  strncpy (output, prefix, POCL_FILENAME_LENGTH);
  size_t len = strlen (prefix);
  strncpy (output + len, "_XXXXXX", (POCL_FILENAME_LENGTH - len));

#ifdef __ANDROID__
  fd = pocl_mkstemp (output);
#else
  if (suffix)
    {
      len += 7;
      strncpy (output + len, suffix, (POCL_FILENAME_LENGTH - len));
#ifdef HAVE_MKOSTEMPS
      fd = mkostemps (output, strlen (suffix), O_CLOEXEC);
#else
      fd = mkstemps (output, strlen (suffix));
#endif
    }
  else
#ifdef HAVE_MKOSTEMPS
    fd = mkostemp (output, O_CLOEXEC);
#else
    fd = mkstemp (output);
#endif
#endif

  if (fd < 0)
    {
      POCL_MSG_ERR ("mkstemp() failed\n");
      return errno;
    }

  int err = 0;
  if (ret_fd)
    *ret_fd = fd;
  else
    err = close (fd);

  return err ? errno : 0;

#else
#error mkostemps() / mkstemps() both unavailable
#endif
}

int
pocl_mk_tempdir (char *output, const char *prefix)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
  assert (0);
#elif defined(HAVE_MKDTEMP)
  /* TODO mkdtemp() might not be portable outside Linux */
  strncpy (output, prefix, POCL_FILENAME_LENGTH);
  size_t len = strlen (prefix);
  strncpy (output + len, "_XXXXXX", (POCL_FILENAME_LENGTH - len));
  return (mkdtemp (output) == NULL);
#else
#error mkdtemp() not available
#endif
}

/* write content[count] into a temporary file, and return the tempfile name in
 * output_path */
int
pocl_write_tempfile (char *output_path, const char *prefix, const char *suffix,
                     const char *content, uint64_t count, int *ret_fd)
{
  assert (output_path);
  assert (prefix);
  assert (suffix);
  assert (content);
  assert (count > 0);

  int fd, err;

  err = pocl_mk_tempname (output_path, prefix, suffix, &fd);
  if (err)
    {
      POCL_MSG_ERR ("pocl_mk_tempname() failed\n");
      return err;
    }
  size_t bytes = (size_t)count;
  ssize_t res;
  do
    {
      res = write (fd, content, bytes);
      if (res < 0)
        {
          POCL_MSG_ERR ("write(%s) failed\n", output_path);
          return errno;
        }
      else
        {
          bytes -= res;
          content += res;
        }
    }
  while (bytes > 0);

#ifdef HAVE_FDATASYNC
  if (fdatasync (fd))
    {
      POCL_MSG_ERR ("fdatasync() failed\n");
      return errno;
    }
#elif defined(HAVE_FSYNC)
  if (fsync (fd))
    return errno;
#endif

  err = 0;
  if (ret_fd)
    *ret_fd = fd;
  else
    err = close (fd);

  return err ? errno : 0;
}
