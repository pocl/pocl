/* preprocess.cc - client-side preprocessing for the remote driver

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, and/or copy the Software,
   and to permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

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

/* Preprocesses the kernel source on the client side using the vendored
   simplecpp library (thirdparty/simplecpp) so that #include directives are
   resolved against the client's files. The server then receives a fully
   expanded, self-contained translation unit. */

#include "preprocess.h"

#include "pocl_debug.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "simplecpp.h"

#define PROGRAM_FILENAME "program.cl"

/* OpenCL C version (e.g. 120 for CL 1.2) from a "-cl-std=CLx.y" build
   option string, or 0 if not present. Mirrors the parsing in
   pocl_llvm_build.cc. */
static int
clStdFromOptions (const char *options)
{
  if (!options)
    return 0;
  std::string opts (options);
  size_t pos = opts.find ("-cl-std=CL");
  if (pos == std::string::npos)
    return 0;
  pos += 9; // skip until first digit after "=CL"
  if (pos + 1 >= opts.size ())
    return 0;
  int major = opts[pos] - '0';
  int minor = opts[pos + 2] - '0';
  if (major < 0 || major > 9 || minor < 0 || minor > 9)
    return 0;
  return major * 100 + minor * 10;
}

static void
defineSpaceSeparated (simplecpp::DUI &dui, const char *str)
{
  if (!str)
    return;
  std::istringstream iss (str);
  std::string tok;
  while (iss >> tok)
    dui.defines.push_back (tok + "=1");
}

static void
setupPredefines (simplecpp::DUI &dui, cl_program program, cl_device_id device)
{
  int clStd = clStdFromOptions (program->compiler_options);
  if (clStd == 0)
    clStd = (int) device->version_as_int;

  dui.defines.push_back ("__OPENCL_VERSION__=" +
                         std::to_string (device->version_as_int));
  dui.defines.push_back ("__OPENCL_C_VERSION__=" + std::to_string (clStd));

  if (program->compiler_options &&
      strstr (program->compiler_options, "-cl-fast-relaxed-math"))
    dui.defines.push_back ("__FAST_RELAXED_MATH__=1");

  if (device->endian_little)
    dui.defines.push_back ("__ENDIAN_LITTLE__=1");
  if (device->image_support)
    dui.defines.push_back ("__IMAGE_SUPPORT__=1");

  dui.defines.push_back ("CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=" +
                         std::to_string (device->global_var_max_size));

  // same rules as the LLVM compiler front-end on the server side
  defineSpaceSeparated (dui, device->extensions);
  if (clStd >= 300)
    defineSpaceSeparated (dui, device->features);
}

/* splits the build options on whitespace and handles the -D/-U/-I options
   that also affect preprocessing. */
static void
setupUserOptions (simplecpp::DUI &dui, const char *options)
{
  if (!options)
    return;
  std::istringstream iss (options);
  std::string tok;
  while (iss >> tok)
    {
      if (tok.size () > 2 && tok[0] == '-' && tok[1] == 'D')
        dui.defines.push_back (tok.substr (2));
      else if (tok.size () > 2 && tok[0] == '-' && tok[1] == 'U')
        dui.undefined.insert (tok.substr (2));
      else if (tok.size () > 2 && tok[0] == '-' && tok[1] == 'I')
        dui.includePaths.push_back (tok.substr (2));
    }
}

extern "C" int
pocl_remote_preprocess_source (cl_program program, cl_device_id device,
                               cl_uint num_input_headers,
                               const cl_program *input_headers,
                               const char **header_include_names,
                               char **out_source, char **err_msg)
{
  *out_source = nullptr;
  *err_msg = nullptr;

  simplecpp::DUI dui;
  setupPredefines (dui, program, device);
  setupUserOptions (dui, program->compiler_options);

  std::vector<std::string> files;
  simplecpp::OutputList outputList;
  simplecpp::FileDataCache fileCache;

  // clCompileProgram() input headers must be resolvable both as
  // #include "name" and #include <name> without touching the disk;
  // system-style lookups are keyed as "<includePath>/<name>", hence the
  // extra "" include path and the "/<name>" virtual files
  if (num_input_headers > 0)
    dui.includePaths.push_back (std::string ());
  for (cl_uint i = 0; i < num_input_headers; ++i)
    {
      cl_program header = input_headers[i];
      const char *name = header_include_names[i];
      if (!header || !header->source || !name)
        continue;
      for (int abs = 0; abs < 2; ++abs)
        {
          std::string fname = abs ? ("/" + std::string (name)) : name;
          fileCache.insert (simplecpp::FileData {
              fname,
              simplecpp::TokenList (std::string_view (
                                        header->source,
                                        strlen (header->source)),
                                    files, fname, dui, &outputList) });
        }
    }

  simplecpp::TokenList *rawtokens = new simplecpp::TokenList (
      std::string_view (program->source, strlen (program->source)), files,
      PROGRAM_FILENAME, dui, &outputList);

  simplecpp::TokenList output (files);
  simplecpp::preprocess (output, *rawtokens, files, fileCache, dui,
                         &outputList);
  delete rawtokens;

  std::ostringstream errs;
  bool failed = false;
  for (const simplecpp::Output &o : outputList)
    {
      if (o.type != simplecpp::Output::ERROR
          && o.type != simplecpp::Output::SYNTAX_ERROR
          && o.type != simplecpp::Output::MISSING_HEADER)
        continue;
      failed = true;
      errs << output.file (o.location) << ':' << o.location.line << ": "
           << o.msg << '\n';
    }

  simplecpp::cleanup (fileCache);

  if (failed)
    {
      std::string msg = errs.str ();
      *err_msg = strdup (msg.c_str ());
      POCL_MSG_PRINT_REMOTE ("client-side preprocessing failed: %s\n",
                             msg.c_str ());
      return CL_BUILD_PROGRAM_FAILURE;
    }

  std::string preprocessed = output.stringify (false);
  *out_source = strdup (preprocessed.c_str ());

  POCL_MSG_PRINT2 (REMOTE, __func__, __LINE__,
                   "preprocessed source size: %zu\n", preprocessed.size ());
  return CL_SUCCESS;
}
