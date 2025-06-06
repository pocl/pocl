/* OpenCL runtime library: Dynalib library utility functions implemented
   using the LLVM Support library.

   Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_dynlib.h"

#include <llvm/Support/DynamicLibrary.h>
#include <unordered_map>
#include <vector>

using namespace llvm::sys;

// Needed book keeping data to implement the API.
struct DynlibData {
  DynamicLibrary DL;
  std::string Path;
};

std::unordered_map<void *, DynlibData> LoadedLibs;

void *pocl_dynlib_open(const char *Path, int, int) {
  std::string Err;
  DynamicLibrary DL = DynamicLibrary::getLibrary(Path, &Err);
  if (!DL.isValid()) {
    POCL_MSG_ERR("DynamicLibrary::getLibrary() failed: '%s'\n", Err.c_str());
    return NULL;
  }
  void *Handle = DL.getOSSpecificHandle();
  DynlibData D = {DL, Path};
  LoadedLibs[Handle] = D;
  return Handle;
}

int pocl_dynlib_close(void *Handle) {
  auto L = LoadedLibs.find(Handle);
  if (L == LoadedLibs.end())
    return 0;
  DynamicLibrary::closeLibrary((*L).second.DL);
  LoadedLibs.erase(L);
  return 1;
}

void *pocl_dynlib_symbol_address(void *, const char *SymbolName) {
  return DynamicLibrary::SearchForAddressOfSymbol(SymbolName);
}

const char *pocl_dynlib_pathname(void *Address) {
#ifdef _WIN32
  // Look up the module handle from the address
  HMODULE Hm;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         (LPCWSTR)Address, &Hm) == 0) {
    int Ret = GetLastError();
    POCL_MSG_WARN("GetModuleHandleEx failed, error = %d; trying fallback\n",
                  Ret);

    // Undocumented hack from https://stackoverflow.com/a/2396380
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(Address, &mbi, sizeof(mbi)) == 0) {
      Ret = GetLastError();
      POCL_MSG_ERR("VirtualQuery fallback failed, error = %d\n", Ret);
      return nullptr;
    }
    Hm = (HMODULE)mbi.AllocationBase;
  }

  // Get the path of the module
  WCHAR wpath[MAX_PATH];
  if (GetModuleFileNameW(Hm, wpath, ARRAYSIZE(wpath)) == 0) {
    int Ret = GetLastError();
    POCL_MSG_ERR("GetModuleFileName failed, error = %d\n", Ret);
    return nullptr;
  }

  // Open a the file to get a handle
  HANDLE hFile =
      CreateFileW(wpath, GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                  OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    int Ret = GetLastError();
    POCL_MSG_ERR("CreateFile failed, error = %d\n", Ret);
    return nullptr;
  }

  // Get the final path, resolving any symlinks
  std::vector<WCHAR> wfinal_buf(MAX_PATH);
  DWORD path_len = GetFinalPathNameByHandleW(
      hFile, wfinal_buf.data(), wfinal_buf.size(), VOLUME_NAME_DOS);
  if (path_len > wfinal_buf.size()) {
    wfinal_buf.resize(path_len);
    path_len = GetFinalPathNameByHandleW(hFile, wfinal_buf.data(),
                                         wfinal_buf.size(), VOLUME_NAME_DOS);
  }
  CloseHandle(hFile);
  if (path_len == 0) {
    int Ret = GetLastError();
    POCL_MSG_ERR("GetFinalPathNameByHandle failed, error = %d\n", Ret);
    return nullptr;
  }

  // Get rid of the long path prefix
  std::wstring wfinal(wfinal_buf.data(), path_len);
  if (wfinal.rfind(L"\\\\?\\", 0) == 0) {
    wfinal = wfinal.substr(4);
  }

  // Convert the wide string to a UTF-8 multi-byte string
  thread_local std::string final;
  int required_size = WideCharToMultiByte(CP_UTF8, 0, wfinal.c_str(), -1,
                                          NULL, 0, NULL, NULL);
  if (required_size == 0) {
    int Ret = GetLastError();
    POCL_MSG_ERR("WideCharToMultiByte (size check) failed, error = %d\n", Ret);
    return nullptr;
  }
  final.resize(required_size);
  if (WideCharToMultiByte(CP_UTF8, 0, wfinal.c_str(), -1,
                          &final[0], required_size, NULL, NULL) == 0) {
    int Ret = GetLastError();
    POCL_MSG_ERR("WideCharToMultiByte (conversion) failed, error = %d\n", Ret);
    return nullptr;
  }
  final.resize(strlen(final.c_str()));

  POCL_MSG_PRINT_INFO("pocl_dynlib_pathname: using DLL path: %s \n",
                      final.c_str());
  return final.c_str();
#else
  POCL_MSG_ERR("pocl_dynlib_pathname does not have C++/LLVM implementation\n");
  return nullptr;
#endif
}
