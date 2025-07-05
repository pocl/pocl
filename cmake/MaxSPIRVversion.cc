#include <cstdio>

#include <LLVMSPIRVLib.h>

int main(int, char**) {
  printf("%u", (unsigned)SPIRV::VersionNumber::MaximumVersion);
  return 0;
}
