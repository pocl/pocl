#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

const char *pocl_id="poclbin";

int error(const char *str){
  printf("%s",str);
  return -1;
}

int main(int argc, char **argv){
  if (argc != 2)
    return error("argument error!\n");
  const char *file = argv[1];

  FILE *f = fopen(file, "r");
  fseek(f, 0, SEEK_END);
  uint32_t size = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *buffer = malloc(size);
  fread(buffer, 1, size, f);
  fclose(f);
  char *end_of_buffer = buffer + size;

  if (!strncmp(buffer, pocl_id, strlen(pocl_id)))
    printf("id_string OK\n");
  else
    return error("id_string KO\n");

  buffer += strlen(pocl_id);

  printf("version id: %i\n", *((unsigned *)buffer));
  buffer += sizeof(unsigned);
  assert(buffer < end_of_buffer);

  while (buffer <end_of_buffer){    
    uint32_t sizeofdevice = *((uint32_t *)buffer);
    char *end_of_device = buffer + sizeofdevice + sizeof(unsigned);

    printf("sizeof first device: %i\n", sizeofdevice);
    buffer += sizeof(uint32_t);
    assert(buffer < end_of_buffer);
    
    if (!strncmp(buffer, pocl_id, strlen(pocl_id)))
      printf("pocl_id OK\n");
    else
      return error("pocl_id KO\n");
    buffer += strlen(pocl_id);
    
    printf("device id: %i\n", *((unsigned *)buffer));
    buffer += sizeof(unsigned);
    assert(buffer < end_of_buffer);

    while (buffer <end_of_device){
      uint32_t sizeofkernelname = *((uint32_t *)buffer);
      printf("sizeof kernel name: %i\n", sizeofkernelname);
      buffer += sizeof(uint32_t);
      assert(buffer < end_of_buffer);

      char *kernel_name = malloc(sizeofkernelname*sizeof(char)+1);
      snprintf(kernel_name,sizeofkernelname+1, "%s", buffer);
      kernel_name[sizeofkernelname]='\0';
      printf("kernel name: %s\n",kernel_name);
      buffer += sizeofkernelname;
      assert(buffer < end_of_buffer);

      uint32_t sizeofbinary = *((uint32_t *)buffer);
      printf("binary size: %i\n", sizeofbinary);
      buffer += sizeof(uint32_t);
      assert(buffer < end_of_buffer);

      uint32_t sizeoffilename = strlen(argv[1]);
      char *binary_file_name = malloc(sizeoffilename + sizeofkernelname + 2);
      sprintf(binary_file_name, "%s.%s", argv[1], kernel_name);
      binary_file_name[sizeoffilename + sizeofkernelname + 1]='\0';
    
      FILE *f=fopen(binary_file_name, "w");
      fwrite(buffer, 1, sizeofbinary, f);
      fclose(f);
      buffer += sizeofbinary;
    
      free(binary_file_name);
      free(kernel_name);
    }
  }
}
