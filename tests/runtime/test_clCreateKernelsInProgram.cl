__kernel void
kernel_1()
{
  printf("Hello\n");
}

__kernel void
kernel_2()
{
  printf("World\n");
}

void
notakernel()
{
  printf("This should not get printed!\n");
}

