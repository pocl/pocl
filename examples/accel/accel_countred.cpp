#include <OpenCLcontext.h>

#include <random>
#include <iostream>

#define X 800
#define Y 600

int
main(void)
{
    try {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<unsigned int> dist;

        unsigned int framebuffer[X*Y];
        OpenCL_Manager mgr;
        mgr.initialize(X, Y);
        unsigned long redpixels = 0;
        while (true) {
            for (size_t i = 0; i < X*Y; ++i)
                framebuffer[i] = dist(mt);
            redpixels = 0;
            mgr.processCameraFrame(reinterpret_cast<unsigned char*>(framebuffer), &redpixels);
            std::cout << "FRAME red pixels: " << redpixels << "\n";
        }
    }
    catch (std::exception &e) {
         std::cerr
             << "ERROR: "
             << e.what()
             << std::endl;

         return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
