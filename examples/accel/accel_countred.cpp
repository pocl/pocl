#include <OpenCLcontext.h>

#include <random>
#include <iostream>

#define X 800
#define Y 600

#define SAMPLES 40

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
        std::vector<float> samples;
        while (true) {
            using clock_type = std::chrono::steady_clock;
            using second_type = std::chrono::duration<double, std::ratio<1> >;

            for (size_t i = 0; i < X*Y; ++i)
                framebuffer[i] = dist(mt);
            redpixels = 0;
            std::chrono::time_point<clock_type> m_beg { clock_type::now() };
            mgr.processCameraFrame(reinterpret_cast<unsigned char*>(framebuffer), &redpixels);
            std::chrono::time_point<clock_type> m_end { clock_type::now() };
            double diff = std::chrono::duration_cast<second_type>(m_end - m_beg).count();
            std::cout << "FRAME red pixels: " << redpixels << " time: " << diff << "\n";
            if (samples.size() < SAMPLES) {
              samples.push_back((float)diff);
            } else {
              float sum = std::accumulate(samples.begin(), samples.end(), 0.0f);
              float fps = (float)SAMPLES / sum;
              std::cout << "FPS : " << fps << "\n";
              samples.clear();
            }
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
