/* accel_countred.cpp - Dual-device Carla example for counting red pixels

   Copyright (c) 2022 Topi Leppänen / Tampere University

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
