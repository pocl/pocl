// g++ -Og -ggdb3 -I/usr/include -o test_math test_math.cc -lOpenCL -pthread

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream> // std::stringstream
#include <streambuf>
#include <string>

#include <chrono>
#include <thread>

// #define DEBUGG

#ifdef DEBUGG
#define ITERS 2
#define CHUNK_SIZE_B 64000
#else
#define ITERS 1000
#define CHUNK_SIZE_B 64000
#endif

#define SEED 75902834

using namespace std;

static const char *SOURCE_1_1 = R"RAW(

kernel void generate(global const ARGTYPE* restrict in1, global ARGTYPE* restrict out1)
{
  size_t i = get_global_id(0);
  out1[i] = FUNCNAME(in1[i]);
}

)RAW";

static const char *SOURCE_2_1 = R"RAW(

kernel void generate(global const ARGTYPE* restrict in1, global const ARGTYPE* restrict in2, global ARGTYPE* restrict out1)
{
  size_t i = get_global_id(0);
  out1[i] = FUNCNAME(in1[i], in2[i]);
}

)RAW";

static const char *get_source(unsigned num_inputs, unsigned num_outputs) {

  if (num_inputs == 1 && num_outputs == 1)
    return SOURCE_1_1;

  throw std::runtime_error("has no kernel source for these inputs + outputs!");
}

static void get_kernel(cl::Program &prog, cl::Kernel &ker, std::string &vectype,
                       std::string &func_name, bool has_compiler,
                       unsigned num_inputs, unsigned num_outputs) {

  if (has_compiler) {

    const char *s = get_source(num_inputs, num_outputs);
    cl::Program::Sources sources1({s});
    prog = cl::Program(sources1);

    std::string options = "-cl-std=CL1.2";
    options += " -DARGTYPE=" + vectype + " -DFUNCNAME=" + func_name;
    prog.build(options.c_str());

    ker = cl::Kernel(prog, "generate");

  } else {

    std::string binary_name(func_name);
    binary_name += "_" + vectype + ".poclbin";

    std::ifstream rfile;
    rfile.exceptions(ifstream::badbit);
    rfile.open(binary_name, ios_base::in | ios_base::binary);

    std::stringstream prog_ss;
    prog_ss << rfile.rdbuf();
    rfile.close();

    std::string binary(prog_ss.str());
    std::cerr << "Binary sizE: " << binary.size() << "\n";

    cl::Program::Binaries binaries;
    binaries.resize(1);
    binaries[0] = std::vector<unsigned char>(binary.begin(), binary.end());
    std::vector<cl::Device> devices;
    devices.resize(1);
    devices[0] = cl::Device::getDefault();
    prog = cl::Program(cl::Context::getDefault(), devices, binaries);
    prog.build();
    ker = cl::Kernel(prog, "generate");
  }
}

int compile_only(std::string &func_name, unsigned vecsize, bool has_compiler,
                 unsigned num_inputs, unsigned num_outputs) {
  if (!has_compiler)
    throw std::runtime_error("No OpenCL compiler for this device!");

  cl::Program program(get_source(num_inputs, num_outputs));

  std::string vectype("float");
  if (vecsize > 1)
    vectype += to_string(vecsize);

  std::string options = "-cl-std=CL1.2";
  options += " -DARGTYPE=" + vectype + " -DFUNCNAME=" + func_name;
  program.build(options.c_str());

  std::string binary_name(func_name);
  binary_name += "_" + vectype + ".poclbin";

  std::ofstream rfile;
  rfile.exceptions(ofstream::badbit);
  rfile.open(binary_name, ios_base::out | ios_base::binary | ios_base::trunc);

  cl::Program::Binaries binaries;
  program.getInfo<>(CL_PROGRAM_BINARIES, &binaries);

  if (binaries.size() != 1)
    throw std::runtime_error("wrong siZe of binaries\n");

  rfile.write((const char *)binaries[0].data(), binaries[0].size());
  rfile.close();

  return 0;
}

union casto {
  float f;
  int32_t i;
};

#define SIGN_EXPBITS_SP32 0xff800000
#define MANTBITS_SP32 0x007fffff

// last two bits can differ (= 2 ULP)
static bool epsilon_ok(float ref, float act) {
  casto c1, c2;
  c1.f = ref;
  int32_t ref_u = c1.i;
  c2.f = act;
  int32_t act_u = c2.i;

  int32_t mant_diff = (ref_u & 0x007fffff) - (act_u & 0x007fffff);
  int32_t exp_diff = (ref_u & SIGN_EXPBITS_SP32) - (act_u & SIGN_EXPBITS_SP32);
  if (mant_diff < 0)
    mant_diff = -mant_diff;

  return ((exp_diff == 0) && (mant_diff < 4));
}

#define MAX_INPUTS 4
#define MAX_OUTPUTS 4

static bool compare_floats(unsigned num_inputs, float **inputs,
                           unsigned num_outputs, float **ref_inputs,
                           float **output) {
  unsigned errs = 0;
  unsigned floats_in_chunk = CHUNK_SIZE_B / 4;

  for (unsigned k = 0; k < floats_in_chunk; ++k) {
    for (unsigned o = 0; o < num_outputs; ++o) {
#ifdef DEBUGG
      std::cerr << "REF " << ref_inputs[o][k] << "  OUT " << output[o][k]
                << "\n";
#endif
      if (!epsilon_ok(ref_inputs[o][k], output[o][k])) {
        std::cerr << "ERROR @ IDX " << k << " Inputs[0][IDX] " << inputs[0][k]
                  << " Outputs[" << o << "][k] too different: expected "
                  << ref_inputs[o][k] << ",  got " << output[o][k] << "\n";
        ++errs;
      }
      if (errs > 100)
        break;
    }
  }

  if (errs > 0)
    std::cerr << "Total errs: " << errs << "\n";
  else
    std::cerr << "Chunk compare OK\n";
  return (errs == 0);
}

int generate_or_verify(bool verify, std::string &func_name, unsigned vecsize,
                       bool has_compiler, unsigned num_inputs,
                       unsigned num_outputs) {

  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();

  unsigned vecsize_bytes = vecsize * 4;
  std::string vectype("float");
  if (vecsize > 1)
    vectype += to_string(vecsize);
  unsigned floats_in_chunk = CHUNK_SIZE_B / 4;
  unsigned vecs_in_chunk = CHUNK_SIZE_B / vecsize_bytes;

  cl::Program program;
  cl::Kernel kernel;
  get_kernel(program, kernel, vectype, func_name, has_compiler, num_inputs,
             num_outputs);

  if ((num_inputs == 0) || (num_outputs == 0))
    throw std::runtime_error("Zero inputs or outputs!");

  cl::Buffer input_cl[MAX_INPUTS];
  float *input[MAX_INPUTS];
  cl::Buffer output_cl[MAX_OUTPUTS];
  float *output[MAX_OUTPUTS];

  // output for generate & input for verify
  std::fstream ref_files[MAX_OUTPUTS];
  // only for verify
  float *ref_inputs[MAX_OUTPUTS];

  unsigned arg_n = 0;
  for (unsigned i = 0; i < num_inputs; ++i) {
    input[i] = new float[floats_in_chunk];
    input_cl[i] =
        cl::Buffer((cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
                   CHUNK_SIZE_B, input[i]);
    kernel.setArg(arg_n++, input_cl[i]);
  }

  for (unsigned i = 0; i < num_outputs; ++i) {
    output[i] = new float[floats_in_chunk];
    output_cl[i] = cl::Buffer((cl_mem_flags)(CL_MEM_WRITE_ONLY), CHUNK_SIZE_B);
    kernel.setArg(arg_n++, output_cl[i]);

    std::string binary_name(func_name);
    binary_name += "_output_" + to_string(i) + ".ref";

    ref_files[i].exceptions(fstream::badbit);
    if (verify)
      ref_files[i].open(binary_name, ios_base::in | ios_base::binary);
    else
      ref_files[i].open(binary_name,
                        ios_base::out | ios_base::binary | ios_base::trunc);

    ref_inputs[i] = new float[floats_in_chunk];
  }

  /*
    // This triggers compilation of dynamic WG binaries.
    cl::Program::Binaries binaries{};
    int err = program.getInfo<>(CL_PROGRAM_BINARIES, &binaries);
    assert(err == CL_SUCCESS);
  */

  std::mt19937 gen(SEED);
  auto rnd3 =
      std::bind(std::uniform_real_distribution<float>{0.0f, 20.0f}, gen);

  /****************************************************************/

  bool all_ok = true;

  for (unsigned I = 0; I < ITERS; ++I) {

    for (unsigned i = 0; i < num_inputs; ++i) {

      // gen input
      for (unsigned k = 0; k < floats_in_chunk; ++k)
        input[i][k] = rnd3();

      // write input
      queue.enqueueWriteBuffer(input_cl[i],
                               CL_FALSE, // block
                               0, CHUNK_SIZE_B, input[i]);
    }

    for (unsigned i = 0; i < num_outputs; ++i) {

      // gen input
      for (unsigned k = 0; k < floats_in_chunk; ++k) {
        output[i][k] = 0.0f;
        ref_inputs[i][k] = 0.0f;
      }
    }

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(vecs_in_chunk), cl::NullRange);

    for (unsigned i = 0; i < num_outputs; ++i) {
      queue.enqueueReadBuffer(output_cl[i],
                              CL_FALSE, // block
                              0, CHUNK_SIZE_B, output[i]);
    }

    queue.finish();

    //    std::cerr << "INPUTS[2] = " << input[0][2] <<  "  OUTPUTS[2] = " <<
    //    output[0][2] << "  REF_INPTU[2] = " << ref_inputs[0][2] << "\n";

    //    std::cerr << "INPUTS[2] = " << input[0][2] <<  "  OUTPUTS[2] = " <<
    //    output[0][2] << "\n";

    /* read input for compare / write output for gen */
    for (unsigned i = 0; i < num_outputs; ++i) {
      if (verify) {
        ref_files[i].read((char *)ref_inputs[i], CHUNK_SIZE_B);
      } else {
        ref_files[i].write((const char *)output[i], CHUNK_SIZE_B);
      }
    }

    std::cerr << "INPUTS[2] = " << input[0][2]
              << "  OUTPUTS[2] = " << output[0][2]
              << "  REF_INPTU[2] = " << ref_inputs[0][2] << "\n";

    // compare
    if (verify) {
      all_ok =
          compare_floats(num_inputs, input, num_outputs, ref_inputs, output);
      if (!all_ok)
        break;
      //        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  /***************************************************************/

  for (unsigned i = 0; i < num_inputs; ++i)
    delete[] input[i];

  for (unsigned i = 0; i < num_outputs; ++i) {
    delete[] output[i];

    delete[] ref_inputs[i];

    ref_files[i].close();
  }

  return all_ok ? 0 : 3;
}

/*

TODO combinations of

0
1
nan
inf
+ -

*/

int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cout << "USAGE: $0 -g/-v VECSIZE FUNC_NAME\n";
    return 1;
  }

  try {

    cl::Device device = cl::Device::getDefault();

    /*
        cl_ulong max_alloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        if (max_alloc < CHUNK_SIZE_B) {
          std::cerr << "Device doesn't have enough global memory!\n";
          return 1;
        }
    */

    bool has_compiler =
        (device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>() != CL_FALSE);
    std::cerr << "HAS COMPILER: " << has_compiler << "\n";

    std::string gen_or_verify(argv[1]);
    std::string vecsize_str(argv[2]);
    unsigned vecsize = std::stoi(argv[2]);
    std::string func_name(argv[3]);

    if (gen_or_verify == "-g")
      return generate_or_verify(false, func_name, vecsize, has_compiler, 1, 1);

    if (gen_or_verify == "-v")
      return generate_or_verify(true, func_name, vecsize, has_compiler, 1, 1);

    if (gen_or_verify == "-c")
      return compile_only(func_name, vecsize, has_compiler, 1, 1);

    return -1;

  } catch (const ifstream::failure &e) {
    std::cerr << "IO error: " << e.what();
  } catch (const cl::Error &e) {
    std::cerr << "OpenCL error: " << e.what();
  } catch (const std::runtime_error &e) {
    std::cerr << "Runtime error: " << e.what();
  }

  return 0;
}
