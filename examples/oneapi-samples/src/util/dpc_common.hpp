#include <chrono>

namespace dpc_common {
  class TimeInterval {
    std::chrono::steady_clock Clk;
    std::chrono::time_point<std::chrono::steady_clock> Start;
  public:
    TimeInterval() {
      Start = std::chrono::steady_clock::now();
    }
    double Elapsed() {
      auto End = std::chrono::steady_clock::now();
      std::chrono::duration<double> Diff = End - Start;
      return Diff.count();
    }
  };
}
