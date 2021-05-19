library work;
use work.tta_core_imem_mau.all;

package tta_core_globals is
  -- address width of the instruction memory
  constant IMEMADDRWIDTH : positive := 12;
  -- width of the instruction memory in MAUs
  constant IMEMWIDTHINMAUS : positive := 1;
  -- width of instruction fetch block.
  constant IMEMDATAWIDTH : positive := IMEMWIDTHINMAUS*IMEMMAUWIDTH;
  -- clock period
  constant PERIOD : time := 10 ns;
  -- number of busses.
  constant BUSTRACE_WIDTH : positive := 32;
  -- number of cores
  constant CORECOUNT : positive := 1;
  -- instruction width
  constant INSTRUCTIONWIDTH : positive := 42;
end tta_core_globals;
