-- Copyright (c) 2002-2009 Tampere University.
--
-- This file is part of TTA-Based Codesign Environment (TCE).
--
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
--
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.

library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.numeric_std.all;
use work.tta_core_globals.all;
use work.tta_core_gcu_opcodes.all;
use work.tta_core_imem_mau.all;

use work.tce_util.all;

entity tta_core_ifetch is

  generic (
    no_glock_loopback_g        : std_logic := '0';
    bypass_fetchblock_register : boolean   := false;
    bypass_pc_register         : boolean   := false;
    bypass_decoder_registers   : boolean   := false;
    extra_fetch_cycles         : integer   := 0;
    sync_reset_g               : boolean   := false;
    debug_logic_g              : boolean   := false;
    enable_loop_buffer_g       : boolean   := false;
    enable_infloop_buffer_g    : boolean   := false;
    enable_irf_g               : boolean   := false;
    irf_size_g                 : integer   := 0;

    pc_init_g : std_logic_vector(IMEMADDRWIDTH-1 downto 0) := (others => '0'));

  port (
    -- program counter in
    pc_in      : in  std_logic_vector (IMEMADDRWIDTH-1 downto 0);
    --return address out
    ra_out     : out std_logic_vector (IMEMADDRWIDTH-1 downto 0);
    -- return address in
    ra_in      : in  std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    -- ifetch control signals
    pc_load    : in  std_logic;
    ra_load    : in  std_logic;
    pc_opcode  : in  std_logic_vector(0 downto 0);
    --instruction memory interface
    imem_data  : in  std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
    imem_addr  : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    imem_en_x  : out std_logic;
    fetchblock : out std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
    busy       : in  std_logic;

    -- global lock
    glock : out std_logic;

    -- external control interface
    fetch_en  : in std_logic;             --fetch_enable
    
        -- debugger signals
            db_lockreq  : in std_logic;
            db_rstx     : in std_logic;
            db_pc       : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
            db_cyclecnt : out std_logic_vector(64-1 downto 0);
            db_lockcnt  : out std_logic_vector(64-1 downto 0);
    
    

    clk  : in std_logic;
    rstx : in std_logic);
end tta_core_ifetch;

architecture rtl_andor of tta_core_ifetch is

  -- signals for program counter.
  signal pc_reg      : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal pc_wire     : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal pc_prev_reg : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal next_pc     : std_logic_vector(IMEMADDRWIDTH-1 downto 0);

  signal increased_pc    : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal return_addr_reg : std_logic_vector(IMEMADDRWIDTH-1 downto 0);

  -- internal signals for initializing and locking execution.
  signal lock          : std_logic;
  signal mem_en_lock_r : std_logic;

  -- Delay/latency from retrieving instruction block from instruction memory.
  constant IFETCH_DELAY : integer := 1 + extra_fetch_cycles;
  -- Delay/latency from pc register to dispatching instruction.
  constant PC_TO_DISPATCH_DELAY : integer :=
    to_int(not bypass_fetchblock_register) +
    IFETCH_DELAY;
  -- Delay/latency from control flow operation to dispatching instruction.
  constant NEXT_TO_DISPATCH_DELAY : integer :=
    PC_TO_DISPATCH_DELAY + to_int(not bypass_pc_register);

  signal   reset_cntr   : integer range 0 to IFETCH_DELAY;
  signal   reset_lock   : std_logic;

  -- Loopbuffer signals, or placeholders if lb is not enabled
  -- Placeholder signals for loop buffer ports/constants
  constant LBUFMAXITER  : integer := 1;
  constant LBUFMAXDEPTH : integer := 1;
  constant IFE_LBUFS    : integer := 1;
  constant IFE_INFLOOP  : integer := 1;
  
  signal o1data : std_logic_vector(LBUFMAXITER-1 downto 0);
  signal o1load : std_logic;
  signal loop_start_out : std_logic;
  signal loop_len_out   : std_logic_vector(bit_width(LBUFMAXDEPTH+1)-1 downto 0);
  signal loop_iter_out  : std_logic_vector(LBUFMAXITER-1 downto 0);
  
  signal iteration_count    : std_logic_vector(LBUFMAXITER-1 downto 0);
  signal pc_after_loop      : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal lockcnt_r, cyclecnt_r   : unsigned(64 - 1 downto 0);
  signal db_pc_next       : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  constant db_pc_start : std_logic_vector(IMEMADDRWIDTH-1 downto 0)
                         := (others => '0');
  
  
  

begin

  -- enable instruction memory.
  imem_en_x <= '0'    when (fetch_en = '1' and mem_en_lock_r = '0') else '1';
  -- do not fetch new instruction when processor is locked.
  imem_addr <= pc_wire;

  -- propagate lock to global lock

  glock  <= busy or reset_lock or (not (fetch_en or no_glock_loopback_g));
  ra_out <= return_addr_reg;
  lock   <= not fetch_en or busy or mem_en_lock_r;


  pc_update_generate_0  :  if not enable_irf_g generate
    pc_update_proc : process (clk, rstx)
    begin
      if not sync_reset_g and rstx = '0' then
        pc_reg      <= pc_init_g;
        pc_prev_reg <= (others => '0');
      elsif clk'event and clk = '1' then    -- rising clock edge.
        if (sync_reset_g and rstx = '0') or db_rstx = '0' then
          pc_reg      <= db_pc_start;
          pc_prev_reg <= (others => '0');
        elsif lock = '0' then
          pc_reg      <= next_pc;
          if bypass_pc_register and bypass_fetchblock_register
            and bypass_decoder_registers and pc_load = '1' then
            pc_prev_reg <= pc_in;
          else
            pc_prev_reg <= pc_reg;
          end if;
        end if;
      end if;
    end process pc_update_proc;
  end generate pc_update_generate_0;

  


  -----------------------------------------------------------------------------
  ra_block : block
    signal ra_source : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  begin  -- block ra_block

    -- Default choice generate
    ra_source_select_generate_0 : if not enable_irf_g and not bypass_pc_register generate
      ra_source <= increased_pc;
    end generate ra_source_select_generate_0;

    -- Choice enabled by generic
    ra_source_select_generate_1 : if not enable_irf_g and bypass_pc_register generate
      ra_source <= pc_reg;
    end generate ra_source_select_generate_1;

    -- When using IRF
    ra_source_select_generate_2 : if enable_irf_g generate
      ra_source <= pc_prev_reg;
    end generate ra_source_select_generate_2;

    ra_update_proc : process (clk, rstx)
    begin  -- process ra_update_proc
      if not sync_reset_g and rstx = '0' then -- asynchronous reset (active low)
        return_addr_reg <= (others => '0');
      elsif clk'event and clk = '1' then  -- rising clock edge
        if (sync_reset_g and rstx = '0') or db_rstx = '0' then
          return_addr_reg <= (others => '0');
        elsif lock = '0' then
          -- return address
          if (ra_load = '1') then
            return_addr_reg <= ra_in;
          elsif (pc_load = '1' and unsigned(pc_opcode) = IFE_CALL) then
            -- return address transformed to same form as all others addresses
            -- provided as input
            return_addr_reg <= ra_source;
          end if;

        end if;
      end if;
    end process ra_update_proc;
  end block ra_block;

  -----------------------------------------------------------------------------
  -- Keeps memory enable inactive during reset
  imem_lock_proc : process (clk, rstx)
  begin
    if not sync_reset_g and rstx = '0' then
      mem_en_lock_r <= '1';
    elsif clk'event and clk = '1' then  -- rising clock edge
        if (sync_reset_g and rstx = '0') or db_rstx = '0' then
        mem_en_lock_r <= '1';
      else
        mem_en_lock_r <= '0';
      end if;
    end if;
  end process imem_lock_proc;

  -----------------------------------------------------------------------------
  -- Default fetch implementation
  fetch_block_registered_generate : if
    not bypass_fetchblock_register generate
    fetch_block : block
      signal instruction_reg : std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH*
                                                (extra_fetch_cycles+1)-1 downto 0);
    begin  -- block fetch_block

      fetch_block_proc : process (clk, rstx)
      begin  -- process fetch_block_proc
        if not sync_reset_g and rstx = '0' then   -- asynchronous reset (active low)
          instruction_reg <= (others => '0');
          reset_cntr      <= 0;
          reset_lock      <= '1';
        elsif clk'event and clk = '1' then  -- rising clock edge
          if (sync_reset_g and rstx = '0') or db_rstx = '0' then
            instruction_reg <= (others => '0');
            reset_cntr      <= 0;
            reset_lock      <= '1';
          elsif lock = '0' then
            if reset_cntr < IFETCH_DELAY then
              reset_cntr <= reset_cntr + 1;
            else
              reset_lock <= '0';
            end if;
            if (extra_fetch_cycles > 0) then
              instruction_reg(instruction_reg'length-fetchblock'length-1 downto 0)
                   <= instruction_reg(instruction_reg'length-1 downto fetchblock'length);
            end if;
            instruction_reg(instruction_reg'length-1
                            downto instruction_reg'length - fetchblock'length)
            <= imem_data;
            
          end if;
        end if;
      end process fetch_block_proc;
      fetchblock <= instruction_reg(fetchblock'length-1 downto 0);
    end block fetch_block;
  end generate fetch_block_registered_generate;

  -- Fetch implementation without fetch register.
  fetch_block_bypassed_generate : if
    not (not bypass_fetchblock_register) generate
    fetch_block : block
    begin  -- block fetch_block
      fetch_block_proc : process (clk, rstx)
      begin  -- process fetch_block_proc
        if not sync_reset_g and rstx = '0' then -- asynchronous reset (active low)
          reset_lock <= '1';
        elsif clk'event and clk = '1' then  -- rising clock edge
          if (sync_reset_g and rstx = '0') or db_rstx = '0' then
            reset_lock <= '1';
          elsif lock = '0' then
            reset_lock <= '0';
          end if;
        end if;
      end process fetch_block_proc;
      fetchblock <= imem_data;
    end block fetch_block;

  end generate fetch_block_bypassed_generate;

  -----------------------------------------------------------------------------

  loopbuf_logic : if enable_loop_buffer_g generate
    -- Loop buffer signals --
    signal start_looping      : std_logic;
    signal start_looping_r    : std_logic_vector(NEXT_TO_DISPATCH_DELAY-1
                                                 downto 0);
    signal loop_length, loop_length_reg
            : std_logic_vector(bit_width(LBUFMAXDEPTH+1)-1 downto 0);
    signal loop_iter_reg      : std_logic_vector(LBUFMAXITER-1 downto 0);
    signal loop_iter_temp_reg : std_logic_vector(LBUFMAXITER-1 downto 0);
  begin
    assert not enable_irf_g
      report "IRF is not supported with loop buffer!"
      severity failure;

    -- Loop buffer setup operation logic --
    start_looping <= '1' when (pc_load = '1' and
                               unsigned(pc_opcode) = IFE_LBUFS) else
                     '0';

    iteration_count <= o1data(LBUFMAXITER-1 downto 0)
                       when o1load = '1' else
                       loop_iter_temp_reg;
    loop_length <= pc_in(bit_width(LBUFMAXDEPTH+1)-1 downto 0);

    process (clk, rstx)
    begin
      if not sync_reset_g and rstx = '0' then
        start_looping_r    <= (others => '0');
        loop_length_reg    <= (others => '0');
        loop_iter_reg      <= (others => '0');
        loop_iter_temp_reg <= (others => '0');
      elsif clk'event and clk = '1' then    -- rising clock edge
        -- Loop buffer control --
        if (sync_reset_g and rstx = '0') or db_rstx = '0' then
          start_looping_r    <= (others => '0');
          loop_length_reg    <= (others => '0');
          loop_iter_reg      <= (others => '0');
          loop_iter_temp_reg <= (others => '0');
        elsif lock = '0' then
          if (start_looping = '1' and
              unsigned(iteration_count) /= 0) then
            loop_length_reg    <= loop_length;
            loop_iter_reg      <= iteration_count;
            start_looping_r(0) <= '1';
          else
            start_looping_r(0) <= '0';
          end if;

          if o1load = '1' then
            loop_iter_temp_reg <= o1data(LBUFMAXITER-1 downto 0);
          end if;
          -- Delay slots for lbufs are introduced to avoid need of pipeline
          -- flushing in case the loop is skipped with iteration count of zero.
          start_looping_r(start_looping_r'left downto 1) <=
            start_looping_r(start_looping_r'left-1 downto 0);
        end if;
      end if;
    end process;

    loop_start_out <= start_looping_r(start_looping_r'left);
    loop_iter_out  <= loop_iter_reg;
    loop_len_out   <= loop_length_reg;
    pc_after_loop <= std_logic_vector(
      unsigned(increased_pc) + unsigned(loop_length));
  end generate;

  infloop_logic : if enable_infloop_buffer_g generate
    signal start_looping : std_logic;
    signal start_looping_r
                : std_logic_vector(NEXT_TO_DISPATCH_DELAY-1 downto 0);
    signal loop_length, loop_length_reg
                : std_logic_vector(bit_width(LBUFMAXDEPTH+1)-1 downto 0);
  begin
    -- infinity loop operation control logic --
    start_looping <= '1' when (pc_load = '1' and
                               unsigned(pc_opcode) = IFE_INFLOOP) else
                     '0';
    loop_length <= pc_in(bit_width(LBUFMAXDEPTH+1)-1 downto 0);

    process (clk, rstx)
    begin
      if not sync_reset_g and rstx = '0' then
        start_looping_r    <= (others => '0');
        loop_length_reg    <= (others => '0');
      elsif clk'event and clk = '1' then    -- rising clock edge
        -- Loop buffer control --
        if sync_reset_g and rstx = '0' then
          start_looping_r    <= (others => '0');
          loop_length_reg    <= (others => '0');
        elsif lock = '0' then
          if (start_looping = '1' and to_uint(loop_length) /= 0) then
            assert to_uint(loop_length) <= LBUFMAXDEPTH
              report "The loop body size exceeds loop buffer capacity!"
              severity failure;
            loop_length_reg    <= loop_length;
            start_looping_r(0) <= '1';
          else
            start_looping_r(0) <= '0';
          end if;

          -- Delay slots for lbufs are introduced to avoid need of pipeline
          -- flushing in case the loop is skipped with iteration count of
          -- zero.
          start_looping_r(start_looping_r'left downto 1) <=
            start_looping_r(start_looping_r'left-1 downto 0);
        end if;
      end if;
    end process;

    loop_start_out <= start_looping_r(start_looping_r'left);
    loop_len_out   <= loop_length_reg;


    

  end generate infloop_logic;
--------------------------------------------------------------------------------
  

  
  
  

  

  

  default_pc_generate: if not bypass_pc_register  generate
    pc_wire <= pc_reg when (lock = '0') else pc_prev_reg;
    -- increase program counter
    increased_pc <= std_logic_vector(unsigned(pc_wire) + IMEMWIDTHINMAUS);

    sel_next_pc : process (pc_load, pc_in, increased_pc, pc_opcode)
    begin
      if pc_load = '1' and (unsigned(pc_opcode) = IFE_CALL or unsigned(pc_opcode) = IFE_JUMP) then
        next_pc <= pc_in;
        
      else -- no branch
        next_pc <= increased_pc;
      end if;
    end process sel_next_pc;
  end generate default_pc_generate;

  bypass_pc_register_generate: if bypass_pc_register generate
    -- increase program counter
    increased_pc <= std_logic_vector(unsigned(pc_wire) + IMEMWIDTHINMAUS);

    sel_next_pc : process (pc_in, pc_reg, increased_pc        ,
     pc_load, pc_opcode)
    begin
      if pc_load = '1' and (unsigned(pc_opcode) = IFE_CALL or unsigned(pc_opcode) = IFE_JUMP) then
        pc_wire <= pc_in;
        next_pc      <= increased_pc;
      
      else -- no branch
        pc_wire <= pc_reg;
        next_pc      <= increased_pc;
      end if;
    end process sel_next_pc;

  end generate bypass_pc_register_generate;

  -----------------------------------------------------------------------------
  debug_counters : if debug_logic_g generate
  -----------------------------------------------------------------------------
  -- Debugger processes and signal assignments
  -----------------------------------------------------------------------------
    db_counters : process(clk, rstx)
    begin
      if not sync_reset_g and rstx = '0' then -- async reset (active low)
        lockcnt_r  <= (others => '0');
        cyclecnt_r <= (others => '0');
      elsif rising_edge(clk) then
        if (sync_reset_g and rstx = '0') or db_rstx = '0' then
          lockcnt_r  <= (others => '0');
          cyclecnt_r <= (others => '0');
        elsif db_lockreq = '0' then
          if lock = '1' then
            lockcnt_r  <= lockcnt_r  + 1;
          else
            cyclecnt_r <= cyclecnt_r + 1;
          end if;
        end if;
      end if;
    end process;

    db_cyclecnt <= std_logic_vector(cyclecnt_r);
    db_lockcnt  <= std_logic_vector(lockcnt_r);
    db_pc       <= pc_reg;
    db_pc_next  <= next_pc;
  end generate debug_counters;

  

end rtl_andor;

