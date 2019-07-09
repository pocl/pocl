-- Copyright (c) 2017 Stephan Nolting / IMS, Leibniz Univ. Hannover
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
--
-------------------------------------------------------------------------------
-- Title      : Generic shift and round unit
-------------------------------------------------------------------------------
-- File       : generic_sru.vhd
-- Author     : Stephan Nolting
-- Company    : Leibniz Univ. Hannover
-- Created    : 2018-02-06
-- Last update: 2018-06-18
-------------------------------------------------------------------------------
-- Description: Generic shift unit providing logical left shift, logical right
-- shift & arithmetical right shift. Up to 4 pipeline registers can be inserted
-- into the data path. See the according generic's comments for further
-- information. The more pipeline registers are activated, the more the unit's
-- latency is increased.
--
-- If you use this design in your work, please cite the following publication:
-- Payá-Vayá, Guillermo, Roman Burg, and Holger Blume.
-- "Dynamic data-path self-reconfiguration of a VLIW-SIMD soft-processor
--  architecture."
--  Workshop on Self-Awareness in Reconfigurable Computing Systems, SRCS. 2012.
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2018-02-06  1.0      nolting Created
-- 2018-06-18  1.1      tervoa  Added MIT License, reformatted header
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library UNISIM;
use UNISIM.vcomponents.all;

entity generic_sru is
  generic (
    DATA_WIDTH   : natural := 32; -- data width (power of 2)
    EN_ROUNDING  : boolean := false; -- enable hw for rounding to zero/infinity
    -- pipeline stages --
    EN_INPUT_REG : boolean := false; -- enable input registers
    EN_SHIFT_REG : boolean := false; -- enable shifter output register
    EN_ROUND_REG : boolean := false; -- enable rounding unit shift register
    EN_OUT_REG   : boolean := false  -- enable output register
  );
  port (
    -- global control --
    clk           : in  std_logic;
    -- operand data --
    opa_i         : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    opb_i         : in  std_logic_vector(DATA_WIDTH-1 downto 0);
    -- operation control --
    shift_dir_i   : in  std_logic; -- 0: right, 1: left (shift dreiction)
    arith_shift_i : in  std_logic; -- 0: logical, 1: arithmetical (only for right shifts)
    rnd_en_i      : in  std_logic; -- 0: rounding disabled, 1: rounding enabled
    rnd_mode_i    : in  std_logic; -- 0: floor, 1: infinity (type of rounding)
    -- operation result --
    data_o        : out std_logic_vector(DATA_WIDTH-1 downto 0)
  );
end generic_sru;

architecture generic_sru_xv6_rtl of generic_sru is

  -- muxcy xilinx primitive component (carry chain multiplexer) --
  component muxcy
    port (
      o  : out std_logic;
      ci : in  std_logic;
      di : in  std_logic;
      s  : in  std_logic
    );
  end component;

  -- xorcy xilinx primitive component (carry chain 'adder') --
  component xorcy
    port (
      o  : out std_logic;
      ci : in  std_logic;
      li : in  std_logic
    );
  end component;

  -- local types --
  type smask_array_t is array(0 to DATA_WIDTH-1) of std_logic_vector(DATA_WIDTH-1 downto 0);

  -- Function: Minimum required bit width --
  function index_size(input : natural) return natural is
  begin
    for i in 0 to natural'high loop
      if (2**i >= input) then
        return i;
      end if;
    end loop; -- i
    return 0;
  end function index_size;

  -- Function: init mask for shifter's sign bit cancellation mask --
  function init_smask(n: natural) return smask_array_t is
    variable smask_array_v : smask_array_t;
  begin
    smask_array_v := (others => (others => '0'));
    smask_array_v(0) := (others => '1');
    for i in 0 to n-2 loop
      smask_array_v(i+1) := '0' & (smask_array_v(i)(n-1 downto 1));
    end loop; -- i
    return smask_array_v;
  end function init_smask;

  -- Function: Bit reversal --
  function bit_reversal(input : std_logic_vector) return std_logic_vector is
    variable output_v : std_logic_vector(input'range);
  begin
    for i in 0 to input'length-1 loop
      output_v(input'length-i-1) := input(i);
    end loop; -- i
    return output_v;
  end function bit_reversal;

  -- internal configuration --
  constant log2_data_width_c : natural := index_size(DATA_WIDTH);
  constant subword_c         : natural := 0;

  -- pipeline stage 0 (input register) --
  signal opa_s0, opa_ff0                 : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal opb_s0, opb_ff0                 : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal shift_dir_s0, shift_dir_ff0     : std_logic;
  signal shift_arith_s0, shift_arith_ff0 : std_logic;
  signal rnd_en_s0, rnd_en_ff0           : std_logic;
  signal rnd_mode_s0, rnd_mode_ff0       : std_logic;

  -- shifter core --
  constant smask_array : smask_array_t := init_smask(DATA_WIDTH);
  signal sra_data  : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal sra_mask  : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal shift_in  : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal shift_res : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal carry_sel : std_logic;

  -- pipeline stage 1 --
  signal sra_data_s1, sra_data_ff1       : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal sra_mask_s1, sra_mask_ff1       : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal carry_s1, carry_ff1             : std_logic;
  signal shift_dir_s1, shift_dir_ff1     : std_logic;
  signal shift_arith_s1, shift_arith_ff1 : std_logic;
  signal rnd_en_s1, rnd_en_ff1           : std_logic;
  signal rnd_mode_s1, rnd_mode_ff1       : std_logic;

  -- rounding unit --
  signal inc_chain  : std_logic_vector(DATA_WIDTH downto 0);
  signal inc_data   : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal inc_result : std_logic_vector(DATA_WIDTH-1 downto 0);

  -- pipeline stage 2 --
  signal inc_res_s2, inc_res_ff2 : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal carry_s2, carry_ff2     : std_logic;

  -- zero detector --
  signal nibble_is_zero : std_logic_vector((DATA_WIDTH/4)-1 downto 0);
  signal zero_chain     : std_logic_vector((DATA_WIDTH/4) downto 0);

  -- pipeline stage 3: output register --
  signal inc_res_s3, inc_res_ff3 : std_logic_vector(DATA_WIDTH-1 downto 0);
  signal carry_s3, carry_ff3     : std_logic;
  signal zero_s3, zero_ff3       : std_logic;

  --     zero      overflow  carry     negative
  signal z_flag_o, o_flag_o, c_flag_o, n_flag_o : std_logic;

begin

  -- Pipeline Stage 0: Input Register -------------------------------------------------------
  -- -------------------------------------------------------------------------------------------
  pipe_s0: process(clk)
  begin
    if rising_edge(clk) then
      opa_ff0 <= opa_i;
      opb_ff0 <= opb_i;
      shift_dir_ff0   <= shift_dir_i;
      shift_arith_ff0 <= arith_shift_i;
      rnd_en_ff0      <= rnd_en_i;
      rnd_mode_ff0    <= rnd_mode_i;
    end if;
  end process pipe_s0;

  -- use input registers? --
  opa_s0 <= opa_ff0 when (EN_INPUT_REG = true) else opa_i;
  opb_s0 <= opb_ff0 when (EN_INPUT_REG = true) else opb_i;
  shift_dir_s0   <= shift_dir_ff0 when (EN_INPUT_REG = true) else shift_dir_i;
  shift_arith_s0 <= shift_arith_ff0 when (EN_INPUT_REG = true) else arith_shift_i;
  rnd_en_s0      <= rnd_en_ff0 when (EN_INPUT_REG = true) else rnd_en_i;
  rnd_mode_s0    <= rnd_mode_ff0 when (EN_INPUT_REG = true) else rnd_mode_i;


  -- Shifter Core ---------------------------------------------------------------------------
  -- -------------------------------------------------------------------------------------------
  shift_in <= bit_reversal(opa_s0) when (shift_dir_s0 = '1') else opa_s0; -- reverse bits if left shift

  shifter_core: process(opb_s0, shift_in)
    variable carry_sel_v : std_logic_vector(DATA_WIDTH-1 downto 0);
    variable positions_v : integer;
  begin
    -- all shift types are based on a single arithmetical right shifter
    positions_v := to_integer(unsigned(opb_s0(log2_data_width_c-1 downto 0)));
    sra_data <= std_logic_vector(shift_right(signed(shift_in), positions_v));
    sra_mask <= smask_array(positions_v);
    -- select carry --
    carry_sel_v := shift_in(DATA_WIDTH-2 downto 0) & '0';
    carry_sel <= carry_sel_v(positions_v);
  end process shifter_core;


  -- Pipeline Stage 1: Shifter output register ----------------------------------------------
  -- -------------------------------------------------------------------------------------------
  pipe_s1: process(clk)
  begin
    if rising_edge(clk) then
      sra_data_ff1    <= sra_data;
      sra_mask_ff1    <= sra_mask;
      carry_ff1       <= carry_sel;
      shift_dir_ff1   <= shift_dir_s0;
      shift_arith_ff1 <= shift_arith_s0;
      rnd_en_ff1      <= rnd_en_s0;
      rnd_mode_ff1    <= rnd_mode_s0;
    end if;
  end process pipe_s1;

  -- use pipeline 1 registers? --
  sra_data_s1    <= sra_data_ff1 when (EN_SHIFT_REG = true) else sra_data;
  sra_mask_s1    <= sra_mask_ff1 when (EN_SHIFT_REG = true) else sra_mask;
  carry_s1       <= carry_ff1 when (EN_SHIFT_REG = true) else carry_sel;
  shift_dir_s1   <= shift_dir_ff1 when (EN_SHIFT_REG = true) else shift_dir_s0;
  shift_arith_s1 <= shift_arith_ff1 when (EN_SHIFT_REG = true) else shift_arith_s0;
  rnd_en_s1      <= rnd_en_ff1 when (EN_SHIFT_REG = true) else rnd_en_s0;
  rnd_mode_s1    <= rnd_mode_ff1 when (EN_SHIFT_REG = true) else rnd_mode_s0;


  -- Shifter result masking ------------------------------------------------------------------
  -- --------------------------------------------------------------------------------------------
  shifter_sel: process(shift_dir_s1, shift_arith_s1, sra_data_s1, sra_mask_s1)
    variable lrs_v : std_logic_vector(DATA_WIDTH-1 downto 0);
  begin
    lrs_v := sra_data_s1 and sra_mask_s1;
    if (shift_dir_s1 = '1') then -- logical left shift
      shift_res <= bit_reversal(lrs_v);
    else -- right shift
      if (shift_arith_s1 = '1') then -- arithmetical right shift
        shift_res <= sra_data_s1;
      else -- logical right shift
        shift_res <= lrs_v;
      end if;
    end if;
  end process shifter_sel;


  -- Rounding --------------------------------------------------------------------------------
  -- --------------------------------------------------------------------------------------------
  -- start of incrementer carry line with internal carry input
  inc_chain(0) <= carry_s1 and (rnd_en_s1 and rnd_mode_s1);

  -- simple incrementer, using dedicated hardware (muxcy, xorcy)
  increment_unit:
  for i in 0 to DATA_WIDTH-1 generate

    inc_muxcy_inst: muxcy
    port map (
      o  => inc_chain(i+1),
      ci => inc_chain(i),
      di => '0',
      s  => shift_res(i)
    );
    inc_xorcy_inst: xorcy
    port map (
      o  => inc_data(i),
      ci => inc_chain(i),
      li => shift_res(i)
    );
  end generate; -- i

  -- operation result output --
  inc_result <= inc_data when (EN_ROUNDING = true) else shift_res;


  -- Pipeline Stage 2: Rounding unit output register ----------------------------------------
  -- -------------------------------------------------------------------------------------------
  pipe_s2: process(clk)
  begin
    if rising_edge(clk) then
      inc_res_ff2 <= inc_result;
      carry_ff2   <= carry_s1;
    end if;
  end process pipe_s2;

  -- use pipeline 2 registers? --
  inc_res_s2 <= inc_res_ff2 when (EN_ROUND_REG = true) else inc_result;
  carry_s2   <= carry_ff2 when (EN_ROUND_REG = true) else carry_s1;


  -- Zero Detector --------------------------------------------------------------------------
  -- -------------------------------------------------------------------------------------------
  zero_chain(0) <= '1'; -- start a new subword chain

  zero_detector:
  for i in 0 to (DATA_WIDTH/4)-1 generate -- number of nibbles
    -- zero detection for 4 bit -> 1 LUT + 1 MUXCY, propagate previous zero signal, when nibble is zero
    nibble_is_zero(i) <= '1' when (inc_res_s2(i*4+3 downto i*4) = "0000") else '0'; -- is zero?

    zero_detector_muxcy: muxcy
    port map (
      o  => zero_chain(i+1),  -- chain output signal
      ci => zero_chain(i),    -- s=1: chain input signal
      di => '0',              -- s=0: 0 input, nibble is not zero
      s  => nibble_is_zero(i) -- mux select input
    );
  end generate; -- i


  -- Pipeline Stage 3: Output register ------------------------------------------------------
  -- -------------------------------------------------------------------------------------------
  pipe_s3: process(clk)
  begin
    if rising_edge(clk) then
      inc_res_ff3 <= inc_res_s2;
      carry_ff3   <= carry_s2;
      zero_ff3    <= zero_chain(DATA_WIDTH/4);
    end if;
  end process pipe_s3;

  -- use pipeline 2 registers? --
  inc_res_s3 <= inc_res_ff3 when (EN_OUT_REG = true) else inc_res_s2;
  carry_s3   <= carry_ff3 when (EN_OUT_REG = true) else carry_s2;
  zero_s3    <= zero_ff3 when (EN_OUT_REG = true) else zero_chain(DATA_WIDTH/4);


  -- data output --
  data_o <= inc_res_s3;

  -- negative flag output --
  n_flag_o <= inc_res_s3(DATA_WIDTH-1);

  -- zero flag output --
  z_flag_o <= zero_s3;

  -- carry flag output --
  c_flag_o <= carry_s3;

-- TODO: overflow:
-- all out-shifted bits should be equal to the sign, else overflow
-- for signed operations, also the result sign should be equal to the original sign
  o_flag_o <= '0'; -- implement me!


end generic_sru_xv6_rtl;
