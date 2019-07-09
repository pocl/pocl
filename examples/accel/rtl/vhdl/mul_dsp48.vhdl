-- Copyright (c) 2019 Tampere University.
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
-------------------------------------------------------------------------------
-- Title      : MUL unit for Xilinx devices
-------------------------------------------------------------------------------
-- File       : mul_dsp48.vhdl
-- Author     : Kati Tervo
-- Company    :
-- Created    : 2017-09-13
-- Last update: 2017-09-13
-- Platform   :
-------------------------------------------------------------------------------
-- Description: 32x32 multiplier for Xilinx FPGAs
--
-- Revisions  :
-- Date        Version  Author  Description
-- 2019-03-12  1.0      katte  Created
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

Library UNISIM;
use UNISIM.vcomponents.all;

entity mul_dsp48 is
  generic (
    latency_g : integer
  ); port (
    clk           : in std_logic;
    rstx          : in std_logic;
    glock_in      : in std_logic;
    load_in       : in std_logic;

    operand_a_in  : in std_logic_vector(32-1 downto 0);
    operand_b_in  : in std_logic_vector(32-1 downto 0);
    operand_c_in  : in std_logic_vector(32-1 downto 0);
    result_out    : out std_logic_vector(32-1 downto 0)
  );
end mul_dsp48;

architecture rtl of mul_dsp48 is

  signal A_full, A_half_0, A_half_1 : std_logic_vector(30-1 downto 0);
  signal B_full, B_half_0, B_half_1 : std_logic_vector(18-1 downto 0);
  signal P_full, P_half_0, P_half_1 : std_logic_vector(48-1 downto 0);
  signal C_full                     : std_logic_vector(48-1 downto 0);
  signal full_out, half_out_0, half_out_1 : std_logic_vector(32-1 downto 0);

  signal result, result_r : std_logic_vector(32-1 downto 0);
  attribute use_dsp : string;
  attribute use_dsp of result : signal is "no";

  signal load_r, dsp_enable, output_enable : std_logic;

  function front_latency(total_latency : integer) return integer is
  begin
    if total_latency > 1 then
      return 1;
    else
      return 0;
    end if;
  end function;

  function back_latency(total_latency : integer) return integer is
  begin
    if total_latency > 0 then
      return 1;
    else
      return 0;
    end if;
  end function;
begin

  dsp_enable    <= not glock_in;
  output_enable_no_load_one: if latency_g > 1 generate
    output_enable <= load_r and dsp_enable;
  end generate;

  output_enable_no_load_two: if latency_g <= 1 generate
    output_enable <= load_in and dsp_enable;
  end generate;

  -- Multiplication
  B_full   <= "00" & operand_b_in(15 downto 0);
  A_full   <= "00000000000000" & operand_a_in(15 downto 0);
  C_full   <= X"0000" & operand_c_in;
  B_half_0 <= "00" & operand_b_in(31 downto 16);
  A_half_0 <= "00000000000000" & operand_a_in(15 downto 0);
  B_half_1 <= "00" & operand_a_in(31 downto 16);
  A_half_1 <= "00000000000000" & operand_b_in(15 downto 0);

  full_out   <= P_full(31 downto 0);
  half_out_0 <= P_half_0(15 downto 0) & X"0000";
  half_out_1 <= P_half_1(15 downto 0) & X"0000";

  DSP48E1_full_inst : DSP48E1
  generic map (
     -- Feature Control Attributes: Data Path Selection
     A_INPUT => "DIRECT", -- port A
     B_INPUT => "DIRECT", -- port B
     USE_DPORT => FALSE,                -- Select D port usage (TRUE or FALSE)
     USE_MULT => "DYNAMIC",            -- Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
     USE_SIMD => "ONE48",               -- SIMD selection ("ONE48", "TWO24", "FOUR12")
     -- Pattern Detector Attributes: Pattern Detection Configuration
     AUTORESET_PATDET => "NO_RESET",    -- "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH" 
     MASK => X"3fffffffffff",           -- 48-bit mask value for pattern detect (1=ignore)
     PATTERN => X"000000000000",        -- 48-bit pattern match for pattern detect
     SEL_MASK => "MASK",                -- "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2" 
     SEL_PATTERN => "PATTERN",          -- Select pattern value ("PATTERN" or "C")
     USE_PATTERN_DETECT => "NO_PATDET", -- Enable pattern detect ("PATDET" or "NO_PATDET")
     -- Register Control Attributes: Pipeline Register Configuration
     ACASCREG => 1,                     -- Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
     ADREG => 0,                        -- Number of pipeline stages for pre-adder (0 or 1)
     ALUMODEREG => 1,                   -- Number of pipeline stages for ALUMODE (0 or 1)
     AREG => front_latency(latency_g),  -- Number of pipeline stages for A (0, 1 or 2)
     BCASCREG => 1,                     -- Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
     BREG => front_latency(latency_g),  -- Number of pipeline stages for B (0, 1 or 2)
     CARRYINREG => 1,                   -- Number of pipeline stages for CARRYIN (0 or 1)
     CARRYINSELREG => 1,                -- Number of pipeline stages for CARRYINSEL (0 or 1)
     CREG => 1,                         -- Number of pipeline stages for C (0 or 1)
     DREG => 1,                         -- Number of pipeline stages for D (0 or 1)
     INMODEREG => 1,                    -- Number of pipeline stages for INMODE (0 or 1)
     MREG => 0,                         -- Number of multiplier pipeline stages (0 or 1)
     OPMODEREG => 1,                    -- Number of pipeline stages for OPMODE (0 or 1)
     PREG => back_latency(latency_g)    -- Number of pipeline stages for P (0 or 1)
  )
  port map (
     -- Useless ports, static inputs (no need to reset)
     CLK => clk,
     ACOUT => open, BCOUT => open, CARRYCASCOUT => open, MULTSIGNOUT => open,
     PCOUT => open, OVERFLOW => open, PATTERNBDETECT => open,
     PATTERNDETECT => open, UNDERFLOW => open, CARRYOUT => open,
     ACIN => (others => '0'), BCIN => (others => '0'), CARRYIN => '0',
     CARRYCASCIN => '0', MULTSIGNIN => '0', PCIN => (others => '0'),
     RSTA => '0', RSTALLCARRYIN => '0', RSTALUMODE => '0', RSTB => '0',
     RSTC => '0', RSTCTRL => '0', RSTD => '0', RSTINMODE => '0', RSTM => '0',
     RSTP => '0', CARRYINSEL => "000", ALUMODE => "0000",
     -- Control:
     INMODE => "00000", OPMODE => "0110101",
     -- Data:
     C => C_full, D => (others => '0'),
     A => A_full, B => B_full, P => P_full,
     -- Enable most registers with glock_in
     CEA1 => dsp_enable, CEA2 => dsp_enable, CEAD => dsp_enable,
     CEALUMODE => dsp_enable, CEB1 => dsp_enable, CEB2 => dsp_enable,
     CEC => dsp_enable, CECARRYIN => dsp_enable, CECTRL => dsp_enable,
     CED => dsp_enable, CEINMODE => dsp_enable, CEM => dsp_enable,
     -- enable the output registers when the output needs to be updated
     CEP => output_enable
  );

  DSP48E1_half_0_inst : DSP48E1
  generic map (
     -- Feature Control Attributes: Data Path Selection
     A_INPUT => "DIRECT", -- port A
     B_INPUT => "DIRECT", -- port B
     USE_DPORT => FALSE,                -- Select D port usage (TRUE or FALSE)
     USE_MULT => "DYNAMIC",            -- Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
     USE_SIMD => "ONE48",               -- SIMD selection ("ONE48", "TWO24", "FOUR12")
     -- Pattern Detector Attributes: Pattern Detection Configuration
     AUTORESET_PATDET => "NO_RESET",    -- "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH" 
     MASK => X"3fffffffffff",           -- 48-bit mask value for pattern detect (1=ignore)
     PATTERN => X"000000000000",        -- 48-bit pattern match for pattern detect
     SEL_MASK => "MASK",                -- "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2" 
     SEL_PATTERN => "PATTERN",          -- Select pattern value ("PATTERN" or "C")
     USE_PATTERN_DETECT => "NO_PATDET", -- Enable pattern detect ("PATDET" or "NO_PATDET")
     -- Register Control Attributes: Pipeline Register Configuration
     ACASCREG => 1,                     -- Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
     ADREG => 0,                        -- Number of pipeline stages for pre-adder (0 or 1)
     ALUMODEREG => 1,                   -- Number of pipelgine stages for ALUMODE (0 or 1)
     AREG => front_latency(latency_g),  -- Number of pipeline stages for A (0, 1 or 2)
     BCASCREG => 1,                     -- Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
     BREG => front_latency(latency_g),  -- Number of pipeline stages for B (0, 1 or 2)
     CARRYINREG => 1,                   -- Number of pipeline stages for CARRYIN (0 or 1)
     CARRYINSELREG => 1,                -- Number of pipeline stages for CARRYINSEL (0 or 1)
     CREG => 1,                         -- Number of pipeline stages for C (0 or 1)
     DREG => 1,                         -- Number of pipeline stages for D (0 or 1)
     INMODEREG => 1,                    -- Number of pipeline stages for INMODE (0 or 1)
     MREG => 0,                         -- Number of multiplier pipeline stages (0 or 1)
     OPMODEREG => 1,                    -- Number of pipeline stages for OPMODE (0 or 1)
     PREG => back_latency(latency_g)    -- Number of pipeline stages for P (0 or 1)
  )
  port map (
     -- Useless ports, static inputs (no need to reset)
     CLK => clk,
     ACOUT => open, BCOUT => open, CARRYCASCOUT => open, MULTSIGNOUT => open,
     PCOUT => open, OVERFLOW => open, PATTERNBDETECT => open,
     PATTERNDETECT => open, UNDERFLOW => open, CARRYOUT => open,
     ACIN => (others => '0'), BCIN => (others => '0'), CARRYIN => '0',
     CARRYCASCIN => '0', MULTSIGNIN => '0', PCIN => (others => '0'),
     RSTA => '0', RSTALLCARRYIN => '0', RSTALUMODE => '0', RSTB => '0',
     RSTC => '0', RSTCTRL => '0', RSTD => '0', RSTINMODE => '0', RSTM => '0',
     RSTP => '0', CARRYINSEL => "000", ALUMODE => "0000",
     -- Control:
     INMODE => "00000", OPMODE => "0000101",
     -- Data:
     C => (others => '0'), D => (others => '0'),
     A => A_half_0, B => B_half_0, P => P_half_0,
     -- Enable most registers with glock_in
     CEA1 => dsp_enable, CEA2 => dsp_enable, CEAD => dsp_enable,
     CEALUMODE => dsp_enable, CEB1 => dsp_enable, CEB2 => dsp_enable,
     CEC => dsp_enable, CECARRYIN => dsp_enable, CECTRL => dsp_enable,
     CED => dsp_enable, CEINMODE => dsp_enable, CEM => dsp_enable,
     -- enable the output registers when the output needs to be updated
     CEP => output_enable
  );

  DSP48E1_half_1_inst : DSP48E1
  generic map (
     -- Feature Control Attributes: Data Path Selection
     A_INPUT => "DIRECT", -- port A
     B_INPUT => "DIRECT", -- port B
     USE_DPORT => FALSE,                -- Select D port usage (TRUE or FALSE)
     USE_MULT => "DYNAMIC",            -- Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
     USE_SIMD => "ONE48",               -- SIMD selection ("ONE48", "TWO24", "FOUR12")
     -- Pattern Detector Attributes: Pattern Detection Configuration
     AUTORESET_PATDET => "NO_RESET",    -- "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH" 
     MASK => X"3fffffffffff",           -- 48-bit mask value for pattern detect (1=ignore)
     PATTERN => X"000000000000",        -- 48-bit pattern match for pattern detect
     SEL_MASK => "MASK",                -- "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2" 
     SEL_PATTERN => "PATTERN",          -- Select pattern value ("PATTERN" or "C")
     USE_PATTERN_DETECT => "NO_PATDET", -- Enable pattern detect ("PATDET" or "NO_PATDET")
     -- Register Control Attributes: Pipeline Register Configuration
     ACASCREG => 1,                     -- Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
     ADREG => 0,                        -- Number of pipeline stages for pre-adder (0 or 1)
     ALUMODEREG => 1,                   -- Number of pipelgine stages for ALUMODE (0 or 1)
     AREG => front_latency(latency_g),  -- Number of pipeline stages for A (0, 1 or 2)
     BCASCREG => 1,                     -- Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
     BREG => front_latency(latency_g),  -- Number of pipeline stages for B (0, 1 or 2)
     CARRYINREG => 1,                   -- Number of pipeline stages for CARRYIN (0 or 1)
     CARRYINSELREG => 1,                -- Number of pipeline stages for CARRYINSEL (0 or 1)
     CREG => 1,                         -- Number of pipeline stages for C (0 or 1)
     DREG => 1,                         -- Number of pipeline stages for D (0 or 1)
     INMODEREG => 1,                    -- Number of pipeline stages for INMODE (0 or 1)
     MREG => 0,                         -- Number of multiplier pipeline stages (0 or 1)
     OPMODEREG => 1,                    -- Number of pipeline stages for OPMODE (0 or 1)
     PREG => back_latency(latency_g)    -- Number of pipeline stages for P (0 or 1)
  )
  port map (
     -- Useless ports, static inputs (no need to reset)
     CLK => clk,
     ACOUT => open, BCOUT => open, CARRYCASCOUT => open, MULTSIGNOUT => open,
     PCOUT => open, OVERFLOW => open, PATTERNBDETECT => open,
     PATTERNDETECT => open, UNDERFLOW => open, CARRYOUT => open,
     ACIN => (others => '0'), BCIN => (others => '0'), CARRYIN => '0',
     CARRYCASCIN => '0', MULTSIGNIN => '0', PCIN => (others => '0'),
     RSTA => '0', RSTALLCARRYIN => '0', RSTALUMODE => '0', RSTB => '0',
     RSTC => '0', RSTCTRL => '0', RSTD => '0', RSTINMODE => '0', RSTM => '0',
     RSTP => '0', CARRYINSEL => "000", ALUMODE => "0000",
     -- Control:
     INMODE => "00000", OPMODE => "0000101",
     -- Data:
     C => (others => '0'), D => (others => '0'),
     A => A_half_1, B => B_half_1, P => P_half_1,
     -- Enable most registers with glock_in
     CEA1 => dsp_enable, CEA2 => dsp_enable, CEAD => dsp_enable,
     CEALUMODE => dsp_enable, CEB1 => dsp_enable, CEB2 => dsp_enable,
     CEC => dsp_enable, CECARRYIN => dsp_enable, CECTRL => dsp_enable,
     CED => dsp_enable, CEINMODE => dsp_enable, CEM => dsp_enable,
     -- enable the output registers when the output needs to be updated
     CEP => output_enable
  );

  load_register: if latency_g > 1 generate
    main : process(clk)
    begin
      if rising_edge(clk) then
        if rstx = '0' then
          load_r <= '0';
        elsif glock_in = '0' then
          load_r <= load_in;
        end if;
      end if;
    end process main;
  end generate;

  result <= std_logic_vector(signed(full_out)
                           + signed(half_out_0)
                           + signed(half_out_1));

  result_reg: if latency_g > 2 generate
    result_proc : process(clk)
    begin
      if rising_edge(clk) then
        if rstx = '0' then
          result_r <= (others => '0');
        elsif glock_in = '0' then
          result_r <= result;
        end if;
      end if;
    end process result_proc;

    result_out <= result_r;
  end generate;

  result_comb: if latency_g <= 2 generate
    result_out <= result;
  end generate;

end rtl;
