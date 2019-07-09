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
-- Title      : Simulation model for non-pipelined multiplier
-------------------------------------------------------------------------------
-- File       : mul_dsp_comb_sim.vhdl
-- Author     : Kati Tervo
-- Company    :
-- Created    : 2019-03-10
-- Last update: 2019-03-10
-- Platform   :
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2019-03-10  1.0      katte   Created
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity mul_dsp48 is
  generic (
    latency_g : integer
  ); port(
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

  signal mul_result : std_logic_vector(64 - 1 downto 0);
  signal result     : std_logic_vector(32 - 1 downto 0);

  type result_arr is array (latency_g downto 0)
                  of std_logic_vector(32-1 downto 0);
  signal result_r : result_arr;

begin

  mul_result <= std_logic_vector(unsigned(operand_a_in)
                               * unsigned(operand_b_in));
  result     <= std_logic_vector(unsigned(mul_result(32-1 downto 0))
                               + unsigned(operand_c_in));

  comb: if latency_g = 0 generate
    result_out <= result;
  end generate;

  sync: if latency_g > 0 generate
    operation_sync : process(clk)
    begin
      if rising_edge(clk) then
        if rstx = '0' then
          result_r <= (others => (others => '0'));
        elsif glock_in = '0' then
          if load_in = '1' then
            result_r(0) <= result;
          end if;

          result_r(result_r'high downto 1) <= result_r(result_r'high-1 downto 0);
        end if;
      end if;
    end process;

    result_out <= result_r(latency_g-1);
  end generate;


end rtl;
