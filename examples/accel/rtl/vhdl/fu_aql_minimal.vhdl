-- Copyright (c) 2017 Tampere University of Technology.
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
-- Title      : LSU for AlmaIF Integrator
-- Project    : Almarvi
-------------------------------------------------------------------------------
-- File       : fu_lsu_32b.vhdl
-- Author     : Kati Tervo
-- Company    :
-- Created    : 2019-05-28
-- Last update: 2019-05-28
-- Platform   :
-------------------------------------------------------------------------------
-- Description: 32 bit wide LSU with parametric endianness
--              External ports:
--  | Signal       | Comment
--  ---------------------------------------------------------------------------
--  | read_idx_out | Read index from the FU to the debug interface
--  ---------------------------------------------------------------------------
--
-- Revisions  :
-- Date        Version  Author  Description
-- 2019-05-28  1.0      katte   Created
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity fu_aql_minimal is
  port(
    clk           : in std_logic;
    rstx          : in std_logic;
    glock         : in std_logic;

    -- External signals
    read_idx_out  : out std_logic_vector(64-1 downto 0);
    read_idx_clear_in : in std_logic_vector(0 downto 0);

    -- Architectural ports
    t1_data_in    : in  std_logic_vector(32-1 downto 0);
    t1_load_in    : in  std_logic;
    t1_opcode_in  : in  std_logic_vector(0 downto 0);

    r1_data_out   : out std_logic_vector(32-1 downto 0)

  );
end fu_aql_minimal;

architecture rtl of fu_aql_minimal is

  constant OPC_GET_READ_IDX_LOW  : std_logic_vector(t1_opcode_in'range) := "0";
  constant OPC_INC_READ_IDX      : std_logic_vector(t1_opcode_in'range) := "1";

  signal read_idx_r     : std_logic_vector(read_idx_out'range);
  signal result_r       : std_logic_vector(32 - 1 downto 0);

begin

  read_idx_out <= read_idx_r;
  r1_data_out  <= result_r;

  operation_logic : process(clk, rstx)
  begin
    if rstx = '0' then
      read_idx_r  <= (others => '0');
      result_r    <= (others => '0');
    elsif rising_edge(clk) then
      if read_idx_clear_in = "1" then
        read_idx_r <= (others => '0');
      end if;

      if glock = '0' then
        if t1_load_in = '1' then
          case t1_opcode_in is
            when OPC_GET_READ_IDX_LOW =>
              result_r <= read_idx_r(result_r'range);
            when others => -- Increment
              read_idx_r <= std_logic_vector(unsigned(read_idx_r)
                                           + unsigned(t1_data_in));
          end case;
        end if;
      end if;
    end if;
  end process operation_logic;


end rtl;
