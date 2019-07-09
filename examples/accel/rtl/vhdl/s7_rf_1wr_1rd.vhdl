-- Copyright (c) 2017 Tampere University.
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
-- Title      : 1R1W RF primarily for Xilinx Series 7 devices
-------------------------------------------------------------------------------
-- File       : s7_rf_1wr_1rd.vhdl
-- Author     : Lasse Lehtonen <lasse.lehtonen@phnet.fi>
-- Company    : Tampere University
-- Created    : 2017
-- Last update: 2018-06-09
-- Platform   : Xilinx Series 7
-------------------------------------------------------------------------------
-- Description: Register file with one read and one write port.
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author   Description
-- 2017        1.0      lasse    Initial version
-- 2018-06-09  1.1      tervoa   Cleanup for HDB
-------------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;

use work.tce_util.all;

entity s7_rf_1wr_1rd is

  generic (
    width_g : integer;
    depth_g : integer);
  port (
    clk      : in std_logic;
    rstx     : in std_logic;
    glock_in : in std_logic;

    load_rd_in  : in  std_logic;
    data_rd_out : out std_logic_vector(width_g-1 downto 0);
    addr_rd_in  : in  std_logic_vector(bit_width(depth_g)-1 downto 0);

    load_wr_in : in std_logic;
    data_wr_in : in std_logic_vector(width_g-1 downto 0);
    addr_wr_in : in std_logic_vector(bit_width(depth_g)-1 downto 0)
  );

end entity s7_rf_1wr_1rd;

architecture rtl of s7_rf_1wr_1rd is
  type reg_type is array (0 to depth_g-1) of std_logic_vector(width_g-1 downto 0);
  signal reg_r : reg_type;
begin

  main_sp : process (clk)
  begin
    if clk'event and clk = '1' then
      if load_wr_in = '1' then
        reg_r(to_integer(unsigned(addr_wr_in))) <= data_wr_in;
      end if;
    end if;
  end process main_sp;

  data_rd_out <= reg_r(to_integer(unsigned(addr_rd_in)));

end architecture rtl;
