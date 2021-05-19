-- Copyright (c) 2017 Tampere University
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
-- Title      : Memory bus splitter for AXI slave in AlmaIF
-- Project    :
-------------------------------------------------------------------------------
-- File       : membus_splitter.vhdl
-- Author     : Aleksi Tervo
-- Company    : Tampere University
-- Created    : 2017-06-01
-- Last update: 2017-06-01
-- Platform   :
-- Standard   : VHDL'93
-------------------------------------------------------------------------------
-- Description:
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author    Description
-- 2017-06-01  1.0      tervoa    Created
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.tce_util.all;

entity membus_splitter is
  generic (
    core_count_g      : integer;
    axi_addr_width_g  : integer;
    axi_data_width_g  : integer;
    ctrl_addr_width_g : integer;
    imem_addr_width_g : integer;
    dmem_addr_width_g : integer;
    pmem_addr_width_g : integer
  );
  port (
    -- AXI slave
    avalid_in  : in  std_logic;
    aready_out : out std_logic;
    aaddr_in   : in  std_logic_vector(axi_addr_width_g-2-1 downto 0);
    rvalid_out : out std_logic;
    rready_in  : in  std_logic;
    rdata_out  : out std_logic_vector(axi_data_width_g-1 downto 0);
    -- Control signals to arbiters
    dmem_avalid_out : out std_logic;
    dmem_aready_in  : in  std_logic;
    dmem_rvalid_in  : in  std_logic;
    dmem_rready_out : out std_logic;
    dmem_rdata_in   : in  std_logic_vector(axi_data_width_g-1 downto 0);

    pmem_avalid_out : out std_logic;
    pmem_aready_in  : in  std_logic;
    pmem_rvalid_in  : in  std_logic;
    pmem_rready_out : out std_logic;
    pmem_rdata_in   : in  std_logic_vector(axi_data_width_g-1 downto 0);

    imem_avalid_out : out std_logic;
    imem_aready_in  : in  std_logic;
    imem_rvalid_in  : in  std_logic;
    imem_rready_out : out std_logic;
    imem_rdata_in   : in  std_logic_vector(axi_data_width_g-1 downto 0);

    -- Signals to debugger(s)
    ctrl_avalid_out   : out std_logic;
    ctrl_aready_in    : in  std_logic;
    ctrl_rvalid_in    : in  std_logic;
    ctrl_rready_out   : out std_logic;
    ctrl_rdata_in     : in  std_logic_vector(core_count_g*axi_data_width_g-1
                                             downto 0);
    ctrl_core_sel_out : out std_logic_vector(bit_width(core_count_g)-1 downto 0)
  );
end entity membus_splitter;

architecture rtl of membus_splitter is
  constant CTRL_HIGH : integer := 0;
  constant CQ_HIGH : integer := 1;

  -- Calculate the AXI address width to sanity check toplevel generics
  constant mem_widths_c : integer_array := (ctrl_addr_width_g
                                            * bit_width(core_count_g),
                                            imem_addr_width_g,
                                            dmem_addr_width_g,
                                            pmem_addr_width_g);
  constant axi_addrw_c  : integer := return_highest(mem_widths_c, 4) + 2;

  signal core_sel : std_logic_vector(ctrl_core_sel_out'range);
  signal mem_sel  : std_logic_vector(1 downto 0);
begin
  -- Works as a combinatorial process as long as axislave
  -- keeps the address in range until it has read all data
  comb : process(avalid_in, aaddr_in, rready_in,
                 dmem_aready_in, dmem_rvalid_in, dmem_rdata_in,
                 pmem_aready_in, pmem_rvalid_in, pmem_rdata_in,
                 imem_aready_in, imem_rvalid_in, imem_rdata_in,
                 ctrl_rdata_in, ctrl_aready_in, ctrl_rvalid_in, core_sel)
  variable core_sel_int : integer range 0 to 2**bit_width(core_count_g);
  begin
    ctrl_avalid_out <= '0';
    ctrl_rready_out <= '0';
    imem_avalid_out <= '0';
    imem_rready_out <= '0';
    dmem_avalid_out <= '0';
    dmem_rready_out <= '0';
    pmem_avalid_out <= '0';
    pmem_rready_out <= '0';
    ctrl_core_sel_out  <= (others => '0');
    core_sel        <= aaddr_in(bit_width(core_count_g)+ctrl_addr_width_g-1
                                         downto ctrl_addr_width_g);
    if core_count_g > 1 then
      core_sel_int    := to_integer(unsigned(core_sel));
    else
      core_sel_int    := 0;
    end if;
    case to_integer(unsigned(aaddr_in(aaddr_in'high downto aaddr_in'high-1))) is
      when CTRL_HIGH =>
        ctrl_avalid_out <= avalid_in;
        aready_out      <= ctrl_aready_in;
        rvalid_out      <= ctrl_rvalid_in;
        ctrl_rready_out <= rready_in;
        rdata_out     <= ctrl_rdata_in((core_sel_int+1)*axi_data_width_g-1
                                      downto core_sel_int*axi_data_width_g);
        ctrl_core_sel_out <= core_sel;

      when CQ_HIGH =>
        dmem_avalid_out <= avalid_in;
        aready_out      <= dmem_aready_in;
        rvalid_out      <= dmem_rvalid_in;
        dmem_rready_out <= rready_in;
        rdata_out       <= dmem_rdata_in;

      when others => -- PMEM_HIGH
        pmem_avalid_out <= avalid_in;
        aready_out      <= pmem_aready_in;
        rvalid_out      <= pmem_rvalid_in;
        pmem_rready_out <= rready_in;
        rdata_out       <= pmem_rdata_in;
    end case;
  end process;


  ------------------------------------------------------------------------------
  -- Design-wide checks:
  ------------------------------------------------------------------------------
  -- coverage off
  -- pragma translate_off
  assert axi_addrw_c = axi_addrw_c
    report "Toplevel generic axi_addr_width_g does not equal address" &
           "width computed from the individual memories' widths."
    severity failure;
  -- pragma translate_on
  -- coverage on

end architecture rtl;
