-- Copyright (c) 2016-2018 Tampere University.
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
-- Title      : AlmaIF memory bus expander
-- Project    : Almarvi
-------------------------------------------------------------------------------
-- File       : almaif_expander.vhdl
-- Author     : Aleksi Tervo  <aleksi.tervo@tut.fi>
-- Company    : TUT/CPC
-- Created    : 2016-11-22
-- Last update: 2016-11-22
-- Platform   :
-- Standard   : VHDL'87
-------------------------------------------------------------------------------
-- Description: Acts as glue between one port of a dual-port memory and the
--              memory bus from AXI
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2016-11-22  1.0      tervoa  Created
-- 2017-04-26  1.1      tervoa  Sensitivity list fix
-- 2017-06-01  1.2      tervoa  Converted to memory buses with handshaking
-- 2017-06-27  1.3      tervoa  Split arbiter in two parts: between TTA cores
--                              and between the multicore TTA and AXI
-- 2017-10-22  1.4      tervoa  Changed to dual-port memory, no need for
--                              arbiting
-- 2018-01-15  1.5      tervoa  Fix typo in signal name
-- 2018-07-30  1.6      tervoa  Support for optional sync reset
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.tce_util.all;

entity almaif_axi_expander is

  generic (
    mem_dataw_g  : integer;
    mem_addrw_g  : integer;
    axi_dataw_g  : integer;
    axi_addrw_g  : integer;
    sync_reset_g : integer
  ); port (
    clk                : in std_logic;
    rstx               : in std_logic;
    -- Bus to AXI if
    axi_avalid_in  : in  std_logic;
    axi_aready_out : out std_logic;
    axi_aaddr_in   : in  std_logic_vector(axi_addrw_g-2-1 downto 0);
    axi_awren_in   : in  std_logic;
    axi_astrb_in   : in  std_logic_vector((axi_dataw_g+7)/8-1 downto 0);
    axi_adata_in   : in  std_logic_vector(axi_dataw_g-1 downto 0);
    axi_rvalid_out : out std_logic;
    axi_rready_in  : in  std_logic;
    axi_rdata_out  : out std_logic_vector(axi_dataw_g-1 downto 0);
    -- Bus to memory
    mem_avalid_out : out std_logic;
    mem_aready_in  : in  std_logic;
    mem_aaddr_out  : out std_logic_vector(mem_addrw_g-1 downto 0);
    mem_awren_out  : out std_logic;
    mem_astrb_out  : out std_logic_vector((mem_dataw_g+7)/8-1 downto 0);
    mem_adata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    mem_rvalid_in  : in  std_logic;
    mem_rready_out : out std_logic;
    mem_rdata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0)
  );
end almaif_axi_expander;

architecture rtl of almaif_axi_expander is

  constant sync_reset_c : boolean := sync_reset_g /= 0;

  constant mem_word_width_c  : integer :=  -- ceil(mem_dataw_g/axi_dataw_g)
                                   (mem_dataw_g+axi_dataw_g-1)/axi_dataw_g;
  constant mem_word_sel_c    : integer := bit_width(mem_word_width_c);
  constant axi_bytes_c       : integer := (axi_dataw_g+7)/8;
  constant mem_bytes_c       : integer := (mem_dataw_g+7)/8;
  constant fifo_depth_log2_c : integer := 2;

  -- AXI signals padded to memory data width
  signal axi_astrb_padded   : std_logic_vector((mem_dataw_g+7)/8-1 downto 0);
  signal axi_adata_padded   : std_logic_vector(mem_dataw_g-1 downto 0);
  signal axi_aaddr_stripped : std_logic_vector(mem_aaddr_out'range);
  signal axi_word_sel       : std_logic_vector(mem_word_sel_c-1 downto 0);
  signal axi_rdata_padded   : std_logic_vector(axi_dataw_g*mem_word_width_c-1
                                               downto 0);

  -- FIFO implemented as a shifted register array
  type fifo_array_t is array (natural range <>)
                       of std_logic_vector(mem_word_sel_c-1 downto 0);
  signal fifo_data_r : fifo_array_t(2**fifo_depth_log2_c-1 downto 0);
  signal fifo_iter_r : unsigned(fifo_depth_log2_c-1 downto 0);

  signal fifo_word_sel   : std_logic_vector(mem_word_sel_c-1 downto 0);
begin
  ------------------------------------------------------------------------------
  -- AXI signal glue:    pad adata and astrb to memory width, strip aaddr to
  --                     meaninful width for memory
  ------------------------------------------------------------------------------
  axi_expand : process(axi_adata_in, axi_astrb_in, axi_word_sel, mem_rdata_in,
                       axi_aaddr_in)
  variable adata_pad_v : std_logic_vector(axi_dataw_g*mem_word_width_c-1
                                            downto 0);
  variable astrb_pad_v : std_logic_vector(axi_bytes_c*mem_word_width_c-1
                                            downto 0);
  variable rdata_pad_v : std_logic_vector(axi_dataw_g*mem_word_width_c-1
                                            downto 0);
  begin
    adata_pad_v   := (others => '0');
    astrb_pad_v   := (others => '0');
    axi_word_sel  <= axi_aaddr_in(mem_word_sel_c-1 downto 0);
    for I in mem_word_width_c-1 downto 0 loop
      adata_pad_v(axi_dataw_g*(I+1)-1 downto axi_dataw_g*I) := axi_adata_in;
      if to_integer(unsigned(axi_word_sel)) = I or mem_word_width_c = 1 then
        astrb_pad_v(axi_bytes_c*(I+1)-1 downto axi_bytes_c*I)
            := axi_astrb_in;
      end if;
    end loop;
    axi_adata_padded <= adata_pad_v(axi_adata_padded'range);
    axi_astrb_padded <= astrb_pad_v(axi_astrb_padded'range);

    if mem_word_width_c = 1 then
      axi_aaddr_stripped <= axi_aaddr_in(mem_addrw_g-1 downto 0);
    else
      axi_aaddr_stripped <= axi_aaddr_in(mem_addrw_g+mem_word_sel_c-1
                                         downto mem_word_sel_c);
    end if;

    axi_rdata_padded                     <= (others => '0');
    axi_rdata_padded(mem_rdata_in'range) <= mem_rdata_in;
  end process;

  ------------------------------------------------------------------------------
  -- Access channel:
  ------------------------------------------------------------------------------
  mem_avalid_out <= axi_avalid_in;
  axi_aready_out <= mem_aready_in;
  mem_awren_out  <= axi_awren_in;
  mem_aaddr_out  <= axi_aaddr_stripped;
  mem_adata_out  <= axi_adata_padded;
  mem_astrb_out  <= axi_astrb_padded;

  ----------------------------------------------------------------------------
  -- FIFO to keep track of reads' destinations
  -- TODO: Handle FIFO filling up (not an issue with current mem model/alu?)
  ----------------------------------------------------------------------------
  fifo_sync : process(clk, rstx)
    variable fifo_data_v : fifo_array_t(fifo_data_r'range);
    variable fifo_iter_v : unsigned(fifo_iter_r'range);
  begin
    if not sync_reset_c and rstx = '0' then
      fifo_data_r <= (others => (others => '0'));
      fifo_iter_r <= (others => '0');
    elsif rising_edge(clk) then
      if sync_reset_c and rstx = '0' then
        fifo_data_r <= (others => (others => '0'));
        fifo_iter_r <= (others => '0');
      else
        fifo_data_v := fifo_data_r;
        fifo_iter_v := fifo_iter_r;
        if mem_rvalid_in = '1' and axi_rready_in = '1' and fifo_iter_r > 0 then
          fifo_data_v(fifo_data_v'high-1 downto 0) :=
              fifo_data_v(fifo_data_v'high downto 1);
          fifo_data_v(fifo_data_v'high) := (others => '0');
          fifo_iter_v := fifo_iter_v - 1;
        end if;

        if axi_avalid_in = '1' and mem_aready_in = '1' and axi_awren_in = '0' then
          fifo_data_v(to_integer(fifo_iter_v)) := axi_word_sel;
          fifo_iter_v := fifo_iter_v + 1;
        end if;

        fifo_iter_r <= fifo_iter_v;
        fifo_data_r <= fifo_data_v;
      end if;
    end if;
  end process fifo_sync;
  fifo_word_sel    <= fifo_data_r(0);

  ------------------------------------------------------------------------------
  -- Response channel mux:
  -- TODO: Handle reset better; works with local memory but will cause issues
  --       w/ axi
  ------------------------------------------------------------------------------
  rmux : process (fifo_word_sel, axi_rdata_padded)
    variable dst_sel_int : integer;
  begin
    if mem_word_width_c = 1 then
      axi_rdata_out <= axi_rdata_padded;
    else
      dst_sel_int   := to_integer(unsigned(fifo_word_sel));
      axi_rdata_out <= axi_rdata_padded(axi_dataw_g*(dst_sel_int+1)-1
                                 downto axi_dataw_g*dst_sel_int);
    end if;
  end process rmux;
  mem_rready_out <= axi_rready_in;
  axi_rvalid_out <= mem_rvalid_in;

  ------------------------------------------------------------------------------
  -- Design-wide checks:
  ------------------------------------------------------------------------------
  -- coverage off
  -- pragma translate_off
  assert axi_addrw_g >= 2+mem_addrw_g+mem_word_sel_c
    report "AXI address width is too short to encode all the addresses"
    severity failure;

  assert mem_dataw_g >= axi_dataw_g
    report "Memory data width must be greater than or equal to AXI data width"
    severity failure;
  -- pragma translate_on
  -- coverage on

end architecture rtl;
