-- Copyright (c) 2016 Tampere University.
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
-- Title      : AlmaIF memory arbiter
-- Project    : Almarvi
-------------------------------------------------------------------------------
-- File       : almaif_arbiter.vhdl
-- Author     : Kati Tervo  <kati.tervo@tuni.fi>
-- Company    : TUT/CPC
-- Created    : 2016-11-22
-- Last update: 2020-02-06
-- Platform   :
-- Standard   : VHDL'87
-------------------------------------------------------------------------------
-- Description: Acts as a memory arbiter between a TTA and an AlmaIF AXI bus
--              Static arbiter: AXI has precedence
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2016-11-22  1.0      katte   Created
-- 2017-04-26  1.1      katte   Sensitivity list fix
-- 2017-06-01  1.2      katte   Converted to memory buses with handshaking
-- 2017-06-27  1.3      katte   Split arbiter in two parts: between TTA cores
--                              and between the multicore TTA and AXI
-- 2020-02-06  1.4      katte   FIFO to TTA output to handle TTA stalling
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.tce_util.all;

entity almaif_axi_arbiter is
  generic (
    mem_dataw_g  : integer;
    mem_addrw_g  : integer
  ); port (
    clk                : in std_logic;
    rstx               : in std_logic;
    -- Bus to TTA
    tta_avalid_in  : in  std_logic;
    tta_aready_out : out std_logic;
    tta_aaddr_in   : in  std_logic_vector(mem_addrw_g-1 downto 0);
    tta_awren_in   : in  std_logic;
    tta_astrb_in   : in std_logic_vector(mem_dataw_g/8-1 downto 0);
    tta_adata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0);
    tta_rvalid_out : out std_logic;
    tta_rready_in  : in  std_logic;
    tta_rdata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    -- Bus to AXI if
    axi_avalid_in  : in  std_logic;
    axi_aready_out : out std_logic;
    axi_aaddr_in   : in  std_logic_vector(mem_addrw_g-1 downto 0);
    axi_awren_in   : in  std_logic;
    axi_astrb_in   : in  std_logic_vector(mem_dataw_g/8-1 downto 0);
    axi_adata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0);
    axi_rvalid_out : out std_logic;
    axi_rready_in  : in  std_logic;
    axi_rdata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    -- Bus to memory
    mem_avalid_out : out std_logic;
    mem_aready_in  : in  std_logic;
    mem_aaddr_out  : out std_logic_vector(mem_addrw_g-1 downto 0);
    mem_awren_out  : out std_logic;
    mem_astrb_out  : out std_logic_vector(mem_dataw_g/8-1 downto 0);
    mem_adata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    mem_rvalid_in  : in  std_logic;
    mem_rready_out : out std_logic;
    mem_rdata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0)
  );
end almaif_axi_arbiter;

architecture rtl of almaif_axi_arbiter is

  constant fifo_depth_log2_c : integer := 2;
  constant fifo_depth_c      : integer := 2**fifo_depth_log2_c;

  -- AXI signals padded to memory data width

  -- FIFO implemented as a shifted register array
  signal fifo_data_r : std_logic_vector(fifo_depth_c-1 downto 0);
  signal fifo_iter_r : unsigned(fifo_depth_log2_c-1 downto 0);

  signal axi_active      : std_logic;
  signal fifo_axi_active : std_logic;
  signal mem_awren       : std_logic;
  signal mem_avalid      : std_logic;
  signal mem_rready      : std_logic;
  signal tta_rvalid      : std_logic;


  constant tta_fifo_depth_log2_c : integer := 3;
  constant tta_fifo_depth_c      : integer := 2**fifo_depth_log2_c;
  type tta_fifo_data_t is array (fifo_depth_c-1 downto 0)
                          of std_logic_vector(mem_dataw_g-1 downto 0);

  signal tta_fifo_valid : unsigned(tta_fifo_depth_log2_c - 1 downto 0);
  signal tta_fifo_empty, tta_fifo_full : std_logic;
  signal tta_fifo_data  : tta_fifo_data_t;
begin

  axi_active <= axi_avalid_in;

  ------------------------------------------------------------------------------
  -- Access channel mux:
  ------------------------------------------------------------------------------
  amux : process (axi_active, mem_aready_in,
                  axi_avalid_in, axi_awren_in,
                  axi_aaddr_in, axi_adata_in, axi_astrb_in,
                  tta_avalid_in, tta_awren_in,
                  tta_aaddr_in, tta_adata_in, tta_astrb_in)
  begin
      if axi_active = '1' then
        mem_avalid     <= axi_avalid_in;
        axi_aready_out <= mem_aready_in;
        mem_awren      <= axi_awren_in;
        mem_aaddr_out  <= axi_aaddr_in;
        mem_adata_out  <= axi_adata_in;
        mem_astrb_out  <= axi_astrb_in;

        tta_aready_out <= '0';
      else
        mem_avalid     <= tta_avalid_in;
        tta_aready_out <= mem_aready_in;
        mem_awren      <= tta_awren_in;
        mem_aaddr_out  <= tta_aaddr_in;
        mem_adata_out  <= tta_adata_in;
        mem_astrb_out  <= tta_astrb_in;

        axi_aready_out <= '0';
      end if;
  end process amux;
  mem_avalid_out <= mem_avalid;
  mem_awren_out  <= mem_awren;

  ----------------------------------------------------------------------------
  -- FIFO to keep track of reads' destinations
  -- TODO: Handle FIFO filling up (not an issue with current mem model/alu?)
  ----------------------------------------------------------------------------
  fifo_sync : process(clk, rstx)
    variable fifo_data_v : std_logic_vector(fifo_data_r'range);
    variable fifo_iter_v : unsigned(fifo_iter_r'range);
  begin
    if rstx = '0' then
      fifo_data_r <= (others => '0');
      fifo_iter_r <= (others => '0');
    elsif rising_edge(clk) then
      fifo_data_v := fifo_data_r;
      fifo_iter_v := fifo_iter_r;
      if mem_rvalid_in = '1' and mem_rready = '1' and fifo_iter_r > 0 then
        fifo_data_v(fifo_data_v'high-1 downto 0) :=
            fifo_data_v(fifo_data_v'high downto 1);
        fifo_data_v(fifo_data_v'high) := '0';
        fifo_iter_v := fifo_iter_v - 1;
      end if;

      if mem_avalid = '1' and mem_aready_in = '1' and mem_awren = '0' then
        fifo_data_v(to_integer(fifo_iter_v)) := axi_active;
        fifo_iter_v := fifo_iter_v + 1;
      end if;

      fifo_iter_r <= fifo_iter_v;
      fifo_data_r <= fifo_data_v;

    end if;
  end process fifo_sync;
  fifo_axi_active <= fifo_data_r(0);

  ------------------------------------------------------------------------------
  -- Response channel mux:
  -- TODO: Handle reset better; works with local memory but will cause issues
  --       w/ axi
  ------------------------------------------------------------------------------
  rmux : process (fifo_axi_active, axi_rready_in,
                  mem_rvalid_in, mem_rdata_in, tta_fifo_full)
  begin
    if fifo_axi_active = '1' then
      mem_rready     <= axi_rready_in;
      axi_rvalid_out <= mem_rvalid_in;
      axi_rdata_out <= mem_rdata_in;
      tta_rvalid <= '0';
    else
      mem_rready <= not tta_fifo_full;
      tta_rvalid <= mem_rvalid_in;
      axi_rvalid_out <= '0';
      axi_rdata_out  <= (others => '0');
    end if;
  end process rmux;
  mem_rready_out <= mem_rready;

  tta_fifo : process(clk, rstx)
    variable fifo_data_v : tta_fifo_data_t;
    variable fifo_iter_v : unsigned(tta_fifo_valid'range);
  begin
    if rstx = '0' then
      tta_fifo_data <= (others => (others => '0'));
      tta_fifo_valid <= (others => '0');
    elsif rising_edge(clk) then
      fifo_data_v := tta_fifo_data;
      fifo_iter_v := tta_fifo_valid;

      if tta_rready_in = '1' and tta_fifo_empty = '0'  then
        fifo_data_v(fifo_data_v'high-1 downto 0) :=
            fifo_data_v(fifo_data_v'high downto 1);
        fifo_data_v(fifo_data_v'high) := (others => '0');
        fifo_iter_v := fifo_iter_v - 1;
      end if;

      if tta_rvalid = '1' and (tta_rready_in = '0' or tta_fifo_empty = '0')
         and tta_fifo_full = '0' then
        fifo_data_v(to_integer(fifo_iter_v)) := mem_rdata_in;
        fifo_iter_v := fifo_iter_v + 1;
      end if;

      tta_fifo_data <= fifo_data_v;
      tta_fifo_valid <= fifo_iter_v;
    end if;
  end process;

  tta_fifo_empty <= '1' when to_integer(tta_fifo_valid) = 0 else '0';
  tta_fifo_full <= '1' when to_integer(tta_fifo_valid) = tta_fifo_depth_c-1 else '0';
  tta_rvalid_out <= tta_rvalid or not tta_fifo_empty;

  tta_rdata_out <= tta_fifo_data(0) when tta_fifo_empty = '0' else mem_rdata_in;


  ------------------------------------------------------------------------------
  -- Design-wide checks:
  ------------------------------------------------------------------------------
  -- coverage off
  -- pragma translate_off
  assert (mem_dataw_g/8)*8 = mem_dataw_g
    report "Data width must be divisible by 8"
    severity failure;
  -- pragma translate_on
  -- coverage on

end architecture rtl;
