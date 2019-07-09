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
-- Title      : Memory access decoder
-- Project    : Almarvi
-------------------------------------------------------------------------------
-- File       : almaif_decoder.vhdl
-- Author     : Aleksi Tervo  <aleksi.tervo@tut.fi>
-- Company    : TUT/CPC
-- Created    : 2017-06-13
-- Last update: 2017-06-13
-- Platform   :
-- Standard   : VHDL'87
-------------------------------------------------------------------------------
-- Description: Directs memory accesses to either the local memory or an
--              AXI master
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2017-06-13  1.0      tervoa  Created
-- 2018-07-30  1.1      tervoa   Support for optional sync reset
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.math_real.all;
use work.tce_util.all;

entity almaif_decoder is

  generic (
    mem_dataw_g  : integer := 32;
    mem_addrw_g  : integer;
    axi_addrw_g  : integer := 32;
    mem_offset_g : integer;
    sync_reset_g : integer
  ); port (
    clk                : in std_logic;
    rstx               : in std_logic;
    -- Bus from arbiter
    arb_avalid_in  : in  std_logic;
    arb_aready_out : out std_logic;
    arb_aaddr_in   : in  std_logic_vector(axi_addrw_g-2-1 downto 0);
    arb_awren_in   : in  std_logic;
    arb_astrb_in   : in std_logic_vector(mem_dataw_g/8-1 downto 0);
    arb_adata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0);
    --
    arb_rvalid_out : out std_logic;
    arb_rready_in  : in  std_logic;
    arb_rdata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    -- Bus to local memory
    mem_avalid_out : out std_logic;
    mem_aready_in  : in  std_logic;
    mem_aaddr_out  : out std_logic_vector(mem_addrw_g-1 downto 0);
    mem_awren_out  : out std_logic;
    mem_astrb_out  : out std_logic_vector(mem_dataw_g/8-1 downto 0);
    mem_adata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    --
    mem_rvalid_in  : in  std_logic;
    mem_rready_out : out std_logic;
    mem_rdata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0);
    -- AXI lite master
    m_axi_awvalid  : out std_logic;
    m_axi_awready  : in  std_logic;
    m_axi_awaddr   : out std_logic_vector(axi_addrw_g-1 downto 0);
    m_axi_awprot   : out std_logic_vector(3-1 downto 0);
    --
    m_axi_wvalid   : out std_logic;
    m_axi_wready   : in  std_logic;
    m_axi_wdata    : out std_logic_vector(mem_dataw_g-1 downto 0);
    m_axi_wstrb    : out std_logic_vector(mem_dataw_g/8-1 downto 0);
    --
    m_axi_bvalid   : in  std_logic;
    m_axi_bready   : out std_logic;
    --
    m_axi_arvalid  : out std_logic;
    m_axi_arready  : in  std_logic;
    m_axi_araddr   : out std_logic_vector(axi_addrw_g-1 downto 0);
    m_axi_arprot   : out std_logic_vector(3-1 downto 0);
    --
    m_axi_rvalid   : in  std_logic;
    m_axi_rready   : out std_logic;
    m_axi_rdata    : in  std_logic_vector(mem_dataw_g-1 downto 0)
  );
end almaif_decoder;

architecture rtl of almaif_decoder is

  constant sync_reset_c : boolean := sync_reset_g /= 0;

  constant dst_axi_c  : std_logic := '1';
  constant dst_mem_c  : std_logic := '0';
  signal dst_sel, fifo_dst_sel, access_success, read_success : std_logic;
  signal fifo_data_r : std_logic_vector(4-1 downto 0);
  signal fifo_iter_r : unsigned(2-1 downto 0);

  constant mem_offset_c : std_logic_vector(axi_addrw_g-2-1 downto 0)
                := std_logic_vector(to_unsigned(mem_offset_g/4, axi_addrw_g-2));
  constant mem_mask_c   : std_logic_vector(axi_addrw_g-2-1 downto 0)
                     -- := (mem_addrw_g-2-1 downto 0 => '0', others => '1');
                        := not std_logic_vector(to_unsigned(2**(mem_addrw_g-2)-1,
                                                            axi_addrw_g-2));

  constant mem_width_log2 : integer := integer(ceil(log2(real(mem_dataw_g))));

  signal m_axi_wdata_r : std_logic_vector(m_axi_wdata'range);
  signal m_axi_wstrb_r : std_logic_vector(m_axi_wstrb'range);

  -- All accesses are unprivileged/secure/data
  constant axi_axprot_c : std_logic_vector(3-1 downto 0) := "000";

  signal m_axi_wvalid_r, write_load : std_logic;

begin


  ------------------------------------------------------------------------------
  -- Direct accesses to AXI or local memory
  ------------------------------------------------------------------------------
  asel_comb : process(arb_aaddr_in, arb_avalid_in, arb_awren_in,
                      mem_aready_in,
                      m_axi_arready, m_axi_awready,
                      m_axi_wvalid_r, m_axi_wready)
  begin
    mem_avalid_out <= '0';
    m_axi_arvalid  <= '0';
    m_axi_awvalid  <= '0';
    arb_aready_out <= '0';
    access_success <= '0';
    write_load     <= '0';
    dst_sel        <= dst_mem_c;
    if (arb_aaddr_in and mem_mask_c) = mem_offset_c then
      mem_avalid_out <= arb_avalid_in;
      arb_aready_out <= mem_aready_in;
      access_success <= mem_aready_in and arb_avalid_in;
    else
      dst_sel        <= dst_axi_c;
      if arb_awren_in = '0' then
        m_axi_arvalid  <= arb_avalid_in;
        arb_aready_out <= m_axi_arready;
        access_success <= arb_avalid_in and m_axi_arready;
      elsif      arb_awren_in = '1'
            and (m_axi_wvalid_r = '0' or m_axi_wready = '1') then
        m_axi_awvalid  <= arb_avalid_in;
        write_load     <= '1';
        arb_aready_out <= m_axi_awready;
        access_success <= arb_avalid_in and m_axi_awready;
      end if;
    end if;
  end process asel_comb;

  -- Common signals to memory interface
  mem_aaddr_out <= arb_aaddr_in(mem_aaddr_out'range);
  mem_awren_out <= arb_awren_in;
  mem_astrb_out <= arb_astrb_in;
  mem_adata_out <= arb_adata_in;

  -- Common signals to AXI
  m_axi_awaddr  <= arb_aaddr_in & "00";
  m_axi_awprot  <= axi_axprot_c;

  m_axi_araddr  <= arb_aaddr_in & "00";
  m_axi_arprot  <= axi_axprot_c;

  ------------------------------------------------------------------------------
  -- FIFO to keep accesses to local/global memory in order
  ------------------------------------------------------------------------------
  fifo_sync : process(clk, rstx)
    variable fifo_data_v : std_logic_vector(fifo_data_r'range);
    variable fifo_iter_v : unsigned(fifo_iter_r'range);
  begin
    if not sync_reset_c and rstx = '0' then
      fifo_data_r <= (others => '0');
      fifo_iter_r <= (others => '0');
    elsif rising_edge(clk) then
      if sync_reset_c and rstx = '0' then
        fifo_data_r <= (others => '0');
        fifo_iter_r <= (others => '0');
      else
        fifo_data_v := fifo_data_r;
        fifo_iter_v := fifo_iter_r;
        if read_success = '1' and fifo_iter_r > 0 then
          fifo_data_v(fifo_data_v'high-1 downto 0) :=
              fifo_data_v(fifo_data_v'high downto 1);
          fifo_data_v(fifo_data_v'high) := '0';
          fifo_iter_v := fifo_iter_v - 1;
        end if;

        if access_success = '1' and arb_awren_in = '0' then
          fifo_data_v(to_integer(fifo_iter_v)) := dst_sel;
          fifo_iter_v := fifo_iter_v + 1;
        end if;

        fifo_iter_r <= fifo_iter_v;
        fifo_data_r <= fifo_data_v;
      end if;
    end if;
  end process fifo_sync;
  fifo_dst_sel    <= fifo_data_r(0);


  ------------------------------------------------------------------------------
  -- Select response to direct to arbiter
  ------------------------------------------------------------------------------
  rsel_comb : process(fifo_dst_sel, m_axi_rvalid, arb_rready_in, mem_rvalid_in,
                      m_axi_rdata, mem_rdata_in)
  begin
    if fifo_dst_sel = dst_axi_c then
      arb_rvalid_out <= m_axi_rvalid;
      mem_rready_out <= '0';
      m_axi_rready   <= arb_rready_in;
      arb_rdata_out  <= m_axi_rdata;
      read_success   <= arb_rready_in and m_axi_rvalid;
    else
      arb_rvalid_out <= mem_rvalid_in;
      mem_rready_out <= arb_rready_in;
      m_axi_rready   <= '0';
      arb_rdata_out  <= mem_rdata_in;
      read_success   <= arb_rready_in and mem_rvalid_in;
    end if;
  end process rsel_comb;

  ------------------------------------------------------------------------------
  -- AXI write, response channels : drive m_axi_{w,b}* out ports
  ------------------------------------------------------------------------------
  axi_w_channels_sync : process(clk, rstx)
  begin
    if not sync_reset_c and rstx = '0' then
      m_axi_wvalid_r <= '0';
      m_axi_wdata_r  <= (others => '0');
      m_axi_wstrb_r  <= (others => '0');
    elsif rising_edge(clk) then
      if sync_reset_c and rstx = '0' then
        m_axi_wvalid_r <= '0';
        m_axi_wdata_r  <= (others => '0');
        m_axi_wstrb_r  <= (others => '0');
      else
        if m_axi_wready = '1' then
          m_axi_wvalid_r <= '0';
        end if;

        if write_load = '1' and m_axi_awready = '1' then
          m_axi_wvalid_r <= '1';
          m_axi_wdata_r  <= arb_adata_in;
          m_axi_wstrb_r  <= arb_astrb_in;
        end if;
      end if;
    end if;
  end process axi_w_channels_sync;
  m_axi_wvalid <= m_axi_wvalid_r;
  m_axi_wdata  <= m_axi_wdata_r;
  m_axi_wstrb  <= m_axi_wstrb_r;
  -- Ignore the response
  m_axi_bready <= '1';

  ------------------------------------------------------------------------------
  -- Design-wide checks:
  ------------------------------------------------------------------------------
  -- coverage off
  -- pragma translate_off
  assert 2**mem_width_log2 = mem_dataw_g and mem_dataw_g >= 8
    report "Data width must be a power-of-two multiple of bytes"
    severity failure;

  assert (mem_offset_c and mem_mask_c) = mem_offset_c
    report "Memory must be aligned to its size"
    severity failure;

  assert mem_addrw_g <= axi_addrw_g
    report "Memory must fit in AXI address space"
    severity failure;
  -- pragma translate_on
  -- coverage on

end architecture rtl;
