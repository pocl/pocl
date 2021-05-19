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
-------------------------------------------------------------------------------
-- File       : almaif_arbiter.vhdl
-- Author     : Kati Tervo <kati.tervo@tuni.fi>
-- Company    : TUT/CPC
-- Created    : 2020-02-04
-- Last update: 2020-02-04
-- Platform   :
-- Standard   : VHDL'87
-------------------------------------------------------------------------------
-- Description: Adds registers into a almaif memory bus
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2020-02-04  1.0      katte  Created
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.tce_util.all;

entity almaif_membus_delay is
  generic (
    mem_dataw_g  : integer;
    mem_addrw_g  : integer
  ); port (
    clk                : in std_logic;
    rstx               : in std_logic;
    -- Bus from upstream
    s_avalid_in  : in  std_logic;
    s_aready_out : out std_logic;
    s_aaddr_in   : in  std_logic_vector(mem_addrw_g-1 downto 0);
    s_awren_in   : in  std_logic;
    s_astrb_in   : in std_logic_vector(mem_dataw_g/8-1 downto 0);
    s_adata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0);
    s_rvalid_out : out std_logic;
    s_rready_in  : in  std_logic;
    s_rdata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    -- Bus to downstream
    m_avalid_out : out std_logic;
    m_aready_in  : in  std_logic;
    m_aaddr_out  : out std_logic_vector(mem_addrw_g-1 downto 0);
    m_awren_out  : out std_logic;
    m_astrb_out  : out std_logic_vector(mem_dataw_g/8-1 downto 0);
    m_adata_out  : out std_logic_vector(mem_dataw_g-1 downto 0);
    m_rvalid_in  : in  std_logic;
    m_rready_out : out std_logic;
    m_rdata_in   : in  std_logic_vector(mem_dataw_g-1 downto 0)
  );
end almaif_membus_delay;

architecture rtl of almaif_membus_delay is

  signal avalid_r, awren_r, rvalid_r : std_logic;
  signal aready, rready : std_logic;
  signal aaddr_r : std_logic_vector(s_aaddr_in'range);
  signal astrb_r : std_logic_vector(s_astrb_in'range);
  signal adata_r, rdata_r : std_logic_vector(s_adata_in'range);
begin

  m_avalid_out <= avalid_r;
  s_aready_out <= aready;
  m_aaddr_out  <= aaddr_r;
  m_awren_out  <= awren_r;
  m_astrb_out  <= astrb_r;
  m_adata_out  <= adata_r;

  s_rvalid_out <= rvalid_r;
  m_rready_out <= rready;
  s_rdata_out  <= rdata_r;


  delay_comb : process(avalid_r, m_aready_in, rvalid_r, s_rready_in)
  begin
    if avalid_r = '0' or m_aready_in = '1' then
      aready <= '1';
    else
      aready <= '0';
    end if;

    if rvalid_r = '0' or s_rready_in = '1' then
      rready <= '1';
    else
      rready <= '0';
    end if;
  end process delay_comb;

  delay_sync : process(clk, rstx)
  begin
    if rstx = '0' then
      avalid_r <= '0';
      awren_r  <= '0';
      aaddr_r  <= (others => '0');
      astrb_r  <= (others => '0');
      adata_r  <= (others => '0');

      rvalid_r <= '0';
      rdata_r  <= (others => '0');
    elsif rising_edge(clk) then

      if avalid_r = '1' and m_aready_in = '1' then
        avalid_r <= '0';
      end if;

      if aready = '1' and s_avalid_in = '1' then
        avalid_r <= '1';
        aaddr_r <= s_aaddr_in;
        awren_r <= s_awren_in;
        astrb_r <= s_astrb_in;
        adata_r <= s_adata_in;
      end if;

      if rvalid_r = '1' and s_rready_in = '1' then
        rvalid_r <= '0';
      end if;

      if rready = '1' and m_rvalid_in = '1' then
        rvalid_r <= '1';
        rdata_r  <= m_rdata_in;
      end if;

    end if;
  end process delay_sync;



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
