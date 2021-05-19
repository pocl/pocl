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
-------------------------------------------------------------------------------
-- Title      : LSU interface registers
-------------------------------------------------------------------------------
-- File       : lsu_registers.vhdl
-- Author     : Kati Tervo
-- Company    :
-- Created    : 2019-08-27
-- Last update: 2019-08-27
-- Platform   :
-------------------------------------------------------------------------------
-- Description: LSU interface, handling registers and locking
--
-- Revisions  :
-- Date        Version  Author  Description
-- 2019-08-27  1.0      katte   Created
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.math_real.all;

entity lsu_registers is
  generic (
    dataw_g           : integer := 32;
    low_bits_g        : integer := 2;
    addrw_g           : integer := 11
  );
  port(
    clk           : in std_logic;
    rstx          : in std_logic;
    glock_in      : in std_logic;
    glockreq_out  : out std_logic;

    avalid_in     : in std_logic;
    awren_in      : in std_logic;
    aaddr_in      : in std_logic_vector(addrw_g-1 downto 0);
    astrb_in      : in std_logic_vector(dataw_g/8-1 downto 0);
    adata_in      : in std_logic_vector(dataw_g-1 downto 0);

    avalid_out    : out std_logic;
    aready_in     : in std_logic;
    aaddr_out     : out std_logic_vector(addrw_g-low_bits_g-1 downto 0);
    awren_out     : out std_logic;
    astrb_out     : out std_logic_vector(dataw_g/8-1 downto 0);
    adata_out     : out std_logic_vector(dataw_g-1 downto 0);

    rvalid_in    : in std_logic;
    rready_out   : out std_logic;

    rdata_in     : in std_logic_vector(dataw_g-1 downto 0);
    rdata_out    : out std_logic_vector(dataw_g-1 downto 0);

    addr_low_out : out std_logic_vector(low_bits_g-1 downto 0)
  );
end lsu_registers;

architecture rtl of lsu_registers is

  -- Access channel registers
  signal avalid_r    : std_logic;
  signal aaddr_r     : std_logic_vector(aaddr_out'range);
  signal awren_r     : std_logic;
  signal astrb_r     : std_logic_vector(astrb_out'range);
  signal adata_r     : std_logic_vector(adata_out'range);
  signal rready_r    : std_logic;
  signal rready_rr    : std_logic;
  signal rdata_r : std_logic_vector(rdata_in'range);

  signal addr_low_r, addr_low_rr  : std_logic_vector(addr_low_out'range);

  signal fu_glock, glockreq : std_logic;
begin
  -- Design-wide assertions
  -- coverage off
  -- synthesis translate_off
  assert low_bits_g = integer(ceil(log2(real(dataw_g/8))))
    report "Incorrect low_bits_g value"
    severity failure;
  -- coverage on
  -- synthesis translate_on

  avalid_out <= avalid_r;
  awren_out  <= awren_r;
  aaddr_out  <= aaddr_r;
  astrb_out  <= astrb_r;
  adata_out  <= adata_r;
  rready_out <= rready_rr;
  addr_low_out <= addr_low_rr;

  gen_lockreq : process(rready_rr, rvalid_in, avalid_r, aready_in,
                        glock_in, glockreq)
  begin
    if    (rready_rr = '1' and rvalid_in = '0')
       or (avalid_r = '1' and aready_in = '0') then
      glockreq <= '1';
    else
      glockreq <= '0';
    end if;

    fu_glock     <= glockreq or glock_in;
    glockreq_out <= glockreq;
  end process gen_lockreq;

  access_channel_sync : process(clk, rstx)
  begin
    if rstx = '0' then
      avalid_r <= '0';
      awren_r  <= '0';
      aaddr_r  <= (others => '0');
      astrb_r  <= (others => '0');
      adata_r  <= (others => '0');
      rready_r <= '0';
      rready_rr <= '0';
      addr_low_r <= (others => '0');
      addr_low_rr <= (others => '0');
    elsif rising_edge(clk) then

      if avalid_r = '1' and aready_in = '1' then
        avalid_r <= '0';
      end if;

      if rready_rr = '1' and rvalid_in = '1' then
        rready_rr <= '0';
        rdata_r   <= rdata_in;
      end if;

      if fu_glock = '0' then
        avalid_r <= avalid_in;
        aaddr_r  <= aaddr_in(aaddr_in'high downto low_bits_g);
        addr_low_r <= aaddr_in(low_bits_g-1 downto 0);
        addr_low_rr <= addr_low_r(low_bits_g-1 downto 0);
        awren_r  <= awren_in;
        astrb_r  <= astrb_in;
        adata_r  <= adata_in;

        if avalid_in = '1' and awren_in = '0' then
            rready_r <= '1';
        else
            rready_r <= '0';
        end if;

        if rready_r = '1' then
          rready_rr <= '1';
        end if;

      end if;
    end if;
  end process access_channel_sync;

  access_channel_comb : process(rready_rr, rvalid_in, rdata_r, rdata_in)
  begin
    if (rready_rr = '1' and rvalid_in = '1') then
      rdata_out <= rdata_in;
    else
      rdata_out <= rdata_r;
    end if;
  end process access_channel_comb;

end rtl;
