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
-- Title      : Xilinx BRAM model with handshaking
-- Project    :
-------------------------------------------------------------------------------
-- File       : xilinx_blockram.vhdl
-- Author     : Aleksi Tervo
-- Company    : Tampere University
-- Created    : 2017-06-01
-- Last update: 2017-06-01
-- Platform   :
-- Standard   : VHDL'93
-------------------------------------------------------------------------------
-- Description: Parametric-width byte strobe memory with handshaking
--              which infers BRAM on (at least) Xilinx Series 7 FPGAs
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author    Description
-- 2017-06-01  1.0      tervoa    Created
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use std.textio.all;
use ieee.numeric_std.all;

entity xilinx_blockram is generic (
    addrw_g : integer := 10;
    dataw_g : integer := 32);
  port (
    clk        : in  std_logic;
    rstx       : in  std_logic;
    -- Access channel
    avalid_in  : in  std_logic;
    aready_out : out std_logic;
    aaddr_in   : in  std_logic_vector(addrw_g-1 downto 0);
    awren_in   : in  std_logic;
    astrb_in   : in  std_logic_vector((dataw_g+7)/8-1 downto 0);
    adata_in   : in  std_logic_vector(dataw_g-1 downto 0);
    -- Read channel
    rvalid_out : out std_logic;
    rready_in  : in  std_logic;
    rdata_out  : out std_logic_vector(dataw_g-1 downto 0)
  );
end xilinx_blockram;

architecture rtl of xilinx_blockram is
constant dataw_padded_c  : integer := ((dataw_g+7)/8)*8;
constant astrb_width_c   : integer := (dataw_g+7)/8;

signal ram_addr        : unsigned(addrw_g-1 downto 0);
signal ram_write_data  : std_logic_vector(dataw_padded_c-1 downto 0);
signal ram_read_data_r : std_logic_vector(dataw_padded_c-1 downto 0);
signal ram_enable      : std_logic;
signal ram_strb        : std_logic_vector(astrb_width_c-1 downto 0);

constant adata_padding   : std_logic_vector(dataw_padded_c-dataw_g-1 downto 0)
                          := (others => '0');
signal adata_padded      : std_logic_vector(dataw_padded_c-1 downto 0);
signal aready_r          : std_logic;
signal live_read         : std_logic;
signal live_read_r       : std_logic;
signal read_data_r       : std_logic_vector(dataw_padded_c-1 downto 0);
signal read_data_valid_r : std_logic;
signal rvalid            : std_logic;

type ram_type is array (2**addrw_g-1 downto 0) of std_logic_vector
                                              (dataw_padded_c-1 downto 0);
signal RAM_ARR : ram_type;

begin

control_comb : process(aaddr_in, avalid_in, aready_r, awren_in, astrb_in,
                       live_read_r, read_data_valid_r)
begin
  if avalid_in = '1' and aready_r = '1' then
    ram_enable <= '1';
    if awren_in = '1' then
      ram_strb <= astrb_in;
      live_read <= '0';
    else
      ram_strb  <= (others => '0');
      live_read <= '1';
    end if;
  else
    ram_strb       <= (others => '0');
    ram_enable     <= '0';
    live_read      <= '0';
  end if;

  ram_addr <= unsigned(aaddr_in);
  rvalid   <= live_read_r or read_data_valid_r;
end process;


control_sync : process(clk, rstx)
begin
  if rstx = '0' then
    live_read_r       <= '0';
    aready_r          <= '0';
    read_data_valid_r <= '0';
    read_data_r       <= (others => '0');
  elsif rising_edge(clk) then
    if rvalid = '1' and rready_in = '1' then
      read_data_valid_r <= '0';
    end if;

    if rvalid = '1' and rready_in = '0' then
      aready_r <= '0';
    else
      aready_r <= '1';
    end if;

    live_read_r <= live_read or live_read_r;
    if live_read_r = '1' and (rready_in = '1' or read_data_valid_r = '0') then
      live_read_r       <= live_read;
      if rready_in = '0' or read_data_valid_r = '1' then
        read_data_valid_r <= '1';
        read_data_r       <= ram_read_data_r;
      end if;
    end if;
  end if;
end process;

adata_padded <= adata_padding & adata_in;

RAM : process(clk)
begin
    if rising_edge(clk) then
        if ram_enable = '1' then
            for i in 0 to astrb_width_c-1 loop
                if ram_strb(i) = '1' then
                    RAM_ARR(to_integer(ram_addr))((i+1)*8-1 downto i*8)
                                      <= adata_padded((i+1)*8-1 downto i*8);
                end if;
            end loop;
            ram_read_data_r <= RAM_ARR(to_integer(ram_addr));
        end if;
    end if;
end process;

rdata_out <= ram_read_data_r(rdata_out'range) when read_data_valid_r = '0'
                                              else read_data_r(rdata_out'range);
rvalid_out <= rvalid;
aready_out <= aready_r;
end rtl;
