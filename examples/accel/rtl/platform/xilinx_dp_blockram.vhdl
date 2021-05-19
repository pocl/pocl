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
-- Title      : Xilinx dual-port BRAM model with handshaking
-- Project    :
-------------------------------------------------------------------------------
-- File       : xilinx_do_blockram.vhdl
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

entity xilinx_dp_blockram is generic (
    addrw_g : integer := 10;
    dataw_g : integer := 32;
    addrw_b_g : integer := 10;
    dataw_b_g : integer := 32);
  port (
    clk        : in  std_logic;
    rstx       : in  std_logic;
    -- PORT A -------------------------------------------------------
    -- Access channel
    a_avalid_in  : in  std_logic;
    a_aready_out : out std_logic;
    a_aaddr_in   : in  std_logic_vector(addrw_g-1 downto 0);
    a_awren_in   : in  std_logic;
    a_astrb_in   : in  std_logic_vector((dataw_g+7)/8-1 downto 0);
    a_adata_in   : in  std_logic_vector(dataw_g-1 downto 0);
    -- Read channel
    a_rvalid_out : out std_logic;
    a_rready_in  : in  std_logic;
    a_rdata_out  : out std_logic_vector(dataw_g-1 downto 0);
    -- PORT B -------------------------------------------------------
    -- Access channel
    b_avalid_in  : in  std_logic;
    b_aready_out : out std_logic;
    b_aaddr_in   : in  std_logic_vector(addrw_b_g-1 downto 0);
    b_awren_in   : in  std_logic;
    b_astrb_in   : in  std_logic_vector((dataw_b_g+7)/8-1 downto 0);
    b_adata_in   : in  std_logic_vector(dataw_b_g-1 downto 0);
    -- Read channel
    b_rvalid_out : out std_logic;
    b_rready_in  : in  std_logic;
    b_rdata_out  : out std_logic_vector(dataw_b_g-1 downto 0)
  );
end xilinx_dp_blockram;

architecture rtl of xilinx_dp_blockram is
  constant b_astrb_width_c : integer := dataw_b_g/8;
  constant a_dataw_padded_c  : integer := (dataw_g+dataw_b_g-1)/dataw_b_g*dataw_b_g;
  constant a_astrb_width_c : integer := a_dataw_padded_c/8;
  constant a_padding_c : std_logic_vector(a_dataw_padded_c-dataw_g-1 downto 0)
                           := (others => '0');
  constant b_addr_padding_c : integer := addrw_b_g-addrw_g;
  constant need_select_c     : boolean := b_addr_padding_c > 0;

  constant a_astrb_padding_c : std_logic_vector(a_astrb_width_c-a_astrb_in'high-2
                                                downto 0) := (others => '0');
  constant a_word_count_c : integer := a_dataw_padded_c/dataw_b_g;

  signal a_addr : unsigned(addrw_g-1 downto 0);
  signal b_addr : unsigned(addrw_g-1 downto 0);
  signal b_select, b_select_r : unsigned(b_addr_padding_c-1 downto 0);
  signal a_wdata, a_adata, a_rdata_r, a_ram_rdata_r : std_logic_vector(a_dataw_padded_c-1 downto 0);
  signal b_wdata, b_adata, b_ram_rdata_r : std_logic_vector(a_dataw_padded_c-1 downto 0);
  signal b_rdata_sel, b_rdata_r : std_logic_vector(dataw_b_g-1 downto 0);
  signal a_enable, b_enable                : std_logic;
  signal a_aready_r, b_aready_r            : std_logic;
  signal a_live_read, b_live_read          : std_logic;
  signal a_live_read_r, b_live_read_r      : std_logic;
  signal a_rdata_valid_r, b_rdata_valid_r  : std_logic;
  signal a_rvalid, b_rvalid                : std_logic;
  signal a_astrb, b_astrb                  : std_logic_vector(a_astrb_width_c-1 downto 0);

  type ram_type is array (2**addrw_g-1 downto 0) of std_logic_vector
                                                (a_dataw_padded_c-1 downto 0);
  shared variable RAM_ARR : ram_type;

begin

  control_comb_a : process(a_aaddr_in, a_avalid_in, a_aready_r,
                           a_awren_in, a_live_read_r, a_rdata_valid_r)
  begin
    if a_avalid_in = '1' and a_aready_r = '1' then
      a_enable <= '1';
      a_live_read <= not a_awren_in;
    else
      a_enable    <= '0';
      a_live_read <= '0';
    end if;

    a_addr   <= unsigned(a_aaddr_in);
    a_rvalid <= a_live_read_r or a_rdata_valid_r;
  end process;

  control_comb_b : process(b_aaddr_in, b_avalid_in, b_aready_r,
                           b_awren_in, b_live_read_r, b_rdata_valid_r,
                           b_astrb_in, b_select, b_adata_in)
  begin
    if b_avalid_in = '1' and b_aready_r = '1' then
      b_enable <= '1';
      b_live_read <= not b_awren_in;
    else
      b_enable    <= '0';
      b_live_read <= '0';
    end if;

    b_addr   <= unsigned(b_aaddr_in(b_aaddr_in'high downto b_addr_padding_c));
    b_select <= unsigned(b_aaddr_in(b_addr_padding_c-1 downto 0));
    b_astrb <= (others => '0');

    if need_select_c then
      b_astrb((to_integer(b_select)+1)*b_astrb_width_c-1 downto to_integer(b_select)*b_astrb_width_c) <= b_astrb_in;
    else
      b_astrb(b_astrb_width_c-1 downto 0) <= b_astrb_in;
    end if;

    for i in 0 to a_word_count_c - 1 loop
      b_wdata((i+1)*dataw_b_g-1 downto i*dataw_b_g) <= b_adata_in;
    end loop;
    b_rvalid <= b_live_read_r or b_rdata_valid_r;
  end process;


  control_sync_a : process(clk, rstx)
  begin
    if rstx = '0' then
      a_live_read_r   <= '0';
      a_aready_r      <= '0';
      a_rdata_valid_r <= '0';
      a_rdata_r       <= (others => '0');
    elsif rising_edge(clk) then
      if a_rvalid = '1' and a_rready_in = '1' then
        a_rdata_valid_r <= '0';
      end if;

      if a_rvalid = '1' and a_rready_in = '0' then
        a_aready_r <= '0';
      else
        a_aready_r <= '1';
      end if;

      a_live_read_r <= a_live_read or a_live_read_r;
      if a_live_read_r = '1' and (a_rready_in = '1' or a_rdata_valid_r = '0') then
        a_live_read_r       <= a_live_read;
        if a_rready_in = '0' or a_rdata_valid_r = '1' then
          a_rdata_valid_r <= '1';
          a_rdata_r       <= a_ram_rdata_r;
        end if;
      end if;
    end if;
  end process;

  control_sync_b : process(clk, rstx)
  begin
    if rstx = '0' then
      b_live_read_r   <= '0';
      b_aready_r      <= '0';
      b_rdata_valid_r <= '0';
      b_rdata_r       <= (others => '0');
      b_select_r      <= (others => '0');
    elsif rising_edge(clk) then

      if b_live_read = '1' then
        b_select_r <= b_select;
      end if;

      if b_rvalid = '1' and b_rready_in = '1' then
        b_rdata_valid_r <= '0';
      end if;

      if b_rvalid = '1' and b_rready_in = '0' then
        b_aready_r <= '0';
      else
        b_aready_r <= '1';
      end if;

      b_live_read_r <= b_live_read or b_live_read_r;
      if b_live_read_r = '1' and (b_rready_in = '1' or b_rdata_valid_r = '0') then
        b_live_read_r       <= b_live_read;
        if b_rready_in = '0' or b_rdata_valid_r = '1' then
          b_rdata_valid_r <= '1';
          b_rdata_r       <= b_rdata_sel;
        end if;
      end if;
    end if;
  end process;

  a_wdata <= a_padding_c & a_adata_in;
  a_astrb <= a_astrb_padding_c & a_astrb_in;

  RAM_A : process(clk)
  begin
      if rising_edge(clk) then
          if a_enable = '1' then
              if a_awren_in = '1' then
                  for b in 0 to a_astrb_width_c-1 loop
                    if a_astrb(b) = '1' then
                        RAM_ARR(to_integer(a_addr))((b+1)*8-1 downto b*8)
                              := a_wdata((b+1)*8-1 downto b*8);
                    end if;
                end loop;
              else
                a_ram_rdata_r <= RAM_ARR(to_integer(a_addr));
              end if;
          end if;
      end if;
  end process;

  RAM_B : process(clk)
  begin
      if rising_edge(clk) then
          if b_enable = '1' then
            if b_awren_in = '1' then
              for i in 0 to a_astrb_width_c-1 loop
                  if b_astrb(i) = '1' then
                      RAM_ARR(to_integer(b_addr))((i+1)*8-1 downto i*8)
                                        := b_wdata((i+1)*8-1 downto i*8);
                  end if;
              end loop;
            else
              b_ram_rdata_r <= RAM_ARR(to_integer(b_addr));
            end if;
          end if;
      end if;
  end process;

  pad: if need_select_c generate
    b_rdata_sel <= b_ram_rdata_r((to_integer(b_select_r)+1)*dataw_b_g-1
                           downto to_integer(b_select_r)*dataw_b_g);
  end generate;
  no_pad: if not need_select_c generate
    b_rdata_sel <= b_ram_rdata_r(dataw_b_g-1 downto 0);
  end generate;
  a_rdata_out <= a_ram_rdata_r(a_rdata_out'range) when a_rdata_valid_r = '0'
                                              else a_rdata_r(a_rdata_out'range);
  b_rdata_out <= b_rdata_sel when b_rdata_valid_r = '0' else b_rdata_r;
  a_aready_out <= a_aready_r;
  a_rvalid_out <= a_rvalid;
  b_aready_out <= b_aready_r;
  b_rvalid_out <= b_rvalid;

end architecture rtl;
