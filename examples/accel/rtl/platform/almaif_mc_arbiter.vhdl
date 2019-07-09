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
-- Title      : AlmaIF memory arbiter between multicore TTAs
-- Project    : Almarvi
-------------------------------------------------------------------------------
-- File       : almaif_mc_arbiter.vhdl
-- Author     : Aleksi Tervo  <aleksi.tervo@tut.fi>
-- Company    : TUT/CPC
-- Created    : 2016-11-22
-- Last update: 2016-11-22
-- Platform   :
-- Standard   : VHDL'87
-------------------------------------------------------------------------------
-- Description: Acts as a memory arbiter for multicore TTAs
--              Static arbiter: lower core ID has precedence
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2016-11-22  1.0      tervoa  Created
-- 2017-04-26  1.1      tervoa  Sensitivity list fix
-- 2017-06-01  1.2      tervoa  Converted to memory buses with handshaking
-- 2017-06-27  1.3      tervoa  Split arbiter in two parts: between TTA cores
--                              and between the multicore TTA and AXI
-- 2018-07-30  1.4      tervoa   Support for optional sync reset
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.tce_util.all;

entity almaif_mc_arbiter is

  generic (
    mem_dataw_g  : integer;
    mem_addrw_g  : integer;
    core_count_g : integer;
    sync_reset_g : integer
  ); port (
    clk                : in std_logic;
    rstx               : in std_logic;
    tta_sync_nreset_in : in std_logic_vector(core_count_g-1 downto 0);
    -- Buses to cores
    tta_avalid_in  : in  std_logic_vector(core_count_g-1 downto 0);
    tta_aready_out : out std_logic_vector(core_count_g-1 downto 0);
    tta_aaddr_in   : in  std_logic_vector(core_count_g*mem_addrw_g-1 downto 0);
    tta_awren_in   : in  std_logic_vector(core_count_g-1 downto 0);
    tta_astrb_in   : in std_logic_vector(core_count_g*mem_dataw_g/8-1 downto 0);
    tta_adata_in   : in  std_logic_vector(core_count_g*mem_dataw_g-1 downto 0);
    tta_rvalid_out : out std_logic_vector(core_count_g-1 downto 0);
    tta_rready_in  : in  std_logic_vector(core_count_g-1 downto 0);
    tta_rdata_out  : out std_logic_vector(core_count_g*mem_dataw_g-1 downto 0);
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
end almaif_mc_arbiter;

architecture rtl of almaif_mc_arbiter is

  constant sync_reset_c : boolean := sync_reset_g /= 0;

  constant mem_bytes_c       : integer := mem_dataw_g/8;
  constant fifo_depth_log2_c : integer := 2;
  constant core_count_log2_c : integer := bit_width(core_count_g);

  -- Breakout signals to arrays corewise
  type addr_array_t is array (natural range 0 to core_count_g-1)
                       of std_logic_vector(mem_addrw_g-1 downto 0);
  type data_array_t is array (natural range 0 to core_count_g-1)
                       of std_logic_vector(mem_dataw_g-1 downto 0);
  type mask_array_t is array (natural range 0 to core_count_g-1)
                       of std_logic_vector(mem_dataw_g/8-1 downto 0);
  signal cores_aaddr : addr_array_t;
  signal cores_astrb : mask_array_t;
  signal cores_adata : data_array_t;
  signal cores_rdata : data_array_t;

  -- FIFO implemented as a shifted register array
  type fifo_array_t is array (natural range <>)
                       of std_logic_vector(core_count_log2_c-1 downto 0);
  signal fifo_data_r : fifo_array_t(2**fifo_depth_log2_c-1 downto 0);
  signal fifo_iter_r : unsigned(fifo_depth_log2_c-1 downto 0);

  signal dst_sel         : std_logic_vector(core_count_log2_c-1 downto 0);
  signal fifo_dst_sel    : std_logic_vector(core_count_log2_c-1 downto 0);
  signal mem_awren       : std_logic;
  signal mem_avalid      : std_logic;
  signal mem_rready      : std_logic;
begin
  ------------------------------------------------------------------------------
  -- Memory interface logic: breakout the concatenated memory buses to arrays
  ------------------------------------------------------------------------------
  signal_breakout : for I in core_count_g-1 downto 0 generate
    cores_aaddr(I) <= tta_aaddr_in(mem_addrw_g*(I+1)-1 downto I*mem_addrw_g);
    cores_astrb(I) <= tta_astrb_in(mem_bytes_c*(I+1)-1 downto I*mem_bytes_c);
    cores_adata(I) <= tta_adata_in(mem_dataw_g*(I+1)-1 downto I*mem_dataw_g);

    tta_rdata_out((I+1)*mem_dataw_g-1 downto I*mem_dataw_g) <= mem_rdata_in;
  end generate;

  arbit_logic : process (tta_avalid_in)
  variable active_core     : integer range 0 to core_count_g-1;
  variable active_core_slv : std_logic_vector(core_count_log2_c-1 downto 0);
  begin
    active_core  := 0;
    for I in core_count_g-1  downto 0 loop
      if (tta_avalid_in(I) = '1') then
        active_core  := I;
      end if;
    end loop;
    dst_sel <= std_logic_vector(to_unsigned(active_core, core_count_log2_c));
  end process arbit_logic;


  ------------------------------------------------------------------------------
  -- Access channel mux:
  ------------------------------------------------------------------------------
  amux : process (mem_aready_in, dst_sel, tta_avalid_in, tta_awren_in,
                  cores_aaddr, cores_adata, cores_astrb)
    variable core_sel_int : integer;
  begin
      core_sel_int                 := to_integer(unsigned(dst_sel));
      mem_avalid                   <= tta_avalid_in(core_sel_int);
      tta_aready_out               <= (others => '0');
      tta_aready_out(core_sel_int) <= mem_aready_in;
      mem_awren                    <= tta_awren_in(core_sel_int);
      mem_aaddr_out                <= cores_aaddr(core_sel_int);
      mem_adata_out                <= cores_adata(core_sel_int);
      mem_astrb_out                <= cores_astrb(core_sel_int);
  end process amux;
  mem_avalid_out <= mem_avalid;
  mem_awren_out  <= mem_awren;

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
        if mem_rvalid_in = '1' and mem_rready = '1' and fifo_iter_r > 0 then
          fifo_data_v(fifo_data_v'high-1 downto 0) :=
              fifo_data_v(fifo_data_v'high downto 1);
          fifo_data_v(fifo_data_v'high) := (others => '0');
          fifo_iter_v := fifo_iter_v - 1;
        end if;

        if mem_avalid = '1' and mem_aready_in = '1' and mem_awren = '0' then
          fifo_data_v(to_integer(fifo_iter_v)) := dst_sel;
          fifo_iter_v := fifo_iter_v + 1;
        end if;

        fifo_iter_r <= fifo_iter_v;
        fifo_data_r <= fifo_data_v;
      end if;

    end if;
  end process fifo_sync;
  fifo_dst_sel <= fifo_data_r(0);

  ------------------------------------------------------------------------------
  -- Response channel mux:
  -- TODO: Handle reset better; works with local memory but will cause issues
  --       w/ axi
  ------------------------------------------------------------------------------
  rmux : process (fifo_dst_sel, mem_rvalid_in, tta_rready_in,
                  tta_sync_nreset_in)
    variable dst_sel_int : integer;
  begin
    dst_sel_int      := to_integer(unsigned(
                                fifo_dst_sel(core_count_log2_c-1 downto 0)));
    -- If the core is in reset, discard reads to avoid possible deadlock
    mem_rready     <=        tta_rready_in(dst_sel_int)
                      or not tta_sync_nreset_in(dst_sel_int);
    tta_rvalid_out <= (others => '0');
    tta_rvalid_out(dst_sel_int) <= mem_rvalid_in;
  end process rmux;
  cores_rdata    <= (others => mem_rdata_in);
  mem_rready_out <= mem_rready;


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
