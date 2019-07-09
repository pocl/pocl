-- Copyright (c) 2016-2017 Tampere University
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
-----------------------------------------------------------------------------
-- Title      : AXI interface for AlmaIF wrapper
-- Project    : Almarvi
-------------------------------------------------------------------------------
-- File       : tta-accel-rtl.vhdl
-- Author     : Viitanen Timo (Tampere University)  <timo.2.viitanen@tut.fi>
-- Company    : 
-- Created    : 2016-01-27
-- Last update: 2017-03-27
-- Platform   : 
-- Standard   : VHDL'93
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2016-01-27  1.0      viitanet  Created
-- 2016-11-18  1.1      tervoa   Added full AXI4 interface
-- 2017-03-27  1.2      tervoa   Change to axislave interface
-- 2017-04-25  1.3      tervoa   Merge entity and architecture, use generics
--                               instead of consts from packages
-- 2017-06-01  1.4      tervoa   Convert to memory buses with handshaking
-- 2018-07-30  1.5      tervoa   Support for optional sync reset
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.tce_util.all;

entity tta_accel is
  generic (
    core_count_g       : integer;
    axi_addr_width_g   : integer;
    axi_id_width_g     : integer;
    imem_data_width_g  : integer;
    imem_addr_width_g  : integer;
    dmem_data_width_g  : integer;
    dmem_addr_width_g  : integer;
    pmem_data_width_g  : integer;
    pmem_addr_width_g  : integer;
    bus_count_g        : integer;
    local_mem_addrw_g  : integer;
    axi_offset_g       : integer := 0;
    full_debugger_g    : integer;
    sync_reset_g       : integer
  ); port (
    clk            : in std_logic;
    rstx           : in std_logic;
    s_axi_awaddr   : in std_logic_vector(axi_addr_width_g-1 downto 0);
    s_axi_awvalid  : in std_logic;
    s_axi_awready  : out std_logic;
    s_axi_wdata    : in std_logic_vector(32-1 downto 0);
    s_axi_wstrb    : in std_logic_vector(4-1 downto 0);
    s_axi_wvalid   : in std_logic;
    s_axi_wready   : out std_logic;
    s_axi_bresp    : out std_logic_vector(2-1 downto 0);
    s_axi_bvalid   : out std_logic;
    s_axi_bready   : in std_logic;
    s_axi_araddr   : in std_logic_vector(axi_addr_width_g-1 downto 0);
    s_axi_arvalid  : in std_logic;
    s_axi_arready  : out std_logic;
    s_axi_rdata    : out std_logic_vector(32-1 downto 0);
    s_axi_rresp    : out std_logic_vector(2-1 downto 0);
    s_axi_rvalid   : out std_logic;
    s_axi_rready   : in std_logic;
    s_axi_awid     : in  std_logic_vector (axi_id_width_g-1 downto 0);
    s_axi_awlen    : in  std_logic_vector (8-1 downto 0);
    s_axi_awsize   : in  std_logic_vector (3-1 downto 0);
    s_axi_awburst  : in  std_logic_vector (2-1 downto 0);
    s_axi_bid      : out std_logic_vector (axi_id_width_g-1 downto 0);
    s_axi_arid     : in  std_logic_vector (axi_id_width_g-1 downto 0);
    s_axi_arlen    : in  std_logic_vector (8-1 downto 0);
    s_axi_arsize   : in  std_logic_vector (3-1 downto 0);
    s_axi_arburst  : in  std_logic_vector (2-1 downto 0);
    s_axi_rid      : out std_logic_vector (axi_id_width_g-1 downto 0);
    s_axi_rlast    : out std_logic;

    -- AXI4-Lite Master for global mem
    m_axi_awvalid  : out std_logic;
    m_axi_awready  : in  std_logic;
    m_axi_awaddr   : out std_logic_vector(32-1 downto 0);
    m_axi_awprot   : out std_logic_vector(3-1 downto 0);
    --
    m_axi_wvalid   : out std_logic;
    m_axi_wready   : in  std_logic;
    m_axi_wdata    : out std_logic_vector(dmem_data_width_g-1 downto 0);
    m_axi_wstrb    : out std_logic_vector(dmem_data_width_g/8-1 downto 0);
    --
    m_axi_bvalid   : in  std_logic;
    m_axi_bready   : out std_logic;
    --
    m_axi_arvalid  : out std_logic;
    m_axi_arready  : in  std_logic;
    m_axi_araddr   : out std_logic_vector(32-1 downto 0);
    m_axi_arprot   : out std_logic_vector(3-1 downto 0);
    --
    m_axi_rvalid   : in  std_logic;
    m_axi_rready   : out std_logic;
    m_axi_rdata    : in  std_logic_vector(dmem_data_width_g-1 downto 0);

    

    aql_read_idx_in   : in std_logic_vector(64-1 downto 0);
    aql_read_idx_clear_out : out std_logic_vector(0 downto 0);
    

    core_dmem_avalid_in  : in std_logic_vector(core_count_g-1 downto 0);
    core_dmem_aready_out : out std_logic_vector(core_count_g-1 downto 0);
    core_dmem_aaddr_in   : in std_logic_vector(core_count_g*dmem_addr_width_g-1
                                               downto 0);
    core_dmem_awren_in : in std_logic_vector(core_count_g-1 downto 0);
    core_dmem_astrb_in : in std_logic_vector((dmem_data_width_g+7)/8*core_count_g-1
                                             downto 0);
    core_dmem_adata_in : in std_logic_vector(core_count_g*dmem_data_width_g-1
                                             downto 0);
    core_dmem_rvalid_out : out std_logic_vector(core_count_g-1 downto 0);
    core_dmem_rready_in  : in std_logic_vector(core_count_g-1 downto 0);
    core_dmem_rdata_out  : out std_logic_vector(core_count_g*dmem_data_width_g-1
                                                downto 0);
    
    data_a_avalid_out : out std_logic_vector(1-1 downto 0);
    data_a_aready_in  : in std_logic_vector(1-1 downto 0);
    data_a_aaddr_out  : out std_logic_vector(dmem_addr_width_g-1 downto 0);
    data_a_awren_out  : out std_logic_vector(1-1 downto 0);
    data_a_astrb_out  : out std_logic_vector((dmem_data_width_g+7)/8-1 downto 0);
    data_a_adata_out  : out std_logic_vector(dmem_data_width_g-1 downto 0);
    data_a_rvalid_in  : in std_logic_vector(1-1 downto 0);
    data_a_rready_out : out std_logic_vector(1-1 downto 0);
    data_a_rdata_in   : in std_logic_vector(dmem_data_width_g-1 downto 0);
    
    data_b_avalid_out : out std_logic_vector(1-1 downto 0);
    data_b_aready_in  : in std_logic_vector(1-1 downto 0);
    data_b_aaddr_out  : out std_logic_vector(dmem_addr_width_g-1 downto 0);
    data_b_awren_out  : out std_logic_vector(1-1 downto 0);
    data_b_astrb_out  : out std_logic_vector((dmem_data_width_g+7)/8-1 downto 0);
    data_b_adata_out  : out std_logic_vector(dmem_data_width_g-1 downto 0);
    data_b_rvalid_in  : in std_logic_vector(1-1 downto 0);
    data_b_rready_out : out std_logic_vector(1-1 downto 0);
    data_b_rdata_in   : in std_logic_vector(dmem_data_width_g-1 downto 0);
    core_pmem_avalid_in  : in std_logic_vector(core_count_g-1 downto 0);
    core_pmem_aready_out : out std_logic_vector(core_count_g-1 downto 0);
    core_pmem_aaddr_in   : in std_logic_vector(core_count_g*pmem_addr_width_g-1
                                               downto 0);
    core_pmem_awren_in : in std_logic_vector(core_count_g-1 downto 0);
    core_pmem_astrb_in : in std_logic_vector((pmem_data_width_g+7)/8*core_count_g-1
                                             downto 0);
    core_pmem_adata_in : in std_logic_vector(core_count_g*pmem_data_width_g-1
                                             downto 0);
    core_pmem_rvalid_out : out std_logic_vector(core_count_g-1 downto 0);
    core_pmem_rready_in  : in std_logic_vector(core_count_g-1 downto 0);
    core_pmem_rdata_out  : out std_logic_vector(core_count_g*pmem_data_width_g-1
                                                downto 0);
    
    param_a_avalid_out : out std_logic_vector(1-1 downto 0);
    param_a_aready_in  : in std_logic_vector(1-1 downto 0);
    param_a_aaddr_out  : out std_logic_vector(local_mem_addrw_g-1 downto 0);
    param_a_awren_out  : out std_logic_vector(1-1 downto 0);
    param_a_astrb_out  : out std_logic_vector((pmem_data_width_g+7)/8-1 downto 0);
    param_a_adata_out  : out std_logic_vector(pmem_data_width_g-1 downto 0);
    param_a_rvalid_in  : in std_logic_vector(1-1 downto 0);
    param_a_rready_out : out std_logic_vector(1-1 downto 0);
    param_a_rdata_in   : in std_logic_vector(pmem_data_width_g-1 downto 0);
    
    param_b_avalid_out : out std_logic_vector(1-1 downto 0);
    param_b_aready_in  : in std_logic_vector(1-1 downto 0);
    param_b_aaddr_out  : out std_logic_vector(local_mem_addrw_g-1 downto 0);
    param_b_awren_out  : out std_logic_vector(1-1 downto 0);
    param_b_astrb_out  : out std_logic_vector((pmem_data_width_g+7)/8-1 downto 0);
    param_b_adata_out  : out std_logic_vector(pmem_data_width_g-1 downto 0);
    param_b_rvalid_in  : in std_logic_vector(1-1 downto 0);
    param_b_rready_out : out std_logic_vector(1-1 downto 0);
    param_b_rdata_in   : in std_logic_vector(pmem_data_width_g-1 downto 0);
    

    -- Debug ports
    core_db_tta_nreset : out std_logic_vector(core_count_g-1 downto 0);
    core_db_lockrq     : out std_logic_vector(core_count_g-1 downto 0);
    core_db_pc         : in std_logic_vector(core_count_g*imem_addr_width_g-1
                                             downto 0);
    core_db_lockcnt    : in std_logic_vector(core_count_g*64-1 downto 0);
    core_db_cyclecnt   : in std_logic_vector(core_count_g*64-1 downto 0)
    );
end entity tta_accel;

architecture rtl of tta_accel is
  constant dataw_c           : integer := 32;
  constant ctrl_addr_width_c : integer := 8;
  constant dbg_core_sel_width_c : integer := bit_width(core_count_g);

  constant imem_byte_sel_width_c : integer := bit_width(imem_data_width_g/8);
  constant dmem_byte_sel_width_c : integer := bit_width(dmem_data_width_g/8);
  constant pmem_byte_sel_width_c : integer := bit_width(pmem_data_width_g/8);

  constant pmem_offset_c : integer := axi_offset_g + 2**(axi_addr_width_g-2)*3;

  constant enable_dmem : boolean := dmem_data_width_g > 0;
  constant enable_pmem : boolean := pmem_data_width_g > 0;

  -- AXI slave memory bus
  signal axi_avalid : std_logic;
  signal axi_aready : std_logic;
  signal axi_aaddr  : std_logic_vector(axi_addr_width_g-2-1 downto 0);
  signal axi_awren  : std_logic;
  signal axi_astrb  : std_logic_vector(dataw_c/8-1 downto 0);
  signal axi_adata  : std_logic_vector(dataw_c-1 downto 0);
  signal axi_rvalid : std_logic;
  signal axi_rready : std_logic;
  signal axi_rdata  : std_logic_vector(dataw_c-1 downto 0);

  signal dmem_avalid   : std_logic;
  signal dmem_aready   : std_logic;
  signal dmem_rvalid   : std_logic;
  signal dmem_rready   : std_logic;
  signal dmem_rdata    : std_logic_vector(dataw_c-1 downto 0);

  signal ctrl_avalid   : std_logic;
  signal ctrl_aready   : std_logic;
  signal ctrl_rvalid   : std_logic;
  signal ctrl_rready   : std_logic;
  signal ctrl_rdata    : std_logic_vector(core_count_g*dataw_c-1 downto 0);

  signal pmem_avalid   : std_logic;
  signal pmem_aready   : std_logic;
  signal pmem_rvalid   : std_logic;
  signal pmem_rready   : std_logic;
  signal pmem_rdata    : std_logic_vector(dataw_c-1 downto 0);

  signal imem_avalid   : std_logic;
  signal imem_aready   : std_logic;
  signal imem_rvalid   : std_logic;
  signal imem_rready   : std_logic;
  signal imem_rdata    : std_logic_vector(dataw_c-1 downto 0);

  signal core_busy       : std_logic_vector(core_count_g-1 downto 0);
  signal tta_sync_nreset : std_logic_vector(core_count_g-1 downto 0);

  signal ctrl_en       : std_logic;
  signal ctrl_data     : std_logic_vector(dataw_c-1 downto 0);
  signal ctrl_core_sel : std_logic_vector(bit_width(core_count_g)-1 downto 0);

  signal mc_arb_pmem_avalid : std_logic;
  signal mc_arb_pmem_aready : std_logic;
  signal mc_arb_pmem_aaddr  : std_logic_vector(pmem_addr_width_g-1 downto 0);
  signal mc_arb_pmem_awren  : std_logic;
  signal mc_arb_pmem_astrb  : std_logic_vector((pmem_data_width_g+7)/8-1 downto 0);
  signal mc_arb_pmem_adata  : std_logic_vector(pmem_data_width_g-1 downto 0);
  signal mc_arb_pmem_rvalid : std_logic;
  signal mc_arb_pmem_rready : std_logic;
  signal mc_arb_pmem_rdata  : std_logic_vector(pmem_data_width_g-1 downto 0);

  signal axi_imem_avalid_out : std_logic;
  signal axi_imem_aready_in  : std_logic;
  signal axi_imem_aaddr_out  : std_logic_vector(imem_addr_width_g-1 downto 0);
  signal axi_imem_awren_out  : std_logic;
  signal axi_imem_astrb_out  : std_logic_vector((imem_data_width_g+7)/8-1 downto 0);
  signal axi_imem_adata_out  : std_logic_vector(imem_data_width_g-1 downto 0);
  signal axi_imem_rvalid_in  : std_logic;
  signal axi_imem_rready_out : std_logic;
  signal axi_imem_rdata_in   : std_logic_vector(imem_data_width_g-1 downto 0);

  signal tta_aready      : std_logic_vector(core_count_g-1 downto 0);
  signal tta_rvalid      : std_logic_vector(core_count_g-1 downto 0);

  signal aql_read_idx, aql_write_idx : std_logic_vector(64-1 downto 0);
  signal aql_read_idx_clear : std_logic_vector(0 downto 0);

  
  
  
  signal core_db_pc_start   : std_logic_vector(core_count_g*imem_addr_width_g-1 downto 0);
  signal core_db_instr      : std_logic_vector(core_count_g*imem_data_width_g-1 downto 0);
  signal core_db_pc_next    : std_logic_vector(core_count_g*imem_addr_width_g-1 downto 0);
  signal core_db_bustraces  : std_logic_vector(core_count_g*32*bus_count_g-1 downto 0);
  
begin

  -----------------------------------------------------------------------------
  -- AXI Controller
  -----------------------------------------------------------------------------
  tta_axislave_1 : entity work.tta_axislave
  generic map (
    axi_addrw_g => axi_addr_width_g,
    axi_idw_g   => axi_id_width_g,
    axi_dataw_g => dataw_c,
    sync_reset_g => sync_reset_g
  )
  port map (
    clk            => clk,
    rstx           => rstx,

    s_axi_awid     => s_axi_awid,
    s_axi_awaddr   => s_axi_awaddr,
    s_axi_awlen    => s_axi_awlen,
    s_axi_awsize   => s_axi_awsize,
    s_axi_awburst  => s_axi_awburst,
    s_axi_awvalid  => s_axi_awvalid,
    s_axi_awready  => s_axi_awready,
    s_axi_wdata    => s_axi_wdata,
    s_axi_wstrb    => s_axi_wstrb,
    s_axi_wvalid   => s_axi_wvalid,
    s_axi_wready   => s_axi_wready,
    s_axi_bid      => s_axi_bid,
    s_axi_bresp    => s_axi_bresp,
    s_axi_bvalid   => s_axi_bvalid,
    s_axi_bready   => s_axi_bready,
    s_axi_arid     => s_axi_arid,
    s_axi_araddr   => s_axi_araddr,
    s_axi_arlen    => s_axi_arlen,
    s_axi_arsize   => s_axi_arsize,
    s_axi_arburst  => s_axi_arburst,
    s_axi_arvalid  => s_axi_arvalid,
    s_axi_arready  => s_axi_arready,
    s_axi_rid      => s_axi_rid,
    s_axi_rdata    => s_axi_rdata,
    s_axi_rresp    => s_axi_rresp,
    s_axi_rlast    => s_axi_rlast,
    s_axi_rvalid   => s_axi_rvalid,
    s_axi_rready   => s_axi_rready,

    avalid_out     => axi_avalid,
    aready_in      => axi_aready,
    aaddr_out      => axi_aaddr,
    awren_out      => axi_awren,
    astrb_out      => axi_astrb,
    adata_out      => axi_adata,
    rvalid_in      => axi_rvalid,
    rready_out     => axi_rready,
    rdata_in       => axi_rdata
  );

  bus_splitter : entity work.membus_splitter
    generic map (
      core_count_g      => core_count_g,
      axi_addr_width_g  => axi_addr_width_g,
      axi_data_width_g  => dataw_c,
      ctrl_addr_width_g => ctrl_addr_width_c,
      imem_addr_width_g => imem_addr_width_g + imem_byte_sel_width_c,
      dmem_addr_width_g => dmem_addr_width_g + dmem_byte_sel_width_c,
      pmem_addr_width_g => pmem_addr_width_g + pmem_byte_sel_width_c
    ) port map (
      -- AXI slave
      avalid_in        => axi_avalid,
      aready_out       => axi_aready,
      aaddr_in         => axi_aaddr,
      rvalid_out       => axi_rvalid,
      rready_in        => axi_rready,
      rdata_out        => axi_rdata,
      -- Control signals to arbiters
      dmem_avalid_out  => dmem_avalid,
      dmem_aready_in   => dmem_aready,
      dmem_rvalid_in   => dmem_rvalid,
      dmem_rready_out  => dmem_rready,
      dmem_rdata_in    => dmem_rdata,

      pmem_avalid_out  => pmem_avalid,
      pmem_aready_in   => pmem_aready,
      pmem_rvalid_in   => pmem_rvalid,
      pmem_rready_out  => pmem_rready,
      pmem_rdata_in    => pmem_rdata,

      imem_avalid_out  => imem_avalid,
      imem_aready_in   => imem_aready,
      imem_rvalid_in   => imem_rvalid,
      imem_rready_out  => imem_rready,
      imem_rdata_in    => imem_rdata,

      -- Signals to debugger(s)
      ctrl_avalid_out   => ctrl_avalid,
      ctrl_aready_in    => ctrl_aready,
      ctrl_rvalid_in    => ctrl_rvalid,
      ctrl_rready_out   => ctrl_rready,
      ctrl_rdata_in     => ctrl_rdata,
      ctrl_core_sel_out => ctrl_core_sel
    );
  ------------------------------------------------------------------------------
  -- Debugger
  ------------------------------------------------------------------------------
  minidebug : entity work.minidebugger
    generic map (
      data_width_g      => 32,
      axi_addr_width_g  => axi_addr_width_g,
      core_count_g      => core_count_g,
      core_id_width_g   => dbg_core_sel_width_c,
      imem_data_width_g    => imem_data_width_g,
      imem_addr_width_g    => imem_addr_width_g,
      dmem_data_width_g    => dmem_data_width_g,
      dmem_addr_width_g    => dmem_addr_width_g,
      pmem_data_width_g    => pmem_data_width_g,
      pmem_addr_width_g    => local_mem_addrw_g
    ) port map (
      clk               => clk,
      rstx              => rstx,
  
      avalid_in         => ctrl_avalid,
      aready_out        => ctrl_aready,
      aaddr_in          => axi_aaddr,
      awren_in          => axi_awren,
      astrb_in          => axi_astrb,
      adata_in          => axi_adata,
      rvalid_out        => ctrl_rvalid,
      rready_in         => ctrl_rready,
      rdata_out         => ctrl_rdata(dataw_c-1 downto 0),
      core_sel_in       => ctrl_core_sel,
  
      tta_locked_in     => core_busy,
      tta_lockrq_out    => core_db_lockrq,
      tta_nreset_out    => core_db_tta_nreset,
      tta_pc_in         => core_db_pc,
      tta_lockcnt_in    => core_db_lockcnt,
      tta_cyclecnt_in   => core_db_cyclecnt,
      tta_read_idx_in   => aql_read_idx,
      tta_read_idx_clear_out => aql_read_idx_clear,
      tta_write_idx_out => aql_write_idx
    );
  
  rdata_broadcast : for I in 1 to core_count_g-1 generate
    ctrl_rdata((I+1)*dataw_c-1 downto I*dataw_c)
        <= ctrl_rdata(dataw_c-1 downto 0);
  end generate rdata_broadcast;
  

  aql_read_idx <= aql_read_idx_in;
  aql_read_idx_clear_out <= aql_read_idx_clear;
  

  ------------------------------------------------------------------------------
  -- Memory arbitration between IO and TTA
  ------------------------------------------------------------------------------

  mc_arbiters : if core_count_g > 1 generate
    gen_dmem_arbiter : if enable_dmem generate
      mc_arbiter_dmem : entity work.almaif_mc_arbiter
      generic map (
        mem_dataw_g  => dmem_data_width_g,
        mem_addrw_g  => dmem_addr_width_g,
        core_count_g => core_count_g,
        sync_reset_g => sync_reset_g
      ) port map (
        clk                => clk,
        rstx               => rstx,
        tta_sync_nreset_in => tta_sync_nreset,
        -- Buses to cores
        tta_avalid_in      => core_dmem_avalid_in,
        tta_aready_out     => core_dmem_aready_out,
        tta_aaddr_in       => core_dmem_aaddr_in,
        tta_awren_in       => core_dmem_awren_in,
        tta_astrb_in       => core_dmem_astrb_in,
        tta_adata_in       => core_dmem_adata_in,
        tta_rvalid_out     => core_dmem_rvalid_out,
        tta_rready_in      => core_dmem_rready_in,
        tta_rdata_out      => core_dmem_rdata_out,
        -- Bus to memory
        mem_avalid_out => data_a_avalid_out(0),
        mem_aready_in  => data_a_aready_in(0),
        mem_aaddr_out  => data_a_aaddr_out,
        mem_awren_out  => data_a_awren_out(0),
        mem_astrb_out  => data_a_astrb_out,
        mem_adata_out  => data_a_adata_out,
        mem_rvalid_in  => data_a_rvalid_in(0),
        mem_rready_out => data_a_rready_out(0),
        mem_rdata_in   => data_a_rdata_in
      );
    end generate;

    gen_pmem_arbiter : if enable_pmem generate
      mc_arbiter_pmem : entity work.almaif_mc_arbiter
      generic map (
        mem_dataw_g  => pmem_data_width_g,
        mem_addrw_g  => pmem_addr_width_g,
        core_count_g => core_count_g,
        sync_reset_g => sync_reset_g
      ) port map (
        clk                => clk,
        rstx               => rstx,
        tta_sync_nreset_in => tta_sync_nreset,
        -- Buses to cores
        tta_avalid_in      => core_pmem_avalid_in,
        tta_aready_out     => core_pmem_aready_out,
        tta_aaddr_in       => core_pmem_aaddr_in,
        tta_awren_in       => core_pmem_awren_in,
        tta_astrb_in       => core_pmem_astrb_in,
        tta_adata_in       => core_pmem_adata_in,
        tta_rvalid_out     => core_pmem_rvalid_out,
        tta_rready_in      => core_pmem_rready_in,
        tta_rdata_out      => core_pmem_rdata_out,
        -- Bus to memory
        mem_avalid_out => mc_arb_pmem_avalid,
        mem_aready_in  => mc_arb_pmem_aready,
        mem_aaddr_out  => mc_arb_pmem_aaddr,
        mem_awren_out  => mc_arb_pmem_awren,
        mem_astrb_out  => mc_arb_pmem_astrb,
        mem_adata_out  => mc_arb_pmem_adata,
        mem_rvalid_in  => mc_arb_pmem_rvalid,
        mem_rready_out => mc_arb_pmem_rready,
        mem_rdata_in   => mc_arb_pmem_rdata
      );
    end generate;
  end generate;

  no_mc_arbiters : if core_count_g <= 1 generate
    data_a_avalid_out       <= core_dmem_avalid_in;
    core_dmem_aready_out    <= data_a_aready_in;
    data_a_aaddr_out        <= core_dmem_aaddr_in;
    data_a_awren_out        <= core_dmem_awren_in;
    data_a_astrb_out        <= core_dmem_astrb_in;
    data_a_adata_out        <= core_dmem_adata_in;
    core_dmem_rvalid_out    <= data_a_rvalid_in;
    data_a_rready_out       <= core_dmem_rready_in;
    core_dmem_rdata_out     <= data_a_rdata_in;

    mc_arb_pmem_avalid      <= core_pmem_avalid_in(0);
    core_pmem_aready_out(0) <= mc_arb_pmem_aready;
    mc_arb_pmem_aaddr       <= core_pmem_aaddr_in;
    mc_arb_pmem_awren       <= core_pmem_awren_in(0);
    mc_arb_pmem_astrb       <= core_pmem_astrb_in;
    mc_arb_pmem_adata       <= core_pmem_adata_in;
    core_pmem_rvalid_out(0) <= mc_arb_pmem_rvalid;
    mc_arb_pmem_rready      <= core_pmem_rready_in(0);
    core_pmem_rdata_out     <= mc_arb_pmem_rdata;
  end generate;

  gen_decoder : if axi_offset_g /= 0 generate
    decoder_pmem : entity work.almaif_decoder
    generic map (
      mem_dataw_g  => pmem_data_width_g,
      mem_addrw_g  => local_mem_addrw_g,
      axi_addrw_g  => 32,
      mem_offset_g => pmem_offset_c,
      sync_reset_g => sync_reset_g
    ) port map (
      clk            => clk,
      rstx           => rstx,
      -- Bus from arbiter
      arb_avalid_in  => mc_arb_pmem_avalid,
      arb_aready_out => mc_arb_pmem_aready,
      arb_aaddr_in   => mc_arb_pmem_aaddr,
      arb_awren_in   => mc_arb_pmem_awren,
      arb_astrb_in   => mc_arb_pmem_astrb,
      arb_adata_in   => mc_arb_pmem_adata,
      --
      arb_rvalid_out => mc_arb_pmem_rvalid,
      arb_rready_in  => mc_arb_pmem_rready,
      arb_rdata_out  => mc_arb_pmem_rdata,
      -- Bus to local memory
      mem_avalid_out => param_a_avalid_out(0),
      mem_aready_in  => param_a_aready_in(0),
      mem_aaddr_out  => param_a_aaddr_out,
      mem_awren_out  => param_a_awren_out(0),
      mem_astrb_out  => param_a_astrb_out,
      mem_adata_out  => param_a_adata_out,
      --
      mem_rvalid_in  => param_a_rvalid_in(0),
      mem_rready_out => param_a_rready_out(0),
      mem_rdata_in   => param_a_rdata_in,
      -- AXI lite master
      m_axi_awvalid  => m_axi_awvalid,
      m_axi_awready  => m_axi_awready,
      m_axi_awaddr   => m_axi_awaddr,
      m_axi_awprot   => m_axi_awprot,
      --
      m_axi_wvalid   => m_axi_wvalid,
      m_axi_wready   => m_axi_wready,
      m_axi_wdata    => m_axi_wdata,
      m_axi_wstrb    => m_axi_wstrb,
      --
      m_axi_bvalid   => m_axi_bvalid,
      m_axi_bready   => m_axi_bready,
      --
      m_axi_arvalid  => m_axi_arvalid,
      m_axi_arready  => m_axi_arready,
      m_axi_araddr   => m_axi_araddr,
      m_axi_arprot   => m_axi_arprot,
      --
      m_axi_rvalid   => m_axi_rvalid,
      m_axi_rready   => m_axi_rready,
      m_axi_rdata    => m_axi_rdata
    );
  end generate;

  no_decoder : if axi_offset_g = 0 generate
    param_a_avalid_out(0)  <= mc_arb_pmem_avalid;
    mc_arb_pmem_aready     <= param_a_aready_in(0);
    param_a_aaddr_out      <= mc_arb_pmem_aaddr;
    param_a_awren_out(0)   <= mc_arb_pmem_awren;
    param_a_astrb_out      <= mc_arb_pmem_astrb;
    param_a_adata_out      <= mc_arb_pmem_adata;
    mc_arb_pmem_rvalid     <= param_a_rvalid_in(0);
    param_a_rready_out(0)  <= mc_arb_pmem_rready;
    mc_arb_pmem_rdata      <= param_a_rdata_in;
  end generate;


  gen_dmem_expander : if enable_dmem generate
    dmem_expander : entity work.almaif_axi_expander
    generic map (
      mem_dataw_g  => dmem_data_width_g,
      mem_addrw_g  => dmem_addr_width_g,
      axi_dataw_g  => dataw_c,
      axi_addrw_g  => axi_addr_width_g,
      sync_reset_g => sync_reset_g
    ) port map (
      clk => clk, rstx => rstx,
      -- Bus to AXI if
      axi_avalid_in  => dmem_avalid,
      axi_aready_out => dmem_aready,
      axi_aaddr_in   => axi_aaddr,
      axi_awren_in   => axi_awren,
      axi_astrb_in   => axi_astrb,
      axi_adata_in   => axi_adata,
      axi_rvalid_out => dmem_rvalid,
      axi_rready_in  => dmem_rready,
      axi_rdata_out  => dmem_rdata,
      -- Bus to memory
      mem_avalid_out => data_b_avalid_out(0),
      mem_aready_in  => data_b_aready_in(0),
      mem_aaddr_out  => data_b_aaddr_out,
      mem_awren_out  => data_b_awren_out(0),
      mem_astrb_out  => data_b_astrb_out,
      mem_adata_out  => data_b_adata_out,
      mem_rvalid_in  => data_b_rvalid_in(0),
      mem_rready_out => data_b_rready_out(0),
      mem_rdata_in   => data_b_rdata_in
    );
  end generate;

  no_dmem_expander : if not enable_dmem generate
    dmem_aready <= '1';
    dmem_rvalid <= '1';
    dmem_rdata  <= (others => '0');
  end generate;

  gen_pmem_expander : if enable_pmem generate
    pmem_epander : entity work.almaif_axi_expander
    generic map (
      mem_dataw_g  => pmem_data_width_g,
      mem_addrw_g  => local_mem_addrw_g,
      axi_dataw_g  => dataw_c,
      axi_addrw_g  => axi_addr_width_g,
      sync_reset_g => sync_reset_g
    ) port map (
      clk => clk, rstx => rstx,
      -- Bus to AXI if
      axi_avalid_in  => pmem_avalid,
      axi_aready_out => pmem_aready,
      axi_aaddr_in   => axi_aaddr,
      axi_awren_in   => axi_awren,
      axi_astrb_in   => axi_astrb,
      axi_adata_in   => axi_adata,
      axi_rvalid_out => pmem_rvalid,
      axi_rready_in  => pmem_rready,
      axi_rdata_out  => pmem_rdata,
      -- Bus to memory
      mem_avalid_out => param_b_avalid_out(0),
      mem_aready_in  => param_b_aready_in(0),
      mem_aaddr_out  => param_b_aaddr_out,
      mem_awren_out  => param_b_awren_out(0),
      mem_astrb_out  => param_b_astrb_out,
      mem_adata_out  => param_b_adata_out,
      mem_rvalid_in  => param_b_rvalid_in(0),
      mem_rready_out => param_b_rready_out(0),
      mem_rdata_in   => param_b_rdata_in
    );
  end generate;

  no_pmem_expander : if not enable_pmem generate
    pmem_aready <= '1';
    pmem_rvalid <= '1';
    pmem_rdata  <= (others => '0');
  end generate;

  
end architecture rtl;

