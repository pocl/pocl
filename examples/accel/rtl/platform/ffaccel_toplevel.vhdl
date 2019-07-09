library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.tce_util.all;
use work.ffaccel_globals.all;
use work.ffaccel_imem_mau.all;
use work.ffaccel_toplevel_params.all;

entity ffaccel_toplevel is

  generic (
    axi_addr_width_g : integer := 17;
    axi_id_width_g : integer := 12;
    local_mem_addrw_g : integer := 10;
    axi_offset_g : integer := 1136656384);

  port (
    clk : in std_logic;
    rstx : in std_logic;
    s_axi_awid : in std_logic_vector(axi_id_width_g-1 downto 0);
    s_axi_awaddr : in std_logic_vector(axi_addr_width_g-1 downto 0);
    s_axi_awlen : in std_logic_vector(7 downto 0);
    s_axi_awsize : in std_logic_vector(2 downto 0);
    s_axi_awburst : in std_logic_vector(1 downto 0);
    s_axi_awvalid : in std_logic;
    s_axi_awready : out std_logic;
    s_axi_wdata : in std_logic_vector(31 downto 0);
    s_axi_wstrb : in std_logic_vector(3 downto 0);
    s_axi_wvalid : in std_logic;
    s_axi_wready : out std_logic;
    s_axi_bid : out std_logic_vector(axi_id_width_g-1 downto 0);
    s_axi_bresp : out std_logic_vector(1 downto 0);
    s_axi_bvalid : out std_logic;
    s_axi_bready : in std_logic;
    s_axi_arid : in std_logic_vector(axi_id_width_g-1 downto 0);
    s_axi_araddr : in std_logic_vector(axi_addr_width_g-1 downto 0);
    s_axi_arlen : in std_logic_vector(7 downto 0);
    s_axi_arsize : in std_logic_vector(2 downto 0);
    s_axi_arburst : in std_logic_vector(1 downto 0);
    s_axi_arvalid : in std_logic;
    s_axi_arready : out std_logic;
    s_axi_rid : out std_logic_vector(axi_id_width_g-1 downto 0);
    s_axi_rdata : out std_logic_vector(31 downto 0);
    s_axi_rresp : out std_logic_vector(1 downto 0);
    s_axi_rlast : out std_logic;
    s_axi_rvalid : out std_logic;
    s_axi_rready : in std_logic;
    m_axi_awaddr : out std_logic_vector(31 downto 0);
    m_axi_awvalid : out std_logic;
    m_axi_awready : in std_logic;
    m_axi_awprot : out std_logic_vector(2 downto 0);
    m_axi_wvalid : out std_logic;
    m_axi_wready : in std_logic;
    m_axi_wdata : out std_logic_vector(31 downto 0);
    m_axi_wstrb : out std_logic_vector(3 downto 0);
    m_axi_bvalid : in std_logic;
    m_axi_bready : out std_logic;
    m_axi_arvalid : out std_logic;
    m_axi_arready : in std_logic;
    m_axi_araddr : out std_logic_vector(31 downto 0);
    m_axi_arprot : out std_logic_vector(2 downto 0);
    m_axi_rdata : in std_logic_vector(31 downto 0);
    m_axi_rvalid : in std_logic;
    m_axi_rready : out std_logic;
    locked : out std_logic);

end ffaccel_toplevel;

architecture structural of ffaccel_toplevel is

  signal core_busy_wire : std_logic;
  signal core_imem_en_x_wire : std_logic;
  signal core_imem_addr_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal core_imem_data_wire : std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
  signal core_fu_DATA_LSU_avalid_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_DATA_LSU_aready_in_wire : std_logic_vector(0 downto 0);
  signal core_fu_DATA_LSU_aaddr_out_wire : std_logic_vector(fu_DATA_LSU_addrw_g-2-1 downto 0);
  signal core_fu_DATA_LSU_awren_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_DATA_LSU_astrb_out_wire : std_logic_vector(3 downto 0);
  signal core_fu_DATA_LSU_adata_out_wire : std_logic_vector(31 downto 0);
  signal core_fu_DATA_LSU_rvalid_in_wire : std_logic_vector(0 downto 0);
  signal core_fu_DATA_LSU_rready_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_DATA_LSU_rdata_in_wire : std_logic_vector(31 downto 0);
  signal core_fu_PARAM_LSU_avalid_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_PARAM_LSU_aready_in_wire : std_logic_vector(0 downto 0);
  signal core_fu_PARAM_LSU_aaddr_out_wire : std_logic_vector(fu_PARAM_LSU_addrw_g-2-1 downto 0);
  signal core_fu_PARAM_LSU_awren_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_PARAM_LSU_astrb_out_wire : std_logic_vector(3 downto 0);
  signal core_fu_PARAM_LSU_adata_out_wire : std_logic_vector(31 downto 0);
  signal core_fu_PARAM_LSU_rvalid_in_wire : std_logic_vector(0 downto 0);
  signal core_fu_PARAM_LSU_rready_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_PARAM_LSU_rdata_in_wire : std_logic_vector(31 downto 0);
  signal core_fu_SP_LSU_avalid_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_SP_LSU_aready_in_wire : std_logic_vector(0 downto 0);
  signal core_fu_SP_LSU_aaddr_out_wire : std_logic_vector(fu_SP_LSU_addrw_g-2-1 downto 0);
  signal core_fu_SP_LSU_awren_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_SP_LSU_astrb_out_wire : std_logic_vector(3 downto 0);
  signal core_fu_SP_LSU_adata_out_wire : std_logic_vector(31 downto 0);
  signal core_fu_SP_LSU_rvalid_in_wire : std_logic_vector(0 downto 0);
  signal core_fu_SP_LSU_rready_out_wire : std_logic_vector(0 downto 0);
  signal core_fu_SP_LSU_rdata_in_wire : std_logic_vector(31 downto 0);
  signal core_fu_AQL_FU_read_idx_out_wire : std_logic_vector(63 downto 0);
  signal core_fu_AQL_FU_read_idx_clear_in_wire : std_logic_vector(0 downto 0);
  signal core_db_tta_nreset_wire : std_logic;
  signal core_db_lockcnt_wire : std_logic_vector(63 downto 0);
  signal core_db_cyclecnt_wire : std_logic_vector(63 downto 0);
  signal core_db_pc_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal core_db_lockrq_wire : std_logic;
  signal imem_array_instance_0_addr_wire : std_logic_vector(11 downto 0);
  signal imem_array_instance_0_dataout_wire : std_logic_vector(42 downto 0);
  signal imem_array_instance_0_en_x_wire : std_logic;
  signal onchip_mem_data_a_aaddr_in_wire : std_logic_vector(9 downto 0);
  signal onchip_mem_data_a_adata_in_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_data_a_aready_out_wire : std_logic;
  signal onchip_mem_data_a_astrb_in_wire : std_logic_vector(3 downto 0);
  signal onchip_mem_data_a_avalid_in_wire : std_logic;
  signal onchip_mem_data_a_awren_in_wire : std_logic;
  signal onchip_mem_data_a_rdata_out_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_data_a_rready_in_wire : std_logic;
  signal onchip_mem_data_a_rvalid_out_wire : std_logic;
  signal onchip_mem_data_b_aaddr_in_wire : std_logic_vector(9 downto 0);
  signal onchip_mem_data_b_adata_in_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_data_b_aready_out_wire : std_logic;
  signal onchip_mem_data_b_astrb_in_wire : std_logic_vector(3 downto 0);
  signal onchip_mem_data_b_avalid_in_wire : std_logic;
  signal onchip_mem_data_b_awren_in_wire : std_logic;
  signal onchip_mem_data_b_rdata_out_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_data_b_rready_in_wire : std_logic;
  signal onchip_mem_data_b_rvalid_out_wire : std_logic;
  signal onchip_mem_param_a_aaddr_in_wire : std_logic_vector(local_mem_addrw_g-1 downto 0);
  signal onchip_mem_param_a_adata_in_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_param_a_aready_out_wire : std_logic;
  signal onchip_mem_param_a_astrb_in_wire : std_logic_vector(3 downto 0);
  signal onchip_mem_param_a_avalid_in_wire : std_logic;
  signal onchip_mem_param_a_awren_in_wire : std_logic;
  signal onchip_mem_param_a_rdata_out_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_param_a_rready_in_wire : std_logic;
  signal onchip_mem_param_a_rvalid_out_wire : std_logic;
  signal onchip_mem_param_b_aaddr_in_wire : std_logic_vector(local_mem_addrw_g-1 downto 0);
  signal onchip_mem_param_b_adata_in_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_param_b_aready_out_wire : std_logic;
  signal onchip_mem_param_b_astrb_in_wire : std_logic_vector(3 downto 0);
  signal onchip_mem_param_b_avalid_in_wire : std_logic;
  signal onchip_mem_param_b_awren_in_wire : std_logic;
  signal onchip_mem_param_b_rdata_out_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_param_b_rready_in_wire : std_logic;
  signal onchip_mem_param_b_rvalid_out_wire : std_logic;
  signal onchip_mem_scratchpad_aaddr_in_wire : std_logic_vector(7 downto 0);
  signal onchip_mem_scratchpad_adata_in_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_scratchpad_aready_out_wire : std_logic;
  signal onchip_mem_scratchpad_astrb_in_wire : std_logic_vector(3 downto 0);
  signal onchip_mem_scratchpad_avalid_in_wire : std_logic;
  signal onchip_mem_scratchpad_awren_in_wire : std_logic;
  signal onchip_mem_scratchpad_rdata_out_wire : std_logic_vector(31 downto 0);
  signal onchip_mem_scratchpad_rready_in_wire : std_logic;
  signal onchip_mem_scratchpad_rvalid_out_wire : std_logic;
  signal tta_accel_0_core_db_pc_wire : std_logic_vector(11 downto 0);
  signal tta_accel_0_core_db_lockcnt_wire : std_logic_vector(63 downto 0);
  signal tta_accel_0_core_db_cyclecnt_wire : std_logic_vector(63 downto 0);
  signal tta_accel_0_core_db_tta_nreset_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_db_lockrq_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_dmem_avalid_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_dmem_aready_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_dmem_aaddr_in_wire : std_logic_vector(9 downto 0);
  signal tta_accel_0_core_dmem_awren_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_dmem_astrb_in_wire : std_logic_vector(3 downto 0);
  signal tta_accel_0_core_dmem_adata_in_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_core_dmem_rvalid_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_dmem_rready_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_dmem_rdata_out_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_data_a_avalid_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_a_aready_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_a_aaddr_out_wire : std_logic_vector(9 downto 0);
  signal tta_accel_0_data_a_awren_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_a_astrb_out_wire : std_logic_vector(3 downto 0);
  signal tta_accel_0_data_a_adata_out_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_data_a_rvalid_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_a_rready_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_a_rdata_in_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_data_b_avalid_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_b_aready_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_b_aaddr_out_wire : std_logic_vector(9 downto 0);
  signal tta_accel_0_data_b_awren_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_b_astrb_out_wire : std_logic_vector(3 downto 0);
  signal tta_accel_0_data_b_adata_out_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_data_b_rvalid_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_b_rready_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_data_b_rdata_in_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_core_pmem_avalid_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_pmem_aready_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_pmem_aaddr_in_wire : std_logic_vector(29 downto 0);
  signal tta_accel_0_core_pmem_awren_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_pmem_astrb_in_wire : std_logic_vector(3 downto 0);
  signal tta_accel_0_core_pmem_adata_in_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_core_pmem_rvalid_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_pmem_rready_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_core_pmem_rdata_out_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_param_a_avalid_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_a_aready_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_a_aaddr_out_wire : std_logic_vector(local_mem_addrw_g-1 downto 0);
  signal tta_accel_0_param_a_awren_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_a_astrb_out_wire : std_logic_vector(3 downto 0);
  signal tta_accel_0_param_a_adata_out_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_param_a_rvalid_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_a_rready_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_a_rdata_in_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_param_b_avalid_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_b_aready_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_b_aaddr_out_wire : std_logic_vector(local_mem_addrw_g-1 downto 0);
  signal tta_accel_0_param_b_awren_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_b_astrb_out_wire : std_logic_vector(3 downto 0);
  signal tta_accel_0_param_b_adata_out_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_param_b_rvalid_in_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_b_rready_out_wire : std_logic_vector(0 downto 0);
  signal tta_accel_0_param_b_rdata_in_wire : std_logic_vector(31 downto 0);
  signal tta_accel_0_aql_read_idx_in_wire : std_logic_vector(63 downto 0);
  signal tta_accel_0_aql_read_idx_clear_out_wire : std_logic_vector(0 downto 0);

  component ffaccel
    generic (
      core_id : integer);
    port (
      clk : in std_logic;
      rstx : in std_logic;
      busy : in std_logic;
      imem_en_x : out std_logic;
      imem_addr : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      imem_data : in std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
      locked : out std_logic;
      fu_DATA_LSU_avalid_out : out std_logic_vector(1-1 downto 0);
      fu_DATA_LSU_aready_in : in std_logic_vector(1-1 downto 0);
      fu_DATA_LSU_aaddr_out : out std_logic_vector(fu_DATA_LSU_addrw_g-2-1 downto 0);
      fu_DATA_LSU_awren_out : out std_logic_vector(1-1 downto 0);
      fu_DATA_LSU_astrb_out : out std_logic_vector(4-1 downto 0);
      fu_DATA_LSU_adata_out : out std_logic_vector(32-1 downto 0);
      fu_DATA_LSU_rvalid_in : in std_logic_vector(1-1 downto 0);
      fu_DATA_LSU_rready_out : out std_logic_vector(1-1 downto 0);
      fu_DATA_LSU_rdata_in : in std_logic_vector(32-1 downto 0);
      fu_PARAM_LSU_avalid_out : out std_logic_vector(1-1 downto 0);
      fu_PARAM_LSU_aready_in : in std_logic_vector(1-1 downto 0);
      fu_PARAM_LSU_aaddr_out : out std_logic_vector(fu_PARAM_LSU_addrw_g-2-1 downto 0);
      fu_PARAM_LSU_awren_out : out std_logic_vector(1-1 downto 0);
      fu_PARAM_LSU_astrb_out : out std_logic_vector(4-1 downto 0);
      fu_PARAM_LSU_adata_out : out std_logic_vector(32-1 downto 0);
      fu_PARAM_LSU_rvalid_in : in std_logic_vector(1-1 downto 0);
      fu_PARAM_LSU_rready_out : out std_logic_vector(1-1 downto 0);
      fu_PARAM_LSU_rdata_in : in std_logic_vector(32-1 downto 0);
      fu_SP_LSU_avalid_out : out std_logic_vector(1-1 downto 0);
      fu_SP_LSU_aready_in : in std_logic_vector(1-1 downto 0);
      fu_SP_LSU_aaddr_out : out std_logic_vector(fu_SP_LSU_addrw_g-2-1 downto 0);
      fu_SP_LSU_awren_out : out std_logic_vector(1-1 downto 0);
      fu_SP_LSU_astrb_out : out std_logic_vector(4-1 downto 0);
      fu_SP_LSU_adata_out : out std_logic_vector(32-1 downto 0);
      fu_SP_LSU_rvalid_in : in std_logic_vector(1-1 downto 0);
      fu_SP_LSU_rready_out : out std_logic_vector(1-1 downto 0);
      fu_SP_LSU_rdata_in : in std_logic_vector(32-1 downto 0);
      fu_AQL_FU_read_idx_out : out std_logic_vector(64-1 downto 0);
      fu_AQL_FU_read_idx_clear_in : in std_logic_vector(1-1 downto 0);
      db_tta_nreset : in std_logic;
      db_lockcnt : out std_logic_vector(64-1 downto 0);
      db_cyclecnt : out std_logic_vector(64-1 downto 0);
      db_pc : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      db_lockrq : in std_logic);
  end component;

  component tta_accel
    generic (
      core_count_g : integer;
      axi_addr_width_g : integer;
      axi_id_width_g : integer;
      imem_data_width_g : integer;
      imem_addr_width_g : integer;
      bus_count_g : integer;
      local_mem_addrw_g : integer;
      sync_reset_g : integer;
      axi_offset_g : integer;
      full_debugger_g : integer;
      dmem_data_width_g : integer;
      dmem_addr_width_g : integer;
      pmem_data_width_g : integer;
      pmem_addr_width_g : integer);
    port (
      clk : in std_logic;
      rstx : in std_logic;
      s_axi_awid : in std_logic_vector(axi_id_width_g-1 downto 0);
      s_axi_awaddr : in std_logic_vector(axi_addr_width_g-1 downto 0);
      s_axi_awlen : in std_logic_vector(8-1 downto 0);
      s_axi_awsize : in std_logic_vector(3-1 downto 0);
      s_axi_awburst : in std_logic_vector(2-1 downto 0);
      s_axi_awvalid : in std_logic;
      s_axi_awready : out std_logic;
      s_axi_wdata : in std_logic_vector(32-1 downto 0);
      s_axi_wstrb : in std_logic_vector(4-1 downto 0);
      s_axi_wvalid : in std_logic;
      s_axi_wready : out std_logic;
      s_axi_bid : out std_logic_vector(axi_id_width_g-1 downto 0);
      s_axi_bresp : out std_logic_vector(2-1 downto 0);
      s_axi_bvalid : out std_logic;
      s_axi_bready : in std_logic;
      s_axi_arid : in std_logic_vector(axi_id_width_g-1 downto 0);
      s_axi_araddr : in std_logic_vector(axi_addr_width_g-1 downto 0);
      s_axi_arlen : in std_logic_vector(8-1 downto 0);
      s_axi_arsize : in std_logic_vector(3-1 downto 0);
      s_axi_arburst : in std_logic_vector(2-1 downto 0);
      s_axi_arvalid : in std_logic;
      s_axi_arready : out std_logic;
      s_axi_rid : out std_logic_vector(axi_id_width_g-1 downto 0);
      s_axi_rdata : out std_logic_vector(32-1 downto 0);
      s_axi_rresp : out std_logic_vector(2-1 downto 0);
      s_axi_rlast : out std_logic;
      s_axi_rvalid : out std_logic;
      s_axi_rready : in std_logic;
      m_axi_awaddr : out std_logic_vector(32-1 downto 0);
      m_axi_awvalid : out std_logic;
      m_axi_awready : in std_logic;
      m_axi_awprot : out std_logic_vector(3-1 downto 0);
      m_axi_wvalid : out std_logic;
      m_axi_wready : in std_logic;
      m_axi_wdata : out std_logic_vector(32-1 downto 0);
      m_axi_wstrb : out std_logic_vector(4-1 downto 0);
      m_axi_bvalid : in std_logic;
      m_axi_bready : out std_logic;
      m_axi_arvalid : out std_logic;
      m_axi_arready : in std_logic;
      m_axi_araddr : out std_logic_vector(32-1 downto 0);
      m_axi_arprot : out std_logic_vector(3-1 downto 0);
      m_axi_rdata : in std_logic_vector(32-1 downto 0);
      m_axi_rvalid : in std_logic;
      m_axi_rready : out std_logic;
      core_db_pc : in std_logic_vector(12-1 downto 0);
      core_db_lockcnt : in std_logic_vector(64-1 downto 0);
      core_db_cyclecnt : in std_logic_vector(64-1 downto 0);
      core_db_tta_nreset : out std_logic_vector(1-1 downto 0);
      core_db_lockrq : out std_logic_vector(1-1 downto 0);
      core_dmem_avalid_in : in std_logic_vector(1-1 downto 0);
      core_dmem_aready_out : out std_logic_vector(1-1 downto 0);
      core_dmem_aaddr_in : in std_logic_vector(10-1 downto 0);
      core_dmem_awren_in : in std_logic_vector(1-1 downto 0);
      core_dmem_astrb_in : in std_logic_vector(4-1 downto 0);
      core_dmem_adata_in : in std_logic_vector(32-1 downto 0);
      core_dmem_rvalid_out : out std_logic_vector(1-1 downto 0);
      core_dmem_rready_in : in std_logic_vector(1-1 downto 0);
      core_dmem_rdata_out : out std_logic_vector(32-1 downto 0);
      data_a_avalid_out : out std_logic_vector(1-1 downto 0);
      data_a_aready_in : in std_logic_vector(1-1 downto 0);
      data_a_aaddr_out : out std_logic_vector(10-1 downto 0);
      data_a_awren_out : out std_logic_vector(1-1 downto 0);
      data_a_astrb_out : out std_logic_vector(4-1 downto 0);
      data_a_adata_out : out std_logic_vector(32-1 downto 0);
      data_a_rvalid_in : in std_logic_vector(1-1 downto 0);
      data_a_rready_out : out std_logic_vector(1-1 downto 0);
      data_a_rdata_in : in std_logic_vector(32-1 downto 0);
      data_b_avalid_out : out std_logic_vector(1-1 downto 0);
      data_b_aready_in : in std_logic_vector(1-1 downto 0);
      data_b_aaddr_out : out std_logic_vector(10-1 downto 0);
      data_b_awren_out : out std_logic_vector(1-1 downto 0);
      data_b_astrb_out : out std_logic_vector(4-1 downto 0);
      data_b_adata_out : out std_logic_vector(32-1 downto 0);
      data_b_rvalid_in : in std_logic_vector(1-1 downto 0);
      data_b_rready_out : out std_logic_vector(1-1 downto 0);
      data_b_rdata_in : in std_logic_vector(32-1 downto 0);
      core_pmem_avalid_in : in std_logic_vector(1-1 downto 0);
      core_pmem_aready_out : out std_logic_vector(1-1 downto 0);
      core_pmem_aaddr_in : in std_logic_vector(30-1 downto 0);
      core_pmem_awren_in : in std_logic_vector(1-1 downto 0);
      core_pmem_astrb_in : in std_logic_vector(4-1 downto 0);
      core_pmem_adata_in : in std_logic_vector(32-1 downto 0);
      core_pmem_rvalid_out : out std_logic_vector(1-1 downto 0);
      core_pmem_rready_in : in std_logic_vector(1-1 downto 0);
      core_pmem_rdata_out : out std_logic_vector(32-1 downto 0);
      param_a_avalid_out : out std_logic_vector(1-1 downto 0);
      param_a_aready_in : in std_logic_vector(1-1 downto 0);
      param_a_aaddr_out : out std_logic_vector(local_mem_addrw_g-1 downto 0);
      param_a_awren_out : out std_logic_vector(1-1 downto 0);
      param_a_astrb_out : out std_logic_vector(4-1 downto 0);
      param_a_adata_out : out std_logic_vector(32-1 downto 0);
      param_a_rvalid_in : in std_logic_vector(1-1 downto 0);
      param_a_rready_out : out std_logic_vector(1-1 downto 0);
      param_a_rdata_in : in std_logic_vector(32-1 downto 0);
      param_b_avalid_out : out std_logic_vector(1-1 downto 0);
      param_b_aready_in : in std_logic_vector(1-1 downto 0);
      param_b_aaddr_out : out std_logic_vector(local_mem_addrw_g-1 downto 0);
      param_b_awren_out : out std_logic_vector(1-1 downto 0);
      param_b_astrb_out : out std_logic_vector(4-1 downto 0);
      param_b_adata_out : out std_logic_vector(32-1 downto 0);
      param_b_rvalid_in : in std_logic_vector(1-1 downto 0);
      param_b_rready_out : out std_logic_vector(1-1 downto 0);
      param_b_rdata_in : in std_logic_vector(32-1 downto 0);
      aql_read_idx_in : in std_logic_vector(64-1 downto 0);
      aql_read_idx_clear_out : out std_logic_vector(1-1 downto 0));
  end component;

  component ffaccel_rom_array_comp
    generic (
      addrw : integer;
      instrw : integer);
    port (
      clock : in std_logic;
      addr : in std_logic_vector(addrw-1 downto 0);
      dataout : out std_logic_vector(instrw-1 downto 0);
      en_x : in std_logic);
  end component;

  component xilinx_dp_blockram
    generic (
      dataw_g : integer;
      addrw_g : integer);
    port (
      a_aaddr_in : in std_logic_vector(addrw_g-1 downto 0);
      a_adata_in : in std_logic_vector(dataw_g-1 downto 0);
      a_aready_out : out std_logic;
      a_astrb_in : in std_logic_vector((dataw_g+7)/8-1 downto 0);
      a_avalid_in : in std_logic;
      a_awren_in : in std_logic;
      a_rdata_out : out std_logic_vector(dataw_g-1 downto 0);
      a_rready_in : in std_logic;
      a_rvalid_out : out std_logic;
      b_aaddr_in : in std_logic_vector(addrw_g-1 downto 0);
      b_adata_in : in std_logic_vector(dataw_g-1 downto 0);
      b_aready_out : out std_logic;
      b_astrb_in : in std_logic_vector((dataw_g+7)/8-1 downto 0);
      b_avalid_in : in std_logic;
      b_awren_in : in std_logic;
      b_rdata_out : out std_logic_vector(dataw_g-1 downto 0);
      b_rready_in : in std_logic;
      b_rvalid_out : out std_logic;
      clk : in std_logic;
      rstx : in std_logic);
  end component;

  component xilinx_blockram
    generic (
      dataw_g : integer;
      addrw_g : integer);
    port (
      aaddr_in : in std_logic_vector(addrw_g-1 downto 0);
      adata_in : in std_logic_vector(dataw_g-1 downto 0);
      aready_out : out std_logic;
      astrb_in : in std_logic_vector((dataw_g+7)/8-1 downto 0);
      avalid_in : in std_logic;
      awren_in : in std_logic;
      clk : in std_logic;
      rdata_out : out std_logic_vector(dataw_g-1 downto 0);
      rready_in : in std_logic;
      rstx : in std_logic;
      rvalid_out : out std_logic);
  end component;


begin

  core_busy_wire <= '0';
  imem_array_instance_0_en_x_wire <= core_imem_en_x_wire;
  imem_array_instance_0_addr_wire <= core_imem_addr_wire;
  core_imem_data_wire <= imem_array_instance_0_dataout_wire;
  tta_accel_0_core_dmem_avalid_in_wire <= core_fu_DATA_LSU_avalid_out_wire;
  core_fu_DATA_LSU_aready_in_wire <= tta_accel_0_core_dmem_aready_out_wire;
  tta_accel_0_core_dmem_aaddr_in_wire <= core_fu_DATA_LSU_aaddr_out_wire;
  tta_accel_0_core_dmem_awren_in_wire <= core_fu_DATA_LSU_awren_out_wire;
  tta_accel_0_core_dmem_astrb_in_wire <= core_fu_DATA_LSU_astrb_out_wire;
  tta_accel_0_core_dmem_adata_in_wire <= core_fu_DATA_LSU_adata_out_wire;
  core_fu_DATA_LSU_rvalid_in_wire <= tta_accel_0_core_dmem_rvalid_out_wire;
  tta_accel_0_core_dmem_rready_in_wire <= core_fu_DATA_LSU_rready_out_wire;
  core_fu_DATA_LSU_rdata_in_wire <= tta_accel_0_core_dmem_rdata_out_wire;
  tta_accel_0_core_pmem_avalid_in_wire <= core_fu_PARAM_LSU_avalid_out_wire;
  core_fu_PARAM_LSU_aready_in_wire <= tta_accel_0_core_pmem_aready_out_wire;
  tta_accel_0_core_pmem_aaddr_in_wire <= core_fu_PARAM_LSU_aaddr_out_wire;
  tta_accel_0_core_pmem_awren_in_wire <= core_fu_PARAM_LSU_awren_out_wire;
  tta_accel_0_core_pmem_astrb_in_wire <= core_fu_PARAM_LSU_astrb_out_wire;
  tta_accel_0_core_pmem_adata_in_wire <= core_fu_PARAM_LSU_adata_out_wire;
  core_fu_PARAM_LSU_rvalid_in_wire <= tta_accel_0_core_pmem_rvalid_out_wire;
  tta_accel_0_core_pmem_rready_in_wire <= core_fu_PARAM_LSU_rready_out_wire;
  core_fu_PARAM_LSU_rdata_in_wire <= tta_accel_0_core_pmem_rdata_out_wire;
  onchip_mem_scratchpad_avalid_in_wire <= core_fu_SP_LSU_avalid_out_wire(0);
  core_fu_SP_LSU_aready_in_wire(0) <= onchip_mem_scratchpad_aready_out_wire;
  onchip_mem_scratchpad_aaddr_in_wire <= core_fu_SP_LSU_aaddr_out_wire;
  onchip_mem_scratchpad_awren_in_wire <= core_fu_SP_LSU_awren_out_wire(0);
  onchip_mem_scratchpad_astrb_in_wire <= core_fu_SP_LSU_astrb_out_wire;
  onchip_mem_scratchpad_adata_in_wire <= core_fu_SP_LSU_adata_out_wire;
  core_fu_SP_LSU_rvalid_in_wire(0) <= onchip_mem_scratchpad_rvalid_out_wire;
  onchip_mem_scratchpad_rready_in_wire <= core_fu_SP_LSU_rready_out_wire(0);
  core_fu_SP_LSU_rdata_in_wire <= onchip_mem_scratchpad_rdata_out_wire;
  tta_accel_0_aql_read_idx_in_wire <= core_fu_AQL_FU_read_idx_out_wire;
  core_fu_AQL_FU_read_idx_clear_in_wire <= tta_accel_0_aql_read_idx_clear_out_wire;
  core_db_tta_nreset_wire <= tta_accel_0_core_db_tta_nreset_wire(0);
  tta_accel_0_core_db_lockcnt_wire <= core_db_lockcnt_wire;
  tta_accel_0_core_db_cyclecnt_wire <= core_db_cyclecnt_wire;
  tta_accel_0_core_db_pc_wire <= core_db_pc_wire;
  core_db_lockrq_wire <= tta_accel_0_core_db_lockrq_wire(0);
  onchip_mem_data_a_avalid_in_wire <= tta_accel_0_data_a_avalid_out_wire(0);
  tta_accel_0_data_a_aready_in_wire(0) <= onchip_mem_data_a_aready_out_wire;
  onchip_mem_data_a_aaddr_in_wire <= tta_accel_0_data_a_aaddr_out_wire;
  onchip_mem_data_a_awren_in_wire <= tta_accel_0_data_a_awren_out_wire(0);
  onchip_mem_data_a_astrb_in_wire <= tta_accel_0_data_a_astrb_out_wire;
  onchip_mem_data_a_adata_in_wire <= tta_accel_0_data_a_adata_out_wire;
  tta_accel_0_data_a_rvalid_in_wire(0) <= onchip_mem_data_a_rvalid_out_wire;
  onchip_mem_data_a_rready_in_wire <= tta_accel_0_data_a_rready_out_wire(0);
  tta_accel_0_data_a_rdata_in_wire <= onchip_mem_data_a_rdata_out_wire;
  onchip_mem_data_b_avalid_in_wire <= tta_accel_0_data_b_avalid_out_wire(0);
  tta_accel_0_data_b_aready_in_wire(0) <= onchip_mem_data_b_aready_out_wire;
  onchip_mem_data_b_aaddr_in_wire <= tta_accel_0_data_b_aaddr_out_wire;
  onchip_mem_data_b_awren_in_wire <= tta_accel_0_data_b_awren_out_wire(0);
  onchip_mem_data_b_astrb_in_wire <= tta_accel_0_data_b_astrb_out_wire;
  onchip_mem_data_b_adata_in_wire <= tta_accel_0_data_b_adata_out_wire;
  tta_accel_0_data_b_rvalid_in_wire(0) <= onchip_mem_data_b_rvalid_out_wire;
  onchip_mem_data_b_rready_in_wire <= tta_accel_0_data_b_rready_out_wire(0);
  tta_accel_0_data_b_rdata_in_wire <= onchip_mem_data_b_rdata_out_wire;
  onchip_mem_param_a_avalid_in_wire <= tta_accel_0_param_a_avalid_out_wire(0);
  tta_accel_0_param_a_aready_in_wire(0) <= onchip_mem_param_a_aready_out_wire;
  onchip_mem_param_a_aaddr_in_wire <= tta_accel_0_param_a_aaddr_out_wire;
  onchip_mem_param_a_awren_in_wire <= tta_accel_0_param_a_awren_out_wire(0);
  onchip_mem_param_a_astrb_in_wire <= tta_accel_0_param_a_astrb_out_wire;
  onchip_mem_param_a_adata_in_wire <= tta_accel_0_param_a_adata_out_wire;
  tta_accel_0_param_a_rvalid_in_wire(0) <= onchip_mem_param_a_rvalid_out_wire;
  onchip_mem_param_a_rready_in_wire <= tta_accel_0_param_a_rready_out_wire(0);
  tta_accel_0_param_a_rdata_in_wire <= onchip_mem_param_a_rdata_out_wire;
  onchip_mem_param_b_avalid_in_wire <= tta_accel_0_param_b_avalid_out_wire(0);
  tta_accel_0_param_b_aready_in_wire(0) <= onchip_mem_param_b_aready_out_wire;
  onchip_mem_param_b_aaddr_in_wire <= tta_accel_0_param_b_aaddr_out_wire;
  onchip_mem_param_b_awren_in_wire <= tta_accel_0_param_b_awren_out_wire(0);
  onchip_mem_param_b_astrb_in_wire <= tta_accel_0_param_b_astrb_out_wire;
  onchip_mem_param_b_adata_in_wire <= tta_accel_0_param_b_adata_out_wire;
  tta_accel_0_param_b_rvalid_in_wire(0) <= onchip_mem_param_b_rvalid_out_wire;
  onchip_mem_param_b_rready_in_wire <= tta_accel_0_param_b_rready_out_wire(0);
  tta_accel_0_param_b_rdata_in_wire <= onchip_mem_param_b_rdata_out_wire;

  core : ffaccel
    generic map (
      core_id => 0)
    port map (
      clk => clk,
      rstx => rstx,
      busy => core_busy_wire,
      imem_en_x => core_imem_en_x_wire,
      imem_addr => core_imem_addr_wire,
      imem_data => core_imem_data_wire,
      locked => locked,
      fu_DATA_LSU_avalid_out => core_fu_DATA_LSU_avalid_out_wire,
      fu_DATA_LSU_aready_in => core_fu_DATA_LSU_aready_in_wire,
      fu_DATA_LSU_aaddr_out => core_fu_DATA_LSU_aaddr_out_wire,
      fu_DATA_LSU_awren_out => core_fu_DATA_LSU_awren_out_wire,
      fu_DATA_LSU_astrb_out => core_fu_DATA_LSU_astrb_out_wire,
      fu_DATA_LSU_adata_out => core_fu_DATA_LSU_adata_out_wire,
      fu_DATA_LSU_rvalid_in => core_fu_DATA_LSU_rvalid_in_wire,
      fu_DATA_LSU_rready_out => core_fu_DATA_LSU_rready_out_wire,
      fu_DATA_LSU_rdata_in => core_fu_DATA_LSU_rdata_in_wire,
      fu_PARAM_LSU_avalid_out => core_fu_PARAM_LSU_avalid_out_wire,
      fu_PARAM_LSU_aready_in => core_fu_PARAM_LSU_aready_in_wire,
      fu_PARAM_LSU_aaddr_out => core_fu_PARAM_LSU_aaddr_out_wire,
      fu_PARAM_LSU_awren_out => core_fu_PARAM_LSU_awren_out_wire,
      fu_PARAM_LSU_astrb_out => core_fu_PARAM_LSU_astrb_out_wire,
      fu_PARAM_LSU_adata_out => core_fu_PARAM_LSU_adata_out_wire,
      fu_PARAM_LSU_rvalid_in => core_fu_PARAM_LSU_rvalid_in_wire,
      fu_PARAM_LSU_rready_out => core_fu_PARAM_LSU_rready_out_wire,
      fu_PARAM_LSU_rdata_in => core_fu_PARAM_LSU_rdata_in_wire,
      fu_SP_LSU_avalid_out => core_fu_SP_LSU_avalid_out_wire,
      fu_SP_LSU_aready_in => core_fu_SP_LSU_aready_in_wire,
      fu_SP_LSU_aaddr_out => core_fu_SP_LSU_aaddr_out_wire,
      fu_SP_LSU_awren_out => core_fu_SP_LSU_awren_out_wire,
      fu_SP_LSU_astrb_out => core_fu_SP_LSU_astrb_out_wire,
      fu_SP_LSU_adata_out => core_fu_SP_LSU_adata_out_wire,
      fu_SP_LSU_rvalid_in => core_fu_SP_LSU_rvalid_in_wire,
      fu_SP_LSU_rready_out => core_fu_SP_LSU_rready_out_wire,
      fu_SP_LSU_rdata_in => core_fu_SP_LSU_rdata_in_wire,
      fu_AQL_FU_read_idx_out => core_fu_AQL_FU_read_idx_out_wire,
      fu_AQL_FU_read_idx_clear_in => core_fu_AQL_FU_read_idx_clear_in_wire,
      db_tta_nreset => core_db_tta_nreset_wire,
      db_lockcnt => core_db_lockcnt_wire,
      db_cyclecnt => core_db_cyclecnt_wire,
      db_pc => core_db_pc_wire,
      db_lockrq => core_db_lockrq_wire);

  tta_accel_0 : tta_accel
    generic map (
      core_count_g => 1,
      axi_addr_width_g => axi_addr_width_g,
      axi_id_width_g => axi_id_width_g,
      imem_data_width_g => 43,
      imem_addr_width_g => 12,
      bus_count_g => 2,
      local_mem_addrw_g => local_mem_addrw_g,
      sync_reset_g => 0,
      axi_offset_g => axi_offset_g,
      full_debugger_g => 0,
      dmem_data_width_g => 32,
      dmem_addr_width_g => 10,
      pmem_data_width_g => 32,
      pmem_addr_width_g => 30)
    port map (
      clk => clk,
      rstx => rstx,
      s_axi_awid => s_axi_awid,
      s_axi_awaddr => s_axi_awaddr,
      s_axi_awlen => s_axi_awlen,
      s_axi_awsize => s_axi_awsize,
      s_axi_awburst => s_axi_awburst,
      s_axi_awvalid => s_axi_awvalid,
      s_axi_awready => s_axi_awready,
      s_axi_wdata => s_axi_wdata,
      s_axi_wstrb => s_axi_wstrb,
      s_axi_wvalid => s_axi_wvalid,
      s_axi_wready => s_axi_wready,
      s_axi_bid => s_axi_bid,
      s_axi_bresp => s_axi_bresp,
      s_axi_bvalid => s_axi_bvalid,
      s_axi_bready => s_axi_bready,
      s_axi_arid => s_axi_arid,
      s_axi_araddr => s_axi_araddr,
      s_axi_arlen => s_axi_arlen,
      s_axi_arsize => s_axi_arsize,
      s_axi_arburst => s_axi_arburst,
      s_axi_arvalid => s_axi_arvalid,
      s_axi_arready => s_axi_arready,
      s_axi_rid => s_axi_rid,
      s_axi_rdata => s_axi_rdata,
      s_axi_rresp => s_axi_rresp,
      s_axi_rlast => s_axi_rlast,
      s_axi_rvalid => s_axi_rvalid,
      s_axi_rready => s_axi_rready,
      m_axi_awaddr => m_axi_awaddr,
      m_axi_awvalid => m_axi_awvalid,
      m_axi_awready => m_axi_awready,
      m_axi_awprot => m_axi_awprot,
      m_axi_wvalid => m_axi_wvalid,
      m_axi_wready => m_axi_wready,
      m_axi_wdata => m_axi_wdata,
      m_axi_wstrb => m_axi_wstrb,
      m_axi_bvalid => m_axi_bvalid,
      m_axi_bready => m_axi_bready,
      m_axi_arvalid => m_axi_arvalid,
      m_axi_arready => m_axi_arready,
      m_axi_araddr => m_axi_araddr,
      m_axi_arprot => m_axi_arprot,
      m_axi_rdata => m_axi_rdata,
      m_axi_rvalid => m_axi_rvalid,
      m_axi_rready => m_axi_rready,
      core_db_pc => tta_accel_0_core_db_pc_wire,
      core_db_lockcnt => tta_accel_0_core_db_lockcnt_wire,
      core_db_cyclecnt => tta_accel_0_core_db_cyclecnt_wire,
      core_db_tta_nreset => tta_accel_0_core_db_tta_nreset_wire,
      core_db_lockrq => tta_accel_0_core_db_lockrq_wire,
      core_dmem_avalid_in => tta_accel_0_core_dmem_avalid_in_wire,
      core_dmem_aready_out => tta_accel_0_core_dmem_aready_out_wire,
      core_dmem_aaddr_in => tta_accel_0_core_dmem_aaddr_in_wire,
      core_dmem_awren_in => tta_accel_0_core_dmem_awren_in_wire,
      core_dmem_astrb_in => tta_accel_0_core_dmem_astrb_in_wire,
      core_dmem_adata_in => tta_accel_0_core_dmem_adata_in_wire,
      core_dmem_rvalid_out => tta_accel_0_core_dmem_rvalid_out_wire,
      core_dmem_rready_in => tta_accel_0_core_dmem_rready_in_wire,
      core_dmem_rdata_out => tta_accel_0_core_dmem_rdata_out_wire,
      data_a_avalid_out => tta_accel_0_data_a_avalid_out_wire,
      data_a_aready_in => tta_accel_0_data_a_aready_in_wire,
      data_a_aaddr_out => tta_accel_0_data_a_aaddr_out_wire,
      data_a_awren_out => tta_accel_0_data_a_awren_out_wire,
      data_a_astrb_out => tta_accel_0_data_a_astrb_out_wire,
      data_a_adata_out => tta_accel_0_data_a_adata_out_wire,
      data_a_rvalid_in => tta_accel_0_data_a_rvalid_in_wire,
      data_a_rready_out => tta_accel_0_data_a_rready_out_wire,
      data_a_rdata_in => tta_accel_0_data_a_rdata_in_wire,
      data_b_avalid_out => tta_accel_0_data_b_avalid_out_wire,
      data_b_aready_in => tta_accel_0_data_b_aready_in_wire,
      data_b_aaddr_out => tta_accel_0_data_b_aaddr_out_wire,
      data_b_awren_out => tta_accel_0_data_b_awren_out_wire,
      data_b_astrb_out => tta_accel_0_data_b_astrb_out_wire,
      data_b_adata_out => tta_accel_0_data_b_adata_out_wire,
      data_b_rvalid_in => tta_accel_0_data_b_rvalid_in_wire,
      data_b_rready_out => tta_accel_0_data_b_rready_out_wire,
      data_b_rdata_in => tta_accel_0_data_b_rdata_in_wire,
      core_pmem_avalid_in => tta_accel_0_core_pmem_avalid_in_wire,
      core_pmem_aready_out => tta_accel_0_core_pmem_aready_out_wire,
      core_pmem_aaddr_in => tta_accel_0_core_pmem_aaddr_in_wire,
      core_pmem_awren_in => tta_accel_0_core_pmem_awren_in_wire,
      core_pmem_astrb_in => tta_accel_0_core_pmem_astrb_in_wire,
      core_pmem_adata_in => tta_accel_0_core_pmem_adata_in_wire,
      core_pmem_rvalid_out => tta_accel_0_core_pmem_rvalid_out_wire,
      core_pmem_rready_in => tta_accel_0_core_pmem_rready_in_wire,
      core_pmem_rdata_out => tta_accel_0_core_pmem_rdata_out_wire,
      param_a_avalid_out => tta_accel_0_param_a_avalid_out_wire,
      param_a_aready_in => tta_accel_0_param_a_aready_in_wire,
      param_a_aaddr_out => tta_accel_0_param_a_aaddr_out_wire,
      param_a_awren_out => tta_accel_0_param_a_awren_out_wire,
      param_a_astrb_out => tta_accel_0_param_a_astrb_out_wire,
      param_a_adata_out => tta_accel_0_param_a_adata_out_wire,
      param_a_rvalid_in => tta_accel_0_param_a_rvalid_in_wire,
      param_a_rready_out => tta_accel_0_param_a_rready_out_wire,
      param_a_rdata_in => tta_accel_0_param_a_rdata_in_wire,
      param_b_avalid_out => tta_accel_0_param_b_avalid_out_wire,
      param_b_aready_in => tta_accel_0_param_b_aready_in_wire,
      param_b_aaddr_out => tta_accel_0_param_b_aaddr_out_wire,
      param_b_awren_out => tta_accel_0_param_b_awren_out_wire,
      param_b_astrb_out => tta_accel_0_param_b_astrb_out_wire,
      param_b_adata_out => tta_accel_0_param_b_adata_out_wire,
      param_b_rvalid_in => tta_accel_0_param_b_rvalid_in_wire,
      param_b_rready_out => tta_accel_0_param_b_rready_out_wire,
      param_b_rdata_in => tta_accel_0_param_b_rdata_in_wire,
      aql_read_idx_in => tta_accel_0_aql_read_idx_in_wire,
      aql_read_idx_clear_out => tta_accel_0_aql_read_idx_clear_out_wire);

  imem_array_instance_0 : ffaccel_rom_array_comp
    generic map (
      addrw => IMEMADDRWIDTH,
      instrw => IMEMMAUWIDTH*IMEMWIDTHINMAUS)
    port map (
      clock => clk,
      addr => imem_array_instance_0_addr_wire,
      dataout => imem_array_instance_0_dataout_wire,
      en_x => imem_array_instance_0_en_x_wire);

  onchip_mem_data : xilinx_dp_blockram
    generic map (
      dataw_g => 32,
      addrw_g => 10)
    port map (
      a_aaddr_in => onchip_mem_data_a_aaddr_in_wire,
      a_adata_in => onchip_mem_data_a_adata_in_wire,
      a_aready_out => onchip_mem_data_a_aready_out_wire,
      a_astrb_in => onchip_mem_data_a_astrb_in_wire,
      a_avalid_in => onchip_mem_data_a_avalid_in_wire,
      a_awren_in => onchip_mem_data_a_awren_in_wire,
      a_rdata_out => onchip_mem_data_a_rdata_out_wire,
      a_rready_in => onchip_mem_data_a_rready_in_wire,
      a_rvalid_out => onchip_mem_data_a_rvalid_out_wire,
      b_aaddr_in => onchip_mem_data_b_aaddr_in_wire,
      b_adata_in => onchip_mem_data_b_adata_in_wire,
      b_aready_out => onchip_mem_data_b_aready_out_wire,
      b_astrb_in => onchip_mem_data_b_astrb_in_wire,
      b_avalid_in => onchip_mem_data_b_avalid_in_wire,
      b_awren_in => onchip_mem_data_b_awren_in_wire,
      b_rdata_out => onchip_mem_data_b_rdata_out_wire,
      b_rready_in => onchip_mem_data_b_rready_in_wire,
      b_rvalid_out => onchip_mem_data_b_rvalid_out_wire,
      clk => clk,
      rstx => rstx);

  onchip_mem_param : xilinx_dp_blockram
    generic map (
      dataw_g => 32,
      addrw_g => local_mem_addrw_g)
    port map (
      a_aaddr_in => onchip_mem_param_a_aaddr_in_wire,
      a_adata_in => onchip_mem_param_a_adata_in_wire,
      a_aready_out => onchip_mem_param_a_aready_out_wire,
      a_astrb_in => onchip_mem_param_a_astrb_in_wire,
      a_avalid_in => onchip_mem_param_a_avalid_in_wire,
      a_awren_in => onchip_mem_param_a_awren_in_wire,
      a_rdata_out => onchip_mem_param_a_rdata_out_wire,
      a_rready_in => onchip_mem_param_a_rready_in_wire,
      a_rvalid_out => onchip_mem_param_a_rvalid_out_wire,
      b_aaddr_in => onchip_mem_param_b_aaddr_in_wire,
      b_adata_in => onchip_mem_param_b_adata_in_wire,
      b_aready_out => onchip_mem_param_b_aready_out_wire,
      b_astrb_in => onchip_mem_param_b_astrb_in_wire,
      b_avalid_in => onchip_mem_param_b_avalid_in_wire,
      b_awren_in => onchip_mem_param_b_awren_in_wire,
      b_rdata_out => onchip_mem_param_b_rdata_out_wire,
      b_rready_in => onchip_mem_param_b_rready_in_wire,
      b_rvalid_out => onchip_mem_param_b_rvalid_out_wire,
      clk => clk,
      rstx => rstx);

  onchip_mem_scratchpad : xilinx_blockram
    generic map (
      dataw_g => 32,
      addrw_g => 8)
    port map (
      aaddr_in => onchip_mem_scratchpad_aaddr_in_wire,
      adata_in => onchip_mem_scratchpad_adata_in_wire,
      aready_out => onchip_mem_scratchpad_aready_out_wire,
      astrb_in => onchip_mem_scratchpad_astrb_in_wire,
      avalid_in => onchip_mem_scratchpad_avalid_in_wire,
      awren_in => onchip_mem_scratchpad_awren_in_wire,
      clk => clk,
      rdata_out => onchip_mem_scratchpad_rdata_out_wire,
      rready_in => onchip_mem_scratchpad_rready_in_wire,
      rstx => rstx,
      rvalid_out => onchip_mem_scratchpad_rvalid_out_wire);

end structural;
