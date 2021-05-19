library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.tce_util.all;
use work.tta_core_globals.all;
use work.tta_core_imem_mau.all;

entity tta_core is

  generic (
    core_id : integer := 0);

  port (
    clk : in std_logic;
    rstx : in std_logic;
    busy : in std_logic;
    imem_en_x : out std_logic;
    imem_addr : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    imem_data : in std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
    locked : out std_logic;
    fu_aql_read_idx_out : out std_logic_vector(63 downto 0);
    fu_aql_read_idx_clear_in : in std_logic_vector(0 downto 0);
    db_tta_nreset : in std_logic;
    db_lockcnt : out std_logic_vector(63 downto 0);
    db_cyclecnt : out std_logic_vector(63 downto 0);
    db_pc : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    db_lockrq : in std_logic;
    fu_buffer_avalid_out : out std_logic_vector(0 downto 0);
    fu_buffer_aready_in : in std_logic_vector(0 downto 0);
    fu_buffer_aaddr_out : out std_logic_vector(32-2-1 downto 0);
    fu_buffer_awren_out : out std_logic_vector(0 downto 0);
    fu_buffer_astrb_out : out std_logic_vector(3 downto 0);
    fu_buffer_rvalid_in : in std_logic_vector(0 downto 0);
    fu_buffer_rready_out : out std_logic_vector(0 downto 0);
    fu_buffer_rdata_in : in std_logic_vector(31 downto 0);
    fu_buffer_adata_out : out std_logic_vector(31 downto 0);
    fu_cq_avalid_out : out std_logic_vector(0 downto 0);
    fu_cq_aready_in : in std_logic_vector(0 downto 0);
    fu_cq_aaddr_out : out std_logic_vector(10-2-1 downto 0);
    fu_cq_awren_out : out std_logic_vector(0 downto 0);
    fu_cq_astrb_out : out std_logic_vector(3 downto 0);
    fu_cq_rvalid_in : in std_logic_vector(0 downto 0);
    fu_cq_rready_out : out std_logic_vector(0 downto 0);
    fu_cq_rdata_in : in std_logic_vector(31 downto 0);
    fu_cq_adata_out : out std_logic_vector(31 downto 0));

end tta_core;

architecture structural of tta_core is

  signal decomp_fetch_en_wire : std_logic;
  signal decomp_lock_wire : std_logic;
  signal decomp_fetchblock_wire : std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
  signal decomp_instructionword_wire : std_logic_vector(INSTRUCTIONWIDTH-1 downto 0);
  signal decomp_glock_wire : std_logic;
  signal decomp_lock_r_wire : std_logic;
  signal fu_alu_generated_glock_in_wire : std_logic;
  signal fu_alu_generated_operation_in_wire : std_logic_vector(4-1 downto 0);
  signal fu_alu_generated_glockreq_out_wire : std_logic;
  signal fu_alu_generated_data_in1t_in_wire : std_logic_vector(32-1 downto 0);
  signal fu_alu_generated_load_in1t_in_wire : std_logic;
  signal fu_alu_generated_data_in2_in_wire : std_logic_vector(32-1 downto 0);
  signal fu_alu_generated_load_in2_in_wire : std_logic;
  signal fu_alu_generated_data_out1_out_wire : std_logic_vector(32-1 downto 0);
  signal fu_aql_t1_data_in_wire : std_logic_vector(31 downto 0);
  signal fu_aql_t1_load_in_wire : std_logic;
  signal fu_aql_r1_data_out_wire : std_logic_vector(31 downto 0);
  signal fu_aql_t1_opcode_in_wire : std_logic_vector(0 downto 0);
  signal fu_aql_glock_wire : std_logic;
  signal fu_buffer_generated_glock_in_wire : std_logic;
  signal fu_buffer_generated_operation_in_wire : std_logic_vector(3-1 downto 0);
  signal fu_buffer_generated_glockreq_out_wire : std_logic;
  signal fu_buffer_generated_data_in1t_in_wire : std_logic_vector(32-1 downto 0);
  signal fu_buffer_generated_load_in1t_in_wire : std_logic;
  signal fu_buffer_generated_data_out1_out_wire : std_logic_vector(32-1 downto 0);
  signal fu_buffer_generated_data_in2_in_wire : std_logic_vector(32-1 downto 0);
  signal fu_buffer_generated_load_in2_in_wire : std_logic;
  signal fu_cq_generated_glock_in_wire : std_logic;
  signal fu_cq_generated_operation_in_wire : std_logic_vector(3-1 downto 0);
  signal fu_cq_generated_glockreq_out_wire : std_logic;
  signal fu_cq_generated_data_in1t_in_wire : std_logic_vector(32-1 downto 0);
  signal fu_cq_generated_load_in1t_in_wire : std_logic;
  signal fu_cq_generated_data_out1_out_wire : std_logic_vector(32-1 downto 0);
  signal fu_cq_generated_data_in2_in_wire : std_logic_vector(32-1 downto 0);
  signal fu_cq_generated_load_in2_in_wire : std_logic;
  signal ic_glock_wire : std_logic;
  signal ic_socket_lsu_i1_data_wire : std_logic_vector(31 downto 0);
  signal ic_socket_lsu_i2_data_wire : std_logic_vector(31 downto 0);
  signal ic_socket_alu_comp_i1_data_wire : std_logic_vector(31 downto 0);
  signal ic_socket_alu_comp_i2_data_wire : std_logic_vector(31 downto 0);
  signal ic_socket_RF_i1_data_wire : std_logic_vector(31 downto 0);
  signal ic_socket_bool_i1_data_wire : std_logic_vector(0 downto 0);
  signal ic_socket_gcu_i1_data_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal ic_socket_gcu_i2_data_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal ic_socket_get_read_idx_low_inc_read_idx_i1_data_wire : std_logic_vector(31 downto 0);
  signal ic_socket_lsu_1_i1_data_wire : std_logic_vector(31 downto 0);
  signal ic_socket_lsu_1_i2_data_wire : std_logic_vector(31 downto 0);
  signal ic_B1_mux_ctrl_in_wire : std_logic_vector(2 downto 0);
  signal ic_B1_data_0_in_wire : std_logic_vector(32-1 downto 0);
  signal ic_B1_data_1_in_wire : std_logic_vector(32-1 downto 0);
  signal ic_B1_data_2_in_wire : std_logic_vector(31 downto 0);
  signal ic_B1_data_3_in_wire : std_logic_vector(0 downto 0);
  signal ic_B1_data_4_in_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal ic_B1_data_5_in_wire : std_logic_vector(31 downto 0);
  signal ic_B1_data_6_in_wire : std_logic_vector(32-1 downto 0);
  signal ic_simm_B1_wire : std_logic_vector(31 downto 0);
  signal ic_simm_cntrl_B1_wire : std_logic_vector(0 downto 0);
  signal inst_decoder_instructionword_wire : std_logic_vector(INSTRUCTIONWIDTH-1 downto 0);
  signal inst_decoder_pc_load_wire : std_logic;
  signal inst_decoder_ra_load_wire : std_logic;
  signal inst_decoder_pc_opcode_wire : std_logic_vector(0 downto 0);
  signal inst_decoder_lock_wire : std_logic;
  signal inst_decoder_lock_r_wire : std_logic;
  signal inst_decoder_simm_B1_wire : std_logic_vector(31 downto 0);
  signal inst_decoder_B1_src_sel_wire : std_logic_vector(2 downto 0);
  signal inst_decoder_fu_buffer_in1t_load_wire : std_logic;
  signal inst_decoder_fu_buffer_in2_load_wire : std_logic;
  signal inst_decoder_fu_buffer_opc_wire : std_logic_vector(2 downto 0);
  signal inst_decoder_fu_alu_in1t_load_wire : std_logic;
  signal inst_decoder_fu_alu_in2_load_wire : std_logic;
  signal inst_decoder_fu_alu_opc_wire : std_logic_vector(3 downto 0);
  signal inst_decoder_fu_aql_in1t_load_wire : std_logic;
  signal inst_decoder_fu_aql_opc_wire : std_logic_vector(0 downto 0);
  signal inst_decoder_fu_cq_in1t_load_wire : std_logic;
  signal inst_decoder_fu_cq_in2_load_wire : std_logic;
  signal inst_decoder_fu_cq_opc_wire : std_logic_vector(2 downto 0);
  signal inst_decoder_rf_RF_wr_load_wire : std_logic;
  signal inst_decoder_rf_RF_wr_opc_wire : std_logic_vector(3 downto 0);
  signal inst_decoder_rf_RF_rd_load_wire : std_logic;
  signal inst_decoder_rf_RF_rd_opc_wire : std_logic_vector(3 downto 0);
  signal inst_decoder_rf_bool_wr_load_wire : std_logic;
  signal inst_decoder_rf_bool_wr_opc_wire : std_logic_vector(0 downto 0);
  signal inst_decoder_rf_bool_rd_load_wire : std_logic;
  signal inst_decoder_rf_bool_rd_opc_wire : std_logic_vector(0 downto 0);
  signal inst_decoder_rf_guard_bool_0_wire : std_logic;
  signal inst_decoder_rf_guard_bool_1_wire : std_logic;
  signal inst_decoder_lock_req_wire : std_logic_vector(3 downto 0);
  signal inst_decoder_glock_wire : std_logic_vector(6 downto 0);
  signal inst_fetch_ra_out_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal inst_fetch_ra_in_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal inst_fetch_pc_in_wire : std_logic_vector(IMEMADDRWIDTH-1 downto 0);
  signal inst_fetch_pc_load_wire : std_logic;
  signal inst_fetch_ra_load_wire : std_logic;
  signal inst_fetch_pc_opcode_wire : std_logic_vector(0 downto 0);
  signal inst_fetch_fetch_en_wire : std_logic;
  signal inst_fetch_glock_wire : std_logic;
  signal inst_fetch_fetchblock_wire : std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
  signal rf_RF_data_rd_out_wire : std_logic_vector(31 downto 0);
  signal rf_RF_load_rd_in_wire : std_logic;
  signal rf_RF_addr_rd_in_wire : std_logic_vector(3 downto 0);
  signal rf_RF_data_wr_in_wire : std_logic_vector(31 downto 0);
  signal rf_RF_load_wr_in_wire : std_logic;
  signal rf_RF_addr_wr_in_wire : std_logic_vector(3 downto 0);
  signal rf_RF_glock_in_wire : std_logic;
  signal rf_bool_t1data_wire : std_logic_vector(0 downto 0);
  signal rf_bool_t1load_wire : std_logic;
  signal rf_bool_t1opcode_wire : std_logic_vector(0 downto 0);
  signal rf_bool_r1data_wire : std_logic_vector(0 downto 0);
  signal rf_bool_r1load_wire : std_logic;
  signal rf_bool_r1opcode_wire : std_logic_vector(0 downto 0);
  signal rf_bool_guard_wire : std_logic_vector(1 downto 0);
  signal rf_bool_glock_wire : std_logic;
  signal ground_signal : std_logic_vector(0 downto 0);

  component tta_core_ifetch is
    generic (
      debug_logic_g : boolean);
    port (
      clk : in std_logic;
      rstx : in std_logic;
      ra_out : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      ra_in : in std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      busy : in std_logic;
      imem_en_x : out std_logic;
      imem_addr : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      imem_data : in std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
      pc_in : in std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      pc_load : in std_logic;
      ra_load : in std_logic;
      pc_opcode : in std_logic_vector(1-1 downto 0);
      fetch_en : in std_logic;
      glock : out std_logic;
      fetchblock : out std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
      db_rstx : in std_logic;
      db_lockreq : in std_logic;
      db_cyclecnt : out std_logic_vector(64-1 downto 0);
      db_lockcnt : out std_logic_vector(64-1 downto 0);
      db_pc : out std_logic_vector(IMEMADDRWIDTH-1 downto 0));
  end component;

  component tta_core_decompressor is
    port (
      fetch_en : out std_logic;
      lock : in std_logic;
      fetchblock : in std_logic_vector(IMEMWIDTHINMAUS*IMEMMAUWIDTH-1 downto 0);
      clk : in std_logic;
      rstx : in std_logic;
      instructionword : out std_logic_vector(INSTRUCTIONWIDTH-1 downto 0);
      glock : out std_logic;
      lock_r : in std_logic);
  end component;

  component tta_core_decoder is
    port (
      instructionword : in std_logic_vector(INSTRUCTIONWIDTH-1 downto 0);
      pc_load : out std_logic;
      ra_load : out std_logic;
      pc_opcode : out std_logic_vector(1-1 downto 0);
      lock : in std_logic;
      lock_r : out std_logic;
      clk : in std_logic;
      rstx : in std_logic;
      locked : out std_logic;
      simm_B1 : out std_logic_vector(32-1 downto 0);
      B1_src_sel : out std_logic_vector(3-1 downto 0);
      fu_buffer_in1t_load : out std_logic;
      fu_buffer_in2_load : out std_logic;
      fu_buffer_opc : out std_logic_vector(3-1 downto 0);
      fu_alu_in1t_load : out std_logic;
      fu_alu_in2_load : out std_logic;
      fu_alu_opc : out std_logic_vector(4-1 downto 0);
      fu_aql_in1t_load : out std_logic;
      fu_aql_opc : out std_logic_vector(1-1 downto 0);
      fu_cq_in1t_load : out std_logic;
      fu_cq_in2_load : out std_logic;
      fu_cq_opc : out std_logic_vector(3-1 downto 0);
      rf_RF_wr_load : out std_logic;
      rf_RF_wr_opc : out std_logic_vector(4-1 downto 0);
      rf_RF_rd_load : out std_logic;
      rf_RF_rd_opc : out std_logic_vector(4-1 downto 0);
      rf_bool_wr_load : out std_logic;
      rf_bool_wr_opc : out std_logic_vector(1-1 downto 0);
      rf_bool_rd_load : out std_logic;
      rf_bool_rd_opc : out std_logic_vector(1-1 downto 0);
      rf_guard_bool_0 : in std_logic;
      rf_guard_bool_1 : in std_logic;
      lock_req : in std_logic_vector(4-1 downto 0);
      glock : out std_logic_vector(7-1 downto 0);
      db_tta_nreset : in std_logic);
  end component;

  component fu_buffer is
    port (
      clk : in std_logic;
      rstx : in std_logic;
      glock_in : in std_logic;
      operation_in : in std_logic_vector(3-1 downto 0);
      glockreq_out : out std_logic;
      data_in1t_in : in std_logic_vector(32-1 downto 0);
      load_in1t_in : in std_logic;
      data_out1_out : out std_logic_vector(32-1 downto 0);
      data_in2_in : in std_logic_vector(32-1 downto 0);
      load_in2_in : in std_logic;
      avalid_out : out std_logic_vector(1-1 downto 0);
      aready_in : in std_logic_vector(1-1 downto 0);
      aaddr_out : out std_logic_vector(32-2-1 downto 0);
      awren_out : out std_logic_vector(1-1 downto 0);
      astrb_out : out std_logic_vector(4-1 downto 0);
      rvalid_in : in std_logic_vector(1-1 downto 0);
      rready_out : out std_logic_vector(1-1 downto 0);
      rdata_in : in std_logic_vector(32-1 downto 0);
      adata_out : out std_logic_vector(32-1 downto 0));
  end component;

  component fu_alu is
    port (
      clk : in std_logic;
      rstx : in std_logic;
      glock_in : in std_logic;
      operation_in : in std_logic_vector(4-1 downto 0);
      glockreq_out : out std_logic;
      data_in1t_in : in std_logic_vector(32-1 downto 0);
      load_in1t_in : in std_logic;
      data_in2_in : in std_logic_vector(32-1 downto 0);
      load_in2_in : in std_logic;
      data_out1_out : out std_logic_vector(32-1 downto 0));
  end component;

  component fu_cq is
    port (
      clk : in std_logic;
      rstx : in std_logic;
      glock_in : in std_logic;
      operation_in : in std_logic_vector(3-1 downto 0);
      glockreq_out : out std_logic;
      data_in1t_in : in std_logic_vector(32-1 downto 0);
      load_in1t_in : in std_logic;
      data_out1_out : out std_logic_vector(32-1 downto 0);
      data_in2_in : in std_logic_vector(32-1 downto 0);
      load_in2_in : in std_logic;
      avalid_out : out std_logic_vector(1-1 downto 0);
      aready_in : in std_logic_vector(1-1 downto 0);
      aaddr_out : out std_logic_vector(10-2-1 downto 0);
      awren_out : out std_logic_vector(1-1 downto 0);
      astrb_out : out std_logic_vector(4-1 downto 0);
      rvalid_in : in std_logic_vector(1-1 downto 0);
      rready_out : out std_logic_vector(1-1 downto 0);
      rdata_in : in std_logic_vector(32-1 downto 0);
      adata_out : out std_logic_vector(32-1 downto 0));
  end component;

  component fu_aql_minimal is
    port (
      t1_data_in : in std_logic_vector(32-1 downto 0);
      t1_load_in : in std_logic;
      r1_data_out : out std_logic_vector(32-1 downto 0);
      t1_opcode_in : in std_logic_vector(1-1 downto 0);
      read_idx_out : out std_logic_vector(64-1 downto 0);
      read_idx_clear_in : in std_logic_vector(1-1 downto 0);
      clk : in std_logic;
      rstx : in std_logic;
      glock : in std_logic);
  end component;

  component s7_rf_1wr_1rd is
    generic (
      width_g : integer;
      depth_g : integer);
    port (
      data_rd_out : out std_logic_vector(width_g-1 downto 0);
      load_rd_in : in std_logic;
      addr_rd_in : in std_logic_vector(bit_width(depth_g)-1 downto 0);
      data_wr_in : in std_logic_vector(width_g-1 downto 0);
      load_wr_in : in std_logic;
      addr_wr_in : in std_logic_vector(bit_width(depth_g)-1 downto 0);
      clk : in std_logic;
      rstx : in std_logic;
      glock_in : in std_logic);
  end component;

  component rf_1wr_1rd_always_1_guarded_0 is
    generic (
      dataw : integer;
      rf_size : integer);
    port (
      t1data : in std_logic_vector(dataw-1 downto 0);
      t1load : in std_logic;
      t1opcode : in std_logic_vector(bit_width(rf_size)-1 downto 0);
      r1data : out std_logic_vector(dataw-1 downto 0);
      r1load : in std_logic;
      r1opcode : in std_logic_vector(bit_width(rf_size)-1 downto 0);
      guard : out std_logic_vector(rf_size-1 downto 0);
      clk : in std_logic;
      rstx : in std_logic;
      glock : in std_logic);
  end component;

  component tta_core_interconn is
    port (
      clk : in std_logic;
      rstx : in std_logic;
      glock : in std_logic;
      socket_lsu_i1_data : out std_logic_vector(32-1 downto 0);
      socket_lsu_i2_data : out std_logic_vector(32-1 downto 0);
      socket_alu_comp_i1_data : out std_logic_vector(32-1 downto 0);
      socket_alu_comp_i2_data : out std_logic_vector(32-1 downto 0);
      socket_RF_i1_data : out std_logic_vector(32-1 downto 0);
      socket_bool_i1_data : out std_logic_vector(1-1 downto 0);
      socket_gcu_i1_data : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      socket_gcu_i2_data : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      socket_get_read_idx_low_inc_read_idx_i1_data : out std_logic_vector(32-1 downto 0);
      socket_lsu_1_i1_data : out std_logic_vector(32-1 downto 0);
      socket_lsu_1_i2_data : out std_logic_vector(32-1 downto 0);
      B1_mux_ctrl_in : in std_logic_vector(3-1 downto 0);
      B1_data_0_in : in std_logic_vector(32-1 downto 0);
      B1_data_1_in : in std_logic_vector(32-1 downto 0);
      B1_data_2_in : in std_logic_vector(32-1 downto 0);
      B1_data_3_in : in std_logic_vector(1-1 downto 0);
      B1_data_4_in : in std_logic_vector(IMEMADDRWIDTH-1 downto 0);
      B1_data_5_in : in std_logic_vector(32-1 downto 0);
      B1_data_6_in : in std_logic_vector(32-1 downto 0);
      simm_B1 : in std_logic_vector(32-1 downto 0);
      simm_cntrl_B1 : in std_logic_vector(1-1 downto 0));
  end component;


begin

  ic_B1_data_4_in_wire <= inst_fetch_ra_out_wire;
  inst_fetch_ra_in_wire <= ic_socket_gcu_i2_data_wire;
  inst_fetch_pc_in_wire <= ic_socket_gcu_i1_data_wire;
  inst_fetch_pc_load_wire <= inst_decoder_pc_load_wire;
  inst_fetch_ra_load_wire <= inst_decoder_ra_load_wire;
  inst_fetch_pc_opcode_wire <= inst_decoder_pc_opcode_wire;
  inst_fetch_fetch_en_wire <= decomp_fetch_en_wire;
  decomp_lock_wire <= inst_fetch_glock_wire;
  decomp_fetchblock_wire <= inst_fetch_fetchblock_wire;
  inst_decoder_instructionword_wire <= decomp_instructionword_wire;
  inst_decoder_lock_wire <= decomp_glock_wire;
  decomp_lock_r_wire <= inst_decoder_lock_r_wire;
  ic_simm_B1_wire <= inst_decoder_simm_B1_wire;
  ic_B1_mux_ctrl_in_wire <= inst_decoder_B1_src_sel_wire;
  fu_buffer_generated_load_in1t_in_wire <= inst_decoder_fu_buffer_in1t_load_wire;
  fu_buffer_generated_load_in2_in_wire <= inst_decoder_fu_buffer_in2_load_wire;
  fu_buffer_generated_operation_in_wire <= inst_decoder_fu_buffer_opc_wire;
  fu_alu_generated_load_in1t_in_wire <= inst_decoder_fu_alu_in1t_load_wire;
  fu_alu_generated_load_in2_in_wire <= inst_decoder_fu_alu_in2_load_wire;
  fu_alu_generated_operation_in_wire <= inst_decoder_fu_alu_opc_wire;
  fu_aql_t1_load_in_wire <= inst_decoder_fu_aql_in1t_load_wire;
  fu_aql_t1_opcode_in_wire <= inst_decoder_fu_aql_opc_wire;
  fu_cq_generated_load_in1t_in_wire <= inst_decoder_fu_cq_in1t_load_wire;
  fu_cq_generated_load_in2_in_wire <= inst_decoder_fu_cq_in2_load_wire;
  fu_cq_generated_operation_in_wire <= inst_decoder_fu_cq_opc_wire;
  rf_RF_load_wr_in_wire <= inst_decoder_rf_RF_wr_load_wire;
  rf_RF_addr_wr_in_wire <= inst_decoder_rf_RF_wr_opc_wire;
  rf_RF_load_rd_in_wire <= inst_decoder_rf_RF_rd_load_wire;
  rf_RF_addr_rd_in_wire <= inst_decoder_rf_RF_rd_opc_wire;
  rf_bool_t1load_wire <= inst_decoder_rf_bool_wr_load_wire;
  rf_bool_t1opcode_wire <= inst_decoder_rf_bool_wr_opc_wire;
  rf_bool_r1load_wire <= inst_decoder_rf_bool_rd_load_wire;
  rf_bool_r1opcode_wire <= inst_decoder_rf_bool_rd_opc_wire;
  inst_decoder_rf_guard_bool_0_wire <= rf_bool_guard_wire(0);
  inst_decoder_rf_guard_bool_1_wire <= rf_bool_guard_wire(1);
  inst_decoder_lock_req_wire(0) <= fu_buffer_generated_glockreq_out_wire;
  inst_decoder_lock_req_wire(1) <= fu_alu_generated_glockreq_out_wire;
  inst_decoder_lock_req_wire(2) <= fu_cq_generated_glockreq_out_wire;
  inst_decoder_lock_req_wire(3) <= db_lockrq;
  fu_buffer_generated_glock_in_wire <= inst_decoder_glock_wire(0);
  fu_alu_generated_glock_in_wire <= inst_decoder_glock_wire(1);
  fu_aql_glock_wire <= inst_decoder_glock_wire(2);
  fu_cq_generated_glock_in_wire <= inst_decoder_glock_wire(3);
  rf_RF_glock_in_wire <= inst_decoder_glock_wire(4);
  rf_bool_glock_wire <= inst_decoder_glock_wire(5);
  ic_glock_wire <= inst_decoder_glock_wire(6);
  fu_buffer_generated_data_in1t_in_wire <= ic_socket_lsu_i1_data_wire;
  ic_B1_data_0_in_wire <= fu_buffer_generated_data_out1_out_wire;
  fu_buffer_generated_data_in2_in_wire <= ic_socket_lsu_i2_data_wire;
  fu_alu_generated_data_in1t_in_wire <= ic_socket_alu_comp_i1_data_wire;
  fu_alu_generated_data_in2_in_wire <= ic_socket_alu_comp_i2_data_wire;
  ic_B1_data_1_in_wire <= fu_alu_generated_data_out1_out_wire;
  fu_cq_generated_data_in1t_in_wire <= ic_socket_lsu_1_i1_data_wire;
  ic_B1_data_6_in_wire <= fu_cq_generated_data_out1_out_wire;
  fu_cq_generated_data_in2_in_wire <= ic_socket_lsu_1_i2_data_wire;
  fu_aql_t1_data_in_wire <= ic_socket_get_read_idx_low_inc_read_idx_i1_data_wire;
  ic_B1_data_5_in_wire <= fu_aql_r1_data_out_wire;
  ic_B1_data_2_in_wire <= rf_RF_data_rd_out_wire;
  rf_RF_data_wr_in_wire <= ic_socket_RF_i1_data_wire;
  rf_bool_t1data_wire <= ic_socket_bool_i1_data_wire;
  ic_B1_data_3_in_wire <= rf_bool_r1data_wire;
  ground_signal <= (others => '0');

  inst_fetch : tta_core_ifetch
    generic map (
      debug_logic_g => true)
    port map (
      clk => clk,
      rstx => rstx,
      ra_out => inst_fetch_ra_out_wire,
      ra_in => inst_fetch_ra_in_wire,
      busy => busy,
      imem_en_x => imem_en_x,
      imem_addr => imem_addr,
      imem_data => imem_data,
      pc_in => inst_fetch_pc_in_wire,
      pc_load => inst_fetch_pc_load_wire,
      ra_load => inst_fetch_ra_load_wire,
      pc_opcode => inst_fetch_pc_opcode_wire,
      fetch_en => inst_fetch_fetch_en_wire,
      glock => inst_fetch_glock_wire,
      fetchblock => inst_fetch_fetchblock_wire,
      db_rstx => db_tta_nreset,
      db_lockreq => db_lockrq,
      db_cyclecnt => db_cyclecnt,
      db_lockcnt => db_lockcnt,
      db_pc => db_pc);

  decomp : tta_core_decompressor
    port map (
      fetch_en => decomp_fetch_en_wire,
      lock => decomp_lock_wire,
      fetchblock => decomp_fetchblock_wire,
      clk => clk,
      rstx => rstx,
      instructionword => decomp_instructionword_wire,
      glock => decomp_glock_wire,
      lock_r => decomp_lock_r_wire);

  inst_decoder : tta_core_decoder
    port map (
      instructionword => inst_decoder_instructionword_wire,
      pc_load => inst_decoder_pc_load_wire,
      ra_load => inst_decoder_ra_load_wire,
      pc_opcode => inst_decoder_pc_opcode_wire,
      lock => inst_decoder_lock_wire,
      lock_r => inst_decoder_lock_r_wire,
      clk => clk,
      rstx => rstx,
      locked => locked,
      simm_B1 => inst_decoder_simm_B1_wire,
      B1_src_sel => inst_decoder_B1_src_sel_wire,
      fu_buffer_in1t_load => inst_decoder_fu_buffer_in1t_load_wire,
      fu_buffer_in2_load => inst_decoder_fu_buffer_in2_load_wire,
      fu_buffer_opc => inst_decoder_fu_buffer_opc_wire,
      fu_alu_in1t_load => inst_decoder_fu_alu_in1t_load_wire,
      fu_alu_in2_load => inst_decoder_fu_alu_in2_load_wire,
      fu_alu_opc => inst_decoder_fu_alu_opc_wire,
      fu_aql_in1t_load => inst_decoder_fu_aql_in1t_load_wire,
      fu_aql_opc => inst_decoder_fu_aql_opc_wire,
      fu_cq_in1t_load => inst_decoder_fu_cq_in1t_load_wire,
      fu_cq_in2_load => inst_decoder_fu_cq_in2_load_wire,
      fu_cq_opc => inst_decoder_fu_cq_opc_wire,
      rf_RF_wr_load => inst_decoder_rf_RF_wr_load_wire,
      rf_RF_wr_opc => inst_decoder_rf_RF_wr_opc_wire,
      rf_RF_rd_load => inst_decoder_rf_RF_rd_load_wire,
      rf_RF_rd_opc => inst_decoder_rf_RF_rd_opc_wire,
      rf_bool_wr_load => inst_decoder_rf_bool_wr_load_wire,
      rf_bool_wr_opc => inst_decoder_rf_bool_wr_opc_wire,
      rf_bool_rd_load => inst_decoder_rf_bool_rd_load_wire,
      rf_bool_rd_opc => inst_decoder_rf_bool_rd_opc_wire,
      rf_guard_bool_0 => inst_decoder_rf_guard_bool_0_wire,
      rf_guard_bool_1 => inst_decoder_rf_guard_bool_1_wire,
      lock_req => inst_decoder_lock_req_wire,
      glock => inst_decoder_glock_wire,
      db_tta_nreset => db_tta_nreset);

  fu_buffer_generated : fu_buffer
    port map (
      clk => clk,
      rstx => rstx,
      glock_in => fu_buffer_generated_glock_in_wire,
      operation_in => fu_buffer_generated_operation_in_wire,
      glockreq_out => fu_buffer_generated_glockreq_out_wire,
      data_in1t_in => fu_buffer_generated_data_in1t_in_wire,
      load_in1t_in => fu_buffer_generated_load_in1t_in_wire,
      data_out1_out => fu_buffer_generated_data_out1_out_wire,
      data_in2_in => fu_buffer_generated_data_in2_in_wire,
      load_in2_in => fu_buffer_generated_load_in2_in_wire,
      avalid_out => fu_buffer_avalid_out,
      aready_in => fu_buffer_aready_in,
      aaddr_out => fu_buffer_aaddr_out,
      awren_out => fu_buffer_awren_out,
      astrb_out => fu_buffer_astrb_out,
      rvalid_in => fu_buffer_rvalid_in,
      rready_out => fu_buffer_rready_out,
      rdata_in => fu_buffer_rdata_in,
      adata_out => fu_buffer_adata_out);

  fu_alu_generated : fu_alu
    port map (
      clk => clk,
      rstx => rstx,
      glock_in => fu_alu_generated_glock_in_wire,
      operation_in => fu_alu_generated_operation_in_wire,
      glockreq_out => fu_alu_generated_glockreq_out_wire,
      data_in1t_in => fu_alu_generated_data_in1t_in_wire,
      load_in1t_in => fu_alu_generated_load_in1t_in_wire,
      data_in2_in => fu_alu_generated_data_in2_in_wire,
      load_in2_in => fu_alu_generated_load_in2_in_wire,
      data_out1_out => fu_alu_generated_data_out1_out_wire);

  fu_cq_generated : fu_cq
    port map (
      clk => clk,
      rstx => rstx,
      glock_in => fu_cq_generated_glock_in_wire,
      operation_in => fu_cq_generated_operation_in_wire,
      glockreq_out => fu_cq_generated_glockreq_out_wire,
      data_in1t_in => fu_cq_generated_data_in1t_in_wire,
      load_in1t_in => fu_cq_generated_load_in1t_in_wire,
      data_out1_out => fu_cq_generated_data_out1_out_wire,
      data_in2_in => fu_cq_generated_data_in2_in_wire,
      load_in2_in => fu_cq_generated_load_in2_in_wire,
      avalid_out => fu_cq_avalid_out,
      aready_in => fu_cq_aready_in,
      aaddr_out => fu_cq_aaddr_out,
      awren_out => fu_cq_awren_out,
      astrb_out => fu_cq_astrb_out,
      rvalid_in => fu_cq_rvalid_in,
      rready_out => fu_cq_rready_out,
      rdata_in => fu_cq_rdata_in,
      adata_out => fu_cq_adata_out);

  fu_aql : fu_aql_minimal
    port map (
      t1_data_in => fu_aql_t1_data_in_wire,
      t1_load_in => fu_aql_t1_load_in_wire,
      r1_data_out => fu_aql_r1_data_out_wire,
      t1_opcode_in => fu_aql_t1_opcode_in_wire,
      read_idx_out => fu_aql_read_idx_out,
      read_idx_clear_in => fu_aql_read_idx_clear_in,
      clk => clk,
      rstx => rstx,
      glock => fu_aql_glock_wire);

  rf_RF : s7_rf_1wr_1rd
    generic map (
      width_g => 32,
      depth_g => 16)
    port map (
      data_rd_out => rf_RF_data_rd_out_wire,
      load_rd_in => rf_RF_load_rd_in_wire,
      addr_rd_in => rf_RF_addr_rd_in_wire,
      data_wr_in => rf_RF_data_wr_in_wire,
      load_wr_in => rf_RF_load_wr_in_wire,
      addr_wr_in => rf_RF_addr_wr_in_wire,
      clk => clk,
      rstx => rstx,
      glock_in => rf_RF_glock_in_wire);

  rf_bool : rf_1wr_1rd_always_1_guarded_0
    generic map (
      dataw => 1,
      rf_size => 2)
    port map (
      t1data => rf_bool_t1data_wire,
      t1load => rf_bool_t1load_wire,
      t1opcode => rf_bool_t1opcode_wire,
      r1data => rf_bool_r1data_wire,
      r1load => rf_bool_r1load_wire,
      r1opcode => rf_bool_r1opcode_wire,
      guard => rf_bool_guard_wire,
      clk => clk,
      rstx => rstx,
      glock => rf_bool_glock_wire);

  ic : tta_core_interconn
    port map (
      clk => clk,
      rstx => rstx,
      glock => ic_glock_wire,
      socket_lsu_i1_data => ic_socket_lsu_i1_data_wire,
      socket_lsu_i2_data => ic_socket_lsu_i2_data_wire,
      socket_alu_comp_i1_data => ic_socket_alu_comp_i1_data_wire,
      socket_alu_comp_i2_data => ic_socket_alu_comp_i2_data_wire,
      socket_RF_i1_data => ic_socket_RF_i1_data_wire,
      socket_bool_i1_data => ic_socket_bool_i1_data_wire,
      socket_gcu_i1_data => ic_socket_gcu_i1_data_wire,
      socket_gcu_i2_data => ic_socket_gcu_i2_data_wire,
      socket_get_read_idx_low_inc_read_idx_i1_data => ic_socket_get_read_idx_low_inc_read_idx_i1_data_wire,
      socket_lsu_1_i1_data => ic_socket_lsu_1_i1_data_wire,
      socket_lsu_1_i2_data => ic_socket_lsu_1_i2_data_wire,
      B1_mux_ctrl_in => ic_B1_mux_ctrl_in_wire,
      B1_data_0_in => ic_B1_data_0_in_wire,
      B1_data_1_in => ic_B1_data_1_in_wire,
      B1_data_2_in => ic_B1_data_2_in_wire,
      B1_data_3_in => ic_B1_data_3_in_wire,
      B1_data_4_in => ic_B1_data_4_in_wire,
      B1_data_5_in => ic_B1_data_5_in_wire,
      B1_data_6_in => ic_B1_data_6_in_wire,
      simm_B1 => ic_simm_B1_wire,
      simm_cntrl_B1 => ic_simm_cntrl_B1_wire);

end structural;
