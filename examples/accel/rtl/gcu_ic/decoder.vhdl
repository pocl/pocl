library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.ffaccel_globals.all;
use work.ffaccel_gcu_opcodes.all;
use work.tce_util.all;

entity ffaccel_decoder is

  port (
    instructionword : in std_logic_vector(INSTRUCTIONWIDTH-1 downto 0);
    pc_load : out std_logic;
    ra_load : out std_logic;
    pc_opcode : out std_logic_vector(0 downto 0);
    lock : in std_logic;
    lock_r : out std_logic;
    clk : in std_logic;
    rstx : in std_logic;
    locked : out std_logic;
    simm_B1 : out std_logic_vector(31 downto 0);
    simm_B1_1 : out std_logic_vector(31 downto 0);
    socket_lsu_i1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_lsu_i2_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_alu_comp_i1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_alu_comp_i2_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_RF_i1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_bool_i1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_gcu_i1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_gcu_i2_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_lsu_i1_1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_lsu_i2_1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_lsu_i2_1_1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_lsu_i1_1_1_bus_cntrl : out std_logic_vector(0 downto 0);
    socket_lsu_i2_1_1_2_1_bus_cntrl : out std_logic_vector(0 downto 0);
    B1_src_sel : out std_logic_vector(3 downto 0);
    B1_1_src_sel : out std_logic_vector(3 downto 0);
    fu_DATA_LSU_in1t_load : out std_logic;
    fu_DATA_LSU_in2_load : out std_logic;
    fu_DATA_LSU_opc : out std_logic_vector(2 downto 0);
    fu_alu_comp_in1t_load : out std_logic;
    fu_alu_comp_in2_load : out std_logic;
    fu_alu_comp_opc : out std_logic_vector(3 downto 0);
    fu_PARAM_LSU_in1t_load : out std_logic;
    fu_PARAM_LSU_in2_load : out std_logic;
    fu_PARAM_LSU_opc : out std_logic_vector(2 downto 0);
    fu_SP_LSU_in1t_load : out std_logic;
    fu_SP_LSU_in2_load : out std_logic;
    fu_SP_LSU_opc : out std_logic_vector(2 downto 0);
    fu_AQL_FU_t1_in_load : out std_logic;
    fu_AQL_FU_opc : out std_logic_vector(0 downto 0);
    rf_RF_wr_load : out std_logic;
    rf_RF_wr_opc : out std_logic_vector(4 downto 0);
    rf_RF_rd_load : out std_logic;
    rf_RF_rd_opc : out std_logic_vector(4 downto 0);
    rf_bool_wr_load : out std_logic;
    rf_bool_wr_opc : out std_logic_vector(0 downto 0);
    rf_bool_rd_load : out std_logic;
    rf_bool_rd_opc : out std_logic_vector(0 downto 0);
    iu_IMM_P1_read_load : out std_logic;
    iu_IMM_P1_read_opc : out std_logic_vector(0 downto 0);
    iu_IMM_write : out std_logic_vector(31 downto 0);
    iu_IMM_write_load : out std_logic;
    iu_IMM_write_opc : out std_logic_vector(0 downto 0);
    rf_guard_bool_0 : in std_logic;
    rf_guard_bool_1 : in std_logic;
    lock_req : in std_logic_vector(4 downto 0);
    glock : out std_logic_vector(8 downto 0);
    db_tta_nreset : in std_logic);

end ffaccel_decoder;

architecture rtl_andor of ffaccel_decoder is

  -- signals for source, destination and guard fields
  signal move_B1 : std_logic_vector(23 downto 0);
  signal src_B1 : std_logic_vector(13 downto 0);
  signal dst_B1 : std_logic_vector(6 downto 0);
  signal grd_B1 : std_logic_vector(2 downto 0);
  signal move_B1_1 : std_logic_vector(17 downto 0);
  signal src_B1_1 : std_logic_vector(7 downto 0);
  signal dst_B1_1 : std_logic_vector(6 downto 0);
  signal grd_B1_1 : std_logic_vector(2 downto 0);

  -- signals for dedicated immediate slots

  -- signal for long immediate tag
  signal limm_tag : std_logic_vector(0 downto 0);

  -- squash signals
  signal squash_B1 : std_logic;
  signal squash_B1_1 : std_logic;

  -- socket control signals
  signal socket_lsu_i1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_lsu_o1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_lsu_i2_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_alu_comp_i1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_alu_comp_i2_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_alu_comp_o1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_RF_i1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_RF_o1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_bool_i1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_bool_o1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_gcu_i1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_gcu_i2_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_gcu_o1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_lsu_i1_1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_lsu_o1_1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_lsu_i2_1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_RF_o1_1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_lsu_i2_1_1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_lsu_i1_1_1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_lsu_o1_1_1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_alu_comp_o1_1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_lsu_o1_1_1_1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal socket_lsu_i2_1_1_2_1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_lsu_o1_1_1_1_1_bus_cntrl_reg : std_logic_vector(1 downto 0);
  signal simm_B1_reg : std_logic_vector(31 downto 0);
  signal B1_src_sel_reg : std_logic_vector(3 downto 0);
  signal simm_B1_1_reg : std_logic_vector(31 downto 0);
  signal B1_1_src_sel_reg : std_logic_vector(3 downto 0);

  -- FU control signals
  signal fu_DATA_LSU_in1t_load_reg : std_logic;
  signal fu_DATA_LSU_in2_load_reg : std_logic;
  signal fu_DATA_LSU_opc_reg : std_logic_vector(2 downto 0);
  signal fu_alu_comp_in1t_load_reg : std_logic;
  signal fu_alu_comp_in2_load_reg : std_logic;
  signal fu_alu_comp_opc_reg : std_logic_vector(3 downto 0);
  signal fu_PARAM_LSU_in1t_load_reg : std_logic;
  signal fu_PARAM_LSU_in2_load_reg : std_logic;
  signal fu_PARAM_LSU_opc_reg : std_logic_vector(2 downto 0);
  signal fu_SP_LSU_in1t_load_reg : std_logic;
  signal fu_SP_LSU_in2_load_reg : std_logic;
  signal fu_SP_LSU_opc_reg : std_logic_vector(2 downto 0);
  signal fu_AQL_FU_t1_in_load_reg : std_logic;
  signal fu_AQL_FU_opc_reg : std_logic_vector(0 downto 0);
  signal fu_gcu_pc_load_reg : std_logic;
  signal fu_gcu_ra_load_reg : std_logic;
  signal fu_gcu_opc_reg : std_logic_vector(0 downto 0);

  -- RF control signals
  signal rf_RF_wr_load_reg : std_logic;
  signal rf_RF_wr_opc_reg : std_logic_vector(4 downto 0);
  signal rf_RF_rd_load_reg : std_logic;
  signal rf_RF_rd_opc_reg : std_logic_vector(4 downto 0);
  signal rf_bool_wr_load_reg : std_logic;
  signal rf_bool_wr_opc_reg : std_logic_vector(0 downto 0);
  signal rf_bool_rd_load_reg : std_logic;
  signal rf_bool_rd_opc_reg : std_logic_vector(0 downto 0);

  signal merged_glock_req : std_logic;
  signal pre_decode_merged_glock : std_logic;
  signal post_decode_merged_glock : std_logic;
  signal post_decode_merged_glock_r : std_logic;

  signal decode_fill_lock_reg : std_logic;
begin

  -- dismembering of instruction
  process (instructionword)
  begin --process
    move_B1 <= instructionword(24-1 downto 0);
    src_B1 <= instructionword(20 downto 7);
    dst_B1 <= instructionword(6 downto 0);
    grd_B1 <= instructionword(23 downto 21);
    move_B1_1 <= instructionword(42-1 downto 24);
    src_B1_1 <= instructionword(38 downto 31);
    dst_B1_1 <= instructionword(30 downto 24);
    grd_B1_1 <= instructionword(41 downto 39);

    limm_tag <= instructionword(42 downto 42);
  end process;

  -- map control registers to outputs
  fu_DATA_LSU_in1t_load <= fu_DATA_LSU_in1t_load_reg;
  fu_DATA_LSU_in2_load <= fu_DATA_LSU_in2_load_reg;
  fu_DATA_LSU_opc <= fu_DATA_LSU_opc_reg;

  fu_alu_comp_in1t_load <= fu_alu_comp_in1t_load_reg;
  fu_alu_comp_in2_load <= fu_alu_comp_in2_load_reg;
  fu_alu_comp_opc <= fu_alu_comp_opc_reg;

  fu_PARAM_LSU_in1t_load <= fu_PARAM_LSU_in1t_load_reg;
  fu_PARAM_LSU_in2_load <= fu_PARAM_LSU_in2_load_reg;
  fu_PARAM_LSU_opc <= fu_PARAM_LSU_opc_reg;

  fu_SP_LSU_in1t_load <= fu_SP_LSU_in1t_load_reg;
  fu_SP_LSU_in2_load <= fu_SP_LSU_in2_load_reg;
  fu_SP_LSU_opc <= fu_SP_LSU_opc_reg;

  fu_AQL_FU_t1_in_load <= fu_AQL_FU_t1_in_load_reg;
  fu_AQL_FU_opc <= fu_AQL_FU_opc_reg;

  ra_load <= fu_gcu_ra_load_reg;
  pc_load <= fu_gcu_pc_load_reg;
  pc_opcode <= fu_gcu_opc_reg;
  rf_RF_wr_load <= rf_RF_wr_load_reg;
  rf_RF_wr_opc <= rf_RF_wr_opc_reg;
  rf_RF_rd_load <= rf_RF_rd_load_reg;
  rf_RF_rd_opc <= rf_RF_rd_opc_reg;
  rf_bool_wr_load <= rf_bool_wr_load_reg;
  rf_bool_wr_opc <= rf_bool_wr_opc_reg;
  rf_bool_rd_load <= rf_bool_rd_load_reg;
  rf_bool_rd_opc <= rf_bool_rd_opc_reg;
  iu_IMM_P1_read_opc <= "0";
  iu_IMM_write_opc <= "0";
  socket_lsu_i1_bus_cntrl <= socket_lsu_i1_bus_cntrl_reg;
  socket_lsu_i2_bus_cntrl <= socket_lsu_i2_bus_cntrl_reg;
  socket_alu_comp_i1_bus_cntrl <= socket_alu_comp_i1_bus_cntrl_reg;
  socket_alu_comp_i2_bus_cntrl <= socket_alu_comp_i2_bus_cntrl_reg;
  socket_RF_i1_bus_cntrl <= socket_RF_i1_bus_cntrl_reg;
  socket_bool_i1_bus_cntrl <= socket_bool_i1_bus_cntrl_reg;
  socket_gcu_i1_bus_cntrl <= socket_gcu_i1_bus_cntrl_reg;
  socket_gcu_i2_bus_cntrl <= socket_gcu_i2_bus_cntrl_reg;
  socket_lsu_i1_1_bus_cntrl <= socket_lsu_i1_1_bus_cntrl_reg;
  socket_lsu_i2_1_bus_cntrl <= socket_lsu_i2_1_bus_cntrl_reg;
  socket_lsu_i2_1_1_bus_cntrl <= socket_lsu_i2_1_1_bus_cntrl_reg;
  socket_lsu_i1_1_1_bus_cntrl <= socket_lsu_i1_1_1_bus_cntrl_reg;
  socket_lsu_i2_1_1_2_1_bus_cntrl <= socket_lsu_i2_1_1_2_1_bus_cntrl_reg;
  B1_src_sel <= B1_src_sel_reg;
  B1_1_src_sel <= B1_1_src_sel_reg;
  simm_B1 <= simm_B1_reg;
  simm_B1_1 <= simm_B1_1_reg;

  -- generate signal squash_B1
  process (grd_B1, move_B1, rf_guard_bool_0, rf_guard_bool_1)
    variable sel : integer;
  begin --process
    -- squash by move NOP encoding
    if (conv_integer(unsigned(move_B1(23 downto 21))) = 5) then
      squash_B1 <= '1';
    else
      sel := conv_integer(unsigned(grd_B1));
      case sel is
        when 1 =>
          squash_B1 <= not rf_guard_bool_0;
        when 2 =>
          squash_B1 <= rf_guard_bool_0;
        when 3 =>
          squash_B1 <= not rf_guard_bool_1;
        when 4 =>
          squash_B1 <= rf_guard_bool_1;
        when others =>
          squash_B1 <= '0';
      end case;
    end if;
  end process;

  -- generate signal squash_B1_1
  process (grd_B1_1, move_B1_1, rf_guard_bool_0, rf_guard_bool_1)
    variable sel : integer;
  begin --process
    -- squash by move NOP encoding
    if (conv_integer(unsigned(move_B1_1(17 downto 15))) = 5) then
      squash_B1_1 <= '1';
    else
      sel := conv_integer(unsigned(grd_B1_1));
      case sel is
        when 1 =>
          squash_B1_1 <= not rf_guard_bool_0;
        when 2 =>
          squash_B1_1 <= rf_guard_bool_0;
        when 3 =>
          squash_B1_1 <= not rf_guard_bool_1;
        when 4 =>
          squash_B1_1 <= rf_guard_bool_1;
        when others =>
          squash_B1_1 <= '0';
      end case;
    end if;
  end process;


  --long immediate write process
  process (clk)
  begin --process
    if (clk'event and clk = '1') then
      if (rstx = '0') then
        iu_IMM_write_load <= '0';
        iu_IMM_write <= (others => '0');
      elsif pre_decode_merged_glock = '0' then
        if (conv_integer(unsigned(limm_tag)) = 0) then
          iu_IMM_write_load <= '0';
          iu_IMM_write(31 downto 0) <= tce_sxt("0", 32);
        else
          iu_IMM_write_load <= '0';
          iu_IMM_write(31 downto 0) <= tce_sxt("0", 32);
        end if;
      end if;
    end if;
  end process;


  -- main decoding process
  process (clk)
  begin
    if (clk'event and clk = '1') then
    if (rstx = '0') then
      socket_lsu_i1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_i1_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_i2_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_o1_bus_cntrl_reg <= (others => '0');
      socket_RF_i1_bus_cntrl_reg <= (others => '0');
      socket_RF_o1_bus_cntrl_reg <= (others => '0');
      socket_bool_i1_bus_cntrl_reg <= (others => '0');
      socket_bool_o1_bus_cntrl_reg <= (others => '0');
      socket_gcu_i1_bus_cntrl_reg <= (others => '0');
      socket_gcu_i2_bus_cntrl_reg <= (others => '0');
      socket_gcu_o1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_1_bus_cntrl_reg <= (others => '0');
      socket_RF_o1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i1_1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_1_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_o1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_1_1_2_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_1_1_1_bus_cntrl_reg <= (others => '0');
      simm_B1_reg <= (others => '0');
      B1_src_sel_reg <= (others => '0');
      simm_B1_1_reg <= (others => '0');
      B1_1_src_sel_reg <= (others => '0');
      fu_DATA_LSU_opc_reg <= (others => '0');
      fu_alu_comp_opc_reg <= (others => '0');
      fu_PARAM_LSU_opc_reg <= (others => '0');
      fu_SP_LSU_opc_reg <= (others => '0');
      fu_AQL_FU_opc_reg <= (others => '0');
      fu_gcu_opc_reg <= (others => '0');
      rf_RF_wr_opc_reg <= (others => '0');
      rf_RF_rd_opc_reg <= (others => '0');
      rf_bool_wr_opc_reg <= (others => '0');
      rf_bool_rd_opc_reg <= (others => '0');

      fu_DATA_LSU_in1t_load_reg <= '0';
      fu_DATA_LSU_in2_load_reg <= '0';
      fu_alu_comp_in1t_load_reg <= '0';
      fu_alu_comp_in2_load_reg <= '0';
      fu_PARAM_LSU_in1t_load_reg <= '0';
      fu_PARAM_LSU_in2_load_reg <= '0';
      fu_SP_LSU_in1t_load_reg <= '0';
      fu_SP_LSU_in2_load_reg <= '0';
      fu_AQL_FU_t1_in_load_reg <= '0';
      fu_gcu_pc_load_reg <= '0';
      fu_gcu_ra_load_reg <= '0';
      rf_RF_wr_load_reg <= '0';
      rf_RF_rd_load_reg <= '0';
      rf_bool_wr_load_reg <= '0';
      rf_bool_rd_load_reg <= '0';
      iu_IMM_P1_read_load <= '0';


    else
      if (db_tta_nreset = '0') then
      socket_lsu_i1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_i1_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_i2_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_o1_bus_cntrl_reg <= (others => '0');
      socket_RF_i1_bus_cntrl_reg <= (others => '0');
      socket_RF_o1_bus_cntrl_reg <= (others => '0');
      socket_bool_i1_bus_cntrl_reg <= (others => '0');
      socket_bool_o1_bus_cntrl_reg <= (others => '0');
      socket_gcu_i1_bus_cntrl_reg <= (others => '0');
      socket_gcu_i2_bus_cntrl_reg <= (others => '0');
      socket_gcu_o1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_1_bus_cntrl_reg <= (others => '0');
      socket_RF_o1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i1_1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_1_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_o1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_1_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_i2_1_1_2_1_bus_cntrl_reg <= (others => '0');
      socket_lsu_o1_1_1_1_1_bus_cntrl_reg <= (others => '0');
      simm_B1_reg <= (others => '0');
      B1_src_sel_reg <= (others => '0');
      simm_B1_1_reg <= (others => '0');
      B1_1_src_sel_reg <= (others => '0');
      fu_DATA_LSU_opc_reg <= (others => '0');
      fu_alu_comp_opc_reg <= (others => '0');
      fu_PARAM_LSU_opc_reg <= (others => '0');
      fu_SP_LSU_opc_reg <= (others => '0');
      fu_AQL_FU_opc_reg <= (others => '0');
      fu_gcu_opc_reg <= (others => '0');
      rf_RF_wr_opc_reg <= (others => '0');
      rf_RF_rd_opc_reg <= (others => '0');
      rf_bool_wr_opc_reg <= (others => '0');
      rf_bool_rd_opc_reg <= (others => '0');

      fu_DATA_LSU_in1t_load_reg <= '0';
      fu_DATA_LSU_in2_load_reg <= '0';
      fu_alu_comp_in1t_load_reg <= '0';
      fu_alu_comp_in2_load_reg <= '0';
      fu_PARAM_LSU_in1t_load_reg <= '0';
      fu_PARAM_LSU_in2_load_reg <= '0';
      fu_SP_LSU_in1t_load_reg <= '0';
      fu_SP_LSU_in2_load_reg <= '0';
      fu_AQL_FU_t1_in_load_reg <= '0';
      fu_gcu_pc_load_reg <= '0';
      fu_gcu_ra_load_reg <= '0';
      rf_RF_wr_load_reg <= '0';
      rf_RF_rd_load_reg <= '0';
      rf_bool_wr_load_reg <= '0';
      rf_bool_rd_load_reg <= '0';
      iu_IMM_P1_read_load <= '0';

      elsif (pre_decode_merged_glock = '0') then

        -- bus control signals for output mux
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 18) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(0, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 19) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(1, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 16) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(2, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 17) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(3, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 20) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(4, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 21) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(5, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 22) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(6, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 23) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(7, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 24) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(8, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 25) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(9, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 26) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(10, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 13))) = 0) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(11, B1_src_sel_reg'length));
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 13))) = 0) then
        simm_B1_reg <= tce_sxt(src_B1(12 downto 0), simm_B1_reg'length);
        end if;
        if (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 21) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(0, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 22) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(1, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 5))) = 4) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(2, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 20) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(3, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 23) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(4, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 24) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(5, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 25) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(6, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 26) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(7, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 27) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(8, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 28) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(9, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 29) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(10, B1_1_src_sel_reg'length));
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 7))) = 0) then
          B1_1_src_sel_reg <= std_logic_vector(conv_unsigned(11, B1_1_src_sel_reg'length));
        end if;
        if (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 7))) = 0) then
        simm_B1_1_reg <= tce_sxt(src_B1_1(6 downto 0), simm_B1_1_reg'length);
        end if;
        -- data control signals for output sockets connected to FUs
        -- control signals for RF read ports
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 16 and true) then
          rf_RF_rd_load_reg <= '1';
          rf_RF_rd_opc_reg <= tce_ext(src_B1(4 downto 0), rf_RF_rd_opc_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 5))) = 4 and true) then
          rf_RF_rd_load_reg <= '1';
          rf_RF_rd_opc_reg <= tce_ext(src_B1_1(4 downto 0), rf_RF_rd_opc_reg'length);
        else
          rf_RF_rd_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 17 and true) then
          rf_bool_rd_load_reg <= '1';
          rf_bool_rd_opc_reg <= tce_ext(src_B1(0 downto 0), rf_bool_rd_opc_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 20 and true) then
          rf_bool_rd_load_reg <= '1';
          rf_bool_rd_opc_reg <= tce_ext(src_B1_1(0 downto 0), rf_bool_rd_opc_reg'length);
        else
          rf_bool_rd_load_reg <= '0';
        end if;

        --control signals for IU read ports
        -- control signals for IU read ports
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(13 downto 9))) = 22) then
          iu_IMM_P1_read_load <= '1';
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(src_B1_1(7 downto 3))) = 25) then
          iu_IMM_P1_read_load <= '1';
        else
          iu_IMM_P1_read_load <= '0';
        end if;

        -- control signals for FU inputs
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 3))) = 6) then
          fu_DATA_LSU_in1t_load_reg <= '1';
          fu_DATA_LSU_opc_reg <= dst_B1(2 downto 0);
          socket_lsu_i1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_lsu_i1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 3))) = 6) then
          fu_DATA_LSU_in1t_load_reg <= '1';
          fu_DATA_LSU_opc_reg <= dst_B1_1(2 downto 0);
          socket_lsu_i1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_lsu_i1_bus_cntrl_reg'length);
        else
          fu_DATA_LSU_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 39) then
          fu_DATA_LSU_in2_load_reg <= '1';
          socket_lsu_i2_bus_cntrl_reg <= conv_std_logic_vector(0, socket_lsu_i2_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 39) then
          fu_DATA_LSU_in2_load_reg <= '1';
          socket_lsu_i2_bus_cntrl_reg <= conv_std_logic_vector(1, socket_lsu_i2_bus_cntrl_reg'length);
        else
          fu_DATA_LSU_in2_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 4))) = 2) then
          fu_alu_comp_in1t_load_reg <= '1';
          fu_alu_comp_opc_reg <= dst_B1(3 downto 0);
          socket_alu_comp_i1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_alu_comp_i1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 4))) = 2) then
          fu_alu_comp_in1t_load_reg <= '1';
          fu_alu_comp_opc_reg <= dst_B1_1(3 downto 0);
          socket_alu_comp_i1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_alu_comp_i1_bus_cntrl_reg'length);
        else
          fu_alu_comp_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 40) then
          fu_alu_comp_in2_load_reg <= '1';
          socket_alu_comp_i2_bus_cntrl_reg <= conv_std_logic_vector(0, socket_alu_comp_i2_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 40) then
          fu_alu_comp_in2_load_reg <= '1';
          socket_alu_comp_i2_bus_cntrl_reg <= conv_std_logic_vector(1, socket_alu_comp_i2_bus_cntrl_reg'length);
        else
          fu_alu_comp_in2_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 3))) = 7) then
          fu_PARAM_LSU_in1t_load_reg <= '1';
          fu_PARAM_LSU_opc_reg <= dst_B1(2 downto 0);
          socket_lsu_i1_1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_lsu_i1_1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 3))) = 7) then
          fu_PARAM_LSU_in1t_load_reg <= '1';
          fu_PARAM_LSU_opc_reg <= dst_B1_1(2 downto 0);
          socket_lsu_i1_1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_lsu_i1_1_bus_cntrl_reg'length);
        else
          fu_PARAM_LSU_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 42) then
          fu_PARAM_LSU_in2_load_reg <= '1';
          socket_lsu_i2_1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_lsu_i2_1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 42) then
          fu_PARAM_LSU_in2_load_reg <= '1';
          socket_lsu_i2_1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_lsu_i2_1_bus_cntrl_reg'length);
        else
          fu_PARAM_LSU_in2_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 3))) = 8) then
          fu_SP_LSU_in1t_load_reg <= '1';
          fu_SP_LSU_opc_reg <= dst_B1(2 downto 0);
          socket_lsu_i2_1_1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_lsu_i2_1_1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 3))) = 8) then
          fu_SP_LSU_in1t_load_reg <= '1';
          fu_SP_LSU_opc_reg <= dst_B1_1(2 downto 0);
          socket_lsu_i2_1_1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_lsu_i2_1_1_bus_cntrl_reg'length);
        else
          fu_SP_LSU_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 43) then
          fu_SP_LSU_in2_load_reg <= '1';
          socket_lsu_i1_1_1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_lsu_i1_1_1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 43) then
          fu_SP_LSU_in2_load_reg <= '1';
          socket_lsu_i1_1_1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_lsu_i1_1_1_bus_cntrl_reg'length);
        else
          fu_SP_LSU_in2_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 38) then
          fu_AQL_FU_t1_in_load_reg <= '1';
          fu_AQL_FU_opc_reg <= dst_B1(0 downto 0);
          socket_lsu_i2_1_1_2_1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_lsu_i2_1_1_2_1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 38) then
          fu_AQL_FU_t1_in_load_reg <= '1';
          fu_AQL_FU_opc_reg <= dst_B1_1(0 downto 0);
          socket_lsu_i2_1_1_2_1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_lsu_i2_1_1_2_1_bus_cntrl_reg'length);
        else
          fu_AQL_FU_t1_in_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 37) then
          fu_gcu_pc_load_reg <= '1';
          fu_gcu_opc_reg <= dst_B1(0 downto 0);
          socket_gcu_i1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_gcu_i1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 37) then
          fu_gcu_pc_load_reg <= '1';
          fu_gcu_opc_reg <= dst_B1_1(0 downto 0);
          socket_gcu_i1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_gcu_i1_bus_cntrl_reg'length);
        else
          fu_gcu_pc_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 41) then
          fu_gcu_ra_load_reg <= '1';
          socket_gcu_i2_bus_cntrl_reg <= conv_std_logic_vector(0, socket_gcu_i2_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 41) then
          fu_gcu_ra_load_reg <= '1';
          socket_gcu_i2_bus_cntrl_reg <= conv_std_logic_vector(1, socket_gcu_i2_bus_cntrl_reg'length);
        else
          fu_gcu_ra_load_reg <= '0';
        end if;
        -- control signals for RF inputs
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 5))) = 0 and true) then
          rf_RF_wr_load_reg <= '1';
          rf_RF_wr_opc_reg <= dst_B1(4 downto 0);
          socket_RF_i1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_RF_i1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 5))) = 0 and true) then
          rf_RF_wr_load_reg <= '1';
          rf_RF_wr_opc_reg <= dst_B1_1(4 downto 0);
          socket_RF_i1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_RF_i1_bus_cntrl_reg'length);
        else
          rf_RF_wr_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(6 downto 1))) = 36 and true) then
          rf_bool_wr_load_reg <= '1';
          rf_bool_wr_opc_reg <= dst_B1(0 downto 0);
          socket_bool_i1_bus_cntrl_reg <= conv_std_logic_vector(0, socket_bool_i1_bus_cntrl_reg'length);
        elsif (squash_B1_1 = '0' and conv_integer(unsigned(dst_B1_1(6 downto 1))) = 36 and true) then
          rf_bool_wr_load_reg <= '1';
          rf_bool_wr_opc_reg <= dst_B1_1(0 downto 0);
          socket_bool_i1_bus_cntrl_reg <= conv_std_logic_vector(1, socket_bool_i1_bus_cntrl_reg'length);
        else
          rf_bool_wr_load_reg <= '0';
        end if;
      end if;
      end if;
    end if;
  end process;

  lock_reg_proc : process (clk)
  begin
    if (clk'event and clk = '1') then
      if (rstx = '0') then
      -- Locked during active reset        post_decode_merged_glock_r <= '1';
      else
        post_decode_merged_glock_r <= post_decode_merged_glock;
      end if;
    end if;
  end process lock_reg_proc;

  lock_r <= merged_glock_req;
  merged_glock_req <= lock_req(0) or lock_req(1) or lock_req(2) or lock_req(3) or lock_req(4);
  pre_decode_merged_glock <= lock or merged_glock_req;
  post_decode_merged_glock <= pre_decode_merged_glock or decode_fill_lock_reg;
  locked <= post_decode_merged_glock_r;
  glock(0) <= post_decode_merged_glock; -- to DATA_LSU
  glock(1) <= post_decode_merged_glock; -- to alu_comp
  glock(2) <= post_decode_merged_glock; -- to PARAM_LSU
  glock(3) <= post_decode_merged_glock; -- to SP_LSU
  glock(4) <= post_decode_merged_glock; -- to AQL_FU
  glock(5) <= post_decode_merged_glock; -- to RF
  glock(6) <= post_decode_merged_glock; -- to bool
  glock(7) <= post_decode_merged_glock; -- to IMM
  glock(8) <= post_decode_merged_glock;

  decode_pipeline_fill_lock: process (clk)
  begin
    if clk'event and clk = '1' then
      if rstx = '0' then
        decode_fill_lock_reg <= '1';
      elsif lock = '0' then
        decode_fill_lock_reg <= '0';
      end if;
    end if;
  end process decode_pipeline_fill_lock;

end rtl_andor;
