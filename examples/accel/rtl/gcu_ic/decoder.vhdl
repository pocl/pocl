library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.tta_core_globals.all;
use work.tta_core_gcu_opcodes.all;
use work.tce_util.all;

entity tta_core_decoder is

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
    B1_src_sel : out std_logic_vector(2 downto 0);
    fu_buffer_in1t_load : out std_logic;
    fu_buffer_in2_load : out std_logic;
    fu_buffer_opc : out std_logic_vector(2 downto 0);
    fu_alu_in1t_load : out std_logic;
    fu_alu_in2_load : out std_logic;
    fu_alu_opc : out std_logic_vector(3 downto 0);
    fu_aql_in1t_load : out std_logic;
    fu_aql_opc : out std_logic_vector(0 downto 0);
    fu_cq_in1t_load : out std_logic;
    fu_cq_in2_load : out std_logic;
    fu_cq_opc : out std_logic_vector(2 downto 0);
    rf_RF_wr_load : out std_logic;
    rf_RF_wr_opc : out std_logic_vector(3 downto 0);
    rf_RF_rd_load : out std_logic;
    rf_RF_rd_opc : out std_logic_vector(3 downto 0);
    rf_bool_wr_load : out std_logic;
    rf_bool_wr_opc : out std_logic_vector(0 downto 0);
    rf_bool_rd_load : out std_logic;
    rf_bool_rd_opc : out std_logic_vector(0 downto 0);
    rf_guard_bool_0 : in std_logic;
    rf_guard_bool_1 : in std_logic;
    lock_req : in std_logic_vector(3 downto 0);
    glock : out std_logic_vector(6 downto 0);
    db_tta_nreset : in std_logic);

end tta_core_decoder;

architecture rtl_andor of tta_core_decoder is

  -- signals for source, destination and guard fields
  signal move_B1 : std_logic_vector(41 downto 0);
  signal src_B1 : std_logic_vector(32 downto 0);
  signal dst_B1 : std_logic_vector(5 downto 0);
  signal grd_B1 : std_logic_vector(2 downto 0);

  -- signals for dedicated immediate slots


  -- squash signals
  signal squash_B1 : std_logic;

  -- socket control signals
  signal socket_lsu_o1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_alu_comp_o1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_RF_o1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_bool_o1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_gcu_o1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_get_read_idx_low_inc_read_idx_o1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal socket_lsu_1_o1_bus_cntrl_reg : std_logic_vector(0 downto 0);
  signal simm_B1_reg : std_logic_vector(31 downto 0);
  signal B1_src_sel_reg : std_logic_vector(2 downto 0);

  -- FU control signals
  signal fu_buffer_in1t_load_reg : std_logic;
  signal fu_buffer_in2_load_reg : std_logic;
  signal fu_buffer_opc_reg : std_logic_vector(2 downto 0);
  signal fu_alu_in1t_load_reg : std_logic;
  signal fu_alu_in2_load_reg : std_logic;
  signal fu_alu_opc_reg : std_logic_vector(3 downto 0);
  signal fu_aql_in1t_load_reg : std_logic;
  signal fu_aql_opc_reg : std_logic_vector(0 downto 0);
  signal fu_cq_in1t_load_reg : std_logic;
  signal fu_cq_in2_load_reg : std_logic;
  signal fu_cq_opc_reg : std_logic_vector(2 downto 0);
  signal fu_gcu_pc_load_reg : std_logic;
  signal fu_gcu_ra_load_reg : std_logic;
  signal fu_gcu_opc_reg : std_logic_vector(0 downto 0);

  -- RF control signals
  signal rf_RF_wr_load_reg : std_logic;
  signal rf_RF_wr_opc_reg : std_logic_vector(3 downto 0);
  signal rf_RF_rd_load_reg : std_logic;
  signal rf_RF_rd_opc_reg : std_logic_vector(3 downto 0);
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
    move_B1 <= instructionword(42-1 downto 0);
    src_B1 <= instructionword(38 downto 6);
    dst_B1 <= instructionword(5 downto 0);
    grd_B1 <= instructionword(41 downto 39);

  end process;

  -- map control registers to outputs
  fu_buffer_in1t_load <= fu_buffer_in1t_load_reg;
  fu_buffer_in2_load <= fu_buffer_in2_load_reg;
  fu_buffer_opc <= fu_buffer_opc_reg;

  fu_alu_in1t_load <= fu_alu_in1t_load_reg;
  fu_alu_in2_load <= fu_alu_in2_load_reg;
  fu_alu_opc <= fu_alu_opc_reg;

  fu_aql_in1t_load <= fu_aql_in1t_load_reg;
  fu_aql_opc <= fu_aql_opc_reg;

  fu_cq_in1t_load <= fu_cq_in1t_load_reg;
  fu_cq_in2_load <= fu_cq_in2_load_reg;
  fu_cq_opc <= fu_cq_opc_reg;

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
  B1_src_sel <= B1_src_sel_reg;
  simm_B1 <= simm_B1_reg;

  -- generate signal squash_B1
  process (grd_B1, move_B1, rf_guard_bool_0, rf_guard_bool_1)
    variable sel : integer;
  begin --process
    -- squash by move NOP encoding
    if (conv_integer(unsigned(move_B1(41 downto 39))) = 5) then
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




  -- main decoding process
  process (clk, rstx)
  begin
    if (rstx = '0') then
      socket_lsu_o1_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_o1_bus_cntrl_reg <= (others => '0');
      socket_RF_o1_bus_cntrl_reg <= (others => '0');
      socket_bool_o1_bus_cntrl_reg <= (others => '0');
      socket_gcu_o1_bus_cntrl_reg <= (others => '0');
      socket_get_read_idx_low_inc_read_idx_o1_bus_cntrl_reg <= (others => '0');
      socket_lsu_1_o1_bus_cntrl_reg <= (others => '0');
      simm_B1_reg <= (others => '0');
      B1_src_sel_reg <= (others => '0');
      fu_buffer_opc_reg <= (others => '0');
      fu_alu_opc_reg <= (others => '0');
      fu_aql_opc_reg <= (others => '0');
      fu_cq_opc_reg <= (others => '0');
      fu_gcu_opc_reg <= (others => '0');
      rf_RF_wr_opc_reg <= (others => '0');
      rf_RF_rd_opc_reg <= (others => '0');
      rf_bool_wr_opc_reg <= (others => '0');
      rf_bool_rd_opc_reg <= (others => '0');

      fu_buffer_in1t_load_reg <= '0';
      fu_buffer_in2_load_reg <= '0';
      fu_alu_in1t_load_reg <= '0';
      fu_alu_in2_load_reg <= '0';
      fu_aql_in1t_load_reg <= '0';
      fu_cq_in1t_load_reg <= '0';
      fu_cq_in2_load_reg <= '0';
      fu_gcu_pc_load_reg <= '0';
      fu_gcu_ra_load_reg <= '0';
      rf_RF_wr_load_reg <= '0';
      rf_RF_rd_load_reg <= '0';
      rf_bool_wr_load_reg <= '0';
      rf_bool_rd_load_reg <= '0';


    elsif (clk'event and clk = '1') then -- rising clock edge
      if (db_tta_nreset = '0') then
      socket_lsu_o1_bus_cntrl_reg <= (others => '0');
      socket_alu_comp_o1_bus_cntrl_reg <= (others => '0');
      socket_RF_o1_bus_cntrl_reg <= (others => '0');
      socket_bool_o1_bus_cntrl_reg <= (others => '0');
      socket_gcu_o1_bus_cntrl_reg <= (others => '0');
      socket_get_read_idx_low_inc_read_idx_o1_bus_cntrl_reg <= (others => '0');
      socket_lsu_1_o1_bus_cntrl_reg <= (others => '0');
      simm_B1_reg <= (others => '0');
      B1_src_sel_reg <= (others => '0');
      fu_buffer_opc_reg <= (others => '0');
      fu_alu_opc_reg <= (others => '0');
      fu_aql_opc_reg <= (others => '0');
      fu_cq_opc_reg <= (others => '0');
      fu_gcu_opc_reg <= (others => '0');
      rf_RF_wr_opc_reg <= (others => '0');
      rf_RF_rd_opc_reg <= (others => '0');
      rf_bool_wr_opc_reg <= (others => '0');
      rf_bool_rd_opc_reg <= (others => '0');

      fu_buffer_in1t_load_reg <= '0';
      fu_buffer_in2_load_reg <= '0';
      fu_alu_in1t_load_reg <= '0';
      fu_alu_in2_load_reg <= '0';
      fu_aql_in1t_load_reg <= '0';
      fu_cq_in1t_load_reg <= '0';
      fu_cq_in2_load_reg <= '0';
      fu_gcu_pc_load_reg <= '0';
      fu_gcu_ra_load_reg <= '0';
      rf_RF_wr_load_reg <= '0';
      rf_RF_rd_load_reg <= '0';
      rf_bool_wr_load_reg <= '0';
      rf_bool_rd_load_reg <= '0';

      elsif (pre_decode_merged_glock = '0') then

        -- bus control signals for output mux
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 10) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(0, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 11) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(1, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 8) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(2, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 9) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(3, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 12) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(4, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 13) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(5, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 14) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(6, B1_src_sel_reg'length));
        elsif (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 32))) = 0) then
          B1_src_sel_reg <= std_logic_vector(conv_unsigned(7, B1_src_sel_reg'length));
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 32))) = 0) then
        simm_B1_reg <= tce_ext(src_B1(31 downto 0), simm_B1_reg'length);
        end if;
        -- data control signals for output sockets connected to FUs
        -- control signals for RF read ports
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 8 and true) then
          rf_RF_rd_load_reg <= '1';
          rf_RF_rd_opc_reg <= tce_ext(src_B1(3 downto 0), rf_RF_rd_opc_reg'length);
        else
          rf_RF_rd_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(src_B1(32 downto 29))) = 9 and true) then
          rf_bool_rd_load_reg <= '1';
          rf_bool_rd_opc_reg <= tce_ext(src_B1(0 downto 0), rf_bool_rd_opc_reg'length);
        else
          rf_bool_rd_load_reg <= '0';
        end if;

        --control signals for IU read ports
        -- control signals for IU read ports

        -- control signals for FU inputs
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 3))) = 4) then
          fu_buffer_in1t_load_reg <= '1';
          fu_buffer_opc_reg <= dst_B1(2 downto 0);
        else
          fu_buffer_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 1))) = 27) then
          fu_buffer_in2_load_reg <= '1';
        else
          fu_buffer_in2_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 4))) = 0) then
          fu_alu_in1t_load_reg <= '1';
          fu_alu_opc_reg <= dst_B1(3 downto 0);
        else
          fu_alu_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 1))) = 28) then
          fu_alu_in2_load_reg <= '1';
        else
          fu_alu_in2_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 1))) = 26) then
          fu_aql_in1t_load_reg <= '1';
          fu_aql_opc_reg <= dst_B1(0 downto 0);
        else
          fu_aql_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 3))) = 5) then
          fu_cq_in1t_load_reg <= '1';
          fu_cq_opc_reg <= dst_B1(2 downto 0);
        else
          fu_cq_in1t_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 1))) = 30) then
          fu_cq_in2_load_reg <= '1';
        else
          fu_cq_in2_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 1))) = 25) then
          fu_gcu_pc_load_reg <= '1';
          fu_gcu_opc_reg <= dst_B1(0 downto 0);
        else
          fu_gcu_pc_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 1))) = 29) then
          fu_gcu_ra_load_reg <= '1';
        else
          fu_gcu_ra_load_reg <= '0';
        end if;
        -- control signals for RF inputs
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 4))) = 1 and true) then
          rf_RF_wr_load_reg <= '1';
          rf_RF_wr_opc_reg <= dst_B1(3 downto 0);
        else
          rf_RF_wr_load_reg <= '0';
        end if;
        if (squash_B1 = '0' and conv_integer(unsigned(dst_B1(5 downto 1))) = 24 and true) then
          rf_bool_wr_load_reg <= '1';
          rf_bool_wr_opc_reg <= dst_B1(0 downto 0);
        else
          rf_bool_wr_load_reg <= '0';
        end if;
      end if;
    end if;
  end process;

  lock_reg_proc : process (clk, rstx)
  begin
    if (rstx = '0') then
      -- Locked during active reset      post_decode_merged_glock_r <= '1';
    elsif (clk'event and clk = '1') then
      post_decode_merged_glock_r <= post_decode_merged_glock;
    end if;
  end process lock_reg_proc;

  lock_r <= merged_glock_req;
  merged_glock_req <= lock_req(0) or lock_req(1) or lock_req(2) or lock_req(3);
  pre_decode_merged_glock <= lock or merged_glock_req;
  post_decode_merged_glock <= pre_decode_merged_glock or decode_fill_lock_reg;
  locked <= post_decode_merged_glock_r;
  glock(0) <= post_decode_merged_glock; -- to buffer
  glock(1) <= post_decode_merged_glock; -- to alu
  glock(2) <= post_decode_merged_glock; -- to aql
  glock(3) <= post_decode_merged_glock; -- to cq
  glock(4) <= post_decode_merged_glock; -- to RF
  glock(5) <= post_decode_merged_glock; -- to bool
  glock(6) <= post_decode_merged_glock;

  decode_pipeline_fill_lock: process (clk, rstx)
  begin
    if rstx = '0' then
      decode_fill_lock_reg <= '1';
    elsif clk'event and clk = '1' then
      if lock = '0' then
        decode_fill_lock_reg <= '0';
      end if;
    end if;
  end process decode_pipeline_fill_lock;

end rtl_andor;
