library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.ext;
use IEEE.std_logic_arith.sxt;
use work.tta_core_globals.all;
use work.tce_util.all;

entity tta_core_interconn is

  port (
    clk : in std_logic;
    rstx : in std_logic;
    glock : in std_logic;
    socket_lsu_i1_data : out std_logic_vector(31 downto 0);
    socket_lsu_i2_data : out std_logic_vector(31 downto 0);
    socket_alu_comp_i1_data : out std_logic_vector(31 downto 0);
    socket_alu_comp_i2_data : out std_logic_vector(31 downto 0);
    socket_RF_i1_data : out std_logic_vector(31 downto 0);
    socket_bool_i1_data : out std_logic_vector(0 downto 0);
    socket_gcu_i1_data : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    socket_gcu_i2_data : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    socket_get_read_idx_low_inc_read_idx_i1_data : out std_logic_vector(31 downto 0);
    socket_lsu_1_i1_data : out std_logic_vector(31 downto 0);
    socket_lsu_1_i2_data : out std_logic_vector(31 downto 0);
    B1_mux_ctrl_in : in std_logic_vector(2 downto 0);
    B1_data_0_in : in std_logic_vector(31 downto 0);
    B1_data_1_in : in std_logic_vector(31 downto 0);
    B1_data_2_in : in std_logic_vector(31 downto 0);
    B1_data_3_in : in std_logic_vector(0 downto 0);
    B1_data_4_in : in std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    B1_data_5_in : in std_logic_vector(31 downto 0);
    B1_data_6_in : in std_logic_vector(31 downto 0);
    simm_B1 : in std_logic_vector(31 downto 0);
    simm_cntrl_B1 : in std_logic_vector(0 downto 0));

end tta_core_interconn;

architecture comb_andor of tta_core_interconn is

  signal databus_B1 : std_logic_vector(31 downto 0);

  component tta_core_input_mux_1 is
    generic (
      BUSW_0 : integer := 32;
      DATAW : integer := 32);
    port (
      databus0 : in std_logic_vector(BUSW_0-1 downto 0);
      data : out std_logic_vector(DATAW-1 downto 0));
  end component;

  component tta_core_input_mux_8 is
    generic (
      BUSW_0 : integer := 32;
      BUSW_1 : integer := 32;
      BUSW_2 : integer := 32;
      BUSW_3 : integer := 32;
      BUSW_4 : integer := 32;
      BUSW_5 : integer := 32;
      BUSW_6 : integer := 32;
      BUSW_7 : integer := 32;
      DATAW : integer := 32);
    port (
      databus0 : in std_logic_vector(BUSW_0-1 downto 0);
      databus1 : in std_logic_vector(BUSW_1-1 downto 0);
      databus2 : in std_logic_vector(BUSW_2-1 downto 0);
      databus3 : in std_logic_vector(BUSW_3-1 downto 0);
      databus4 : in std_logic_vector(BUSW_4-1 downto 0);
      databus5 : in std_logic_vector(BUSW_5-1 downto 0);
      databus6 : in std_logic_vector(BUSW_6-1 downto 0);
      databus7 : in std_logic_vector(BUSW_7-1 downto 0);
      data : out std_logic_vector(DATAW-1 downto 0);
      databus_cntrl : in std_logic_vector(2 downto 0));
  end component;


begin -- comb_andor

  RF_i1 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_RF_i1_data);

  alu_comp_i1 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_alu_comp_i1_data);

  alu_comp_i2 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_alu_comp_i2_data);

  bool_i1 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 1)
    port map (
      databus0 => databus_B1,
      data => socket_bool_i1_data);

  gcu_i1 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => IMEMADDRWIDTH)
    port map (
      databus0 => databus_B1,
      data => socket_gcu_i1_data);

  gcu_i2 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => IMEMADDRWIDTH)
    port map (
      databus0 => databus_B1,
      data => socket_gcu_i2_data);

  get_read_idx_low_inc_read_idx_i1 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_get_read_idx_low_inc_read_idx_i1_data);

  lsu_1_i1 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_lsu_1_i1_data);

  lsu_1_i2 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_lsu_1_i2_data);

  lsu_i1 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_lsu_i1_data);

  lsu_i2 : tta_core_input_mux_1
    generic map (
      BUSW_0 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      data => socket_lsu_i2_data);

  B1_bus_mux_inst : tta_core_input_mux_8
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      BUSW_2 => 32,
      BUSW_3 => 1,
      BUSW_4 => IMEMADDRWIDTH,
      BUSW_5 => 32,
      BUSW_6 => 32,
      BUSW_7 => 32,
      DATAW => 32)
    port map (
      databus0 => B1_data_0_in,
      databus1 => B1_data_1_in,
      databus2 => B1_data_2_in,
      databus3 => B1_data_3_in,
      databus4 => B1_data_4_in,
      databus5 => B1_data_5_in,
      databus6 => B1_data_6_in,
      databus7 => simm_B1,
      data => databus_B1,
      databus_cntrl => B1_mux_ctrl_in);


end comb_andor;
