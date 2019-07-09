library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.ext;
use IEEE.std_logic_arith.sxt;
use work.ffaccel_globals.all;
use work.tce_util.all;

entity ffaccel_interconn is

  port (
    clk : in std_logic;
    rstx : in std_logic;
    glock : in std_logic;
    socket_lsu_i1_data : out std_logic_vector(11 downto 0);
    socket_lsu_i1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_lsu_i2_data : out std_logic_vector(31 downto 0);
    socket_lsu_i2_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_alu_comp_i1_data : out std_logic_vector(31 downto 0);
    socket_alu_comp_i1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_alu_comp_i2_data : out std_logic_vector(31 downto 0);
    socket_alu_comp_i2_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_RF_i1_data : out std_logic_vector(31 downto 0);
    socket_RF_i1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_bool_i1_data : out std_logic_vector(0 downto 0);
    socket_bool_i1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_gcu_i1_data : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    socket_gcu_i1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_gcu_i2_data : out std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    socket_gcu_i2_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_lsu_i1_1_data : out std_logic_vector(31 downto 0);
    socket_lsu_i1_1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_lsu_i2_1_data : out std_logic_vector(31 downto 0);
    socket_lsu_i2_1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_lsu_i2_1_1_data : out std_logic_vector(9 downto 0);
    socket_lsu_i2_1_1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_lsu_i1_1_1_data : out std_logic_vector(31 downto 0);
    socket_lsu_i1_1_1_bus_cntrl : in std_logic_vector(0 downto 0);
    socket_lsu_i2_1_1_2_1_data : out std_logic_vector(31 downto 0);
    socket_lsu_i2_1_1_2_1_bus_cntrl : in std_logic_vector(0 downto 0);
    B1_mux_ctrl_in : in std_logic_vector(3 downto 0);
    B1_data_0_in : in std_logic_vector(31 downto 0);
    B1_data_1_in : in std_logic_vector(31 downto 0);
    B1_data_2_in : in std_logic_vector(31 downto 0);
    B1_data_3_in : in std_logic_vector(0 downto 0);
    B1_data_4_in : in std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    B1_data_5_in : in std_logic_vector(31 downto 0);
    B1_data_6_in : in std_logic_vector(31 downto 0);
    B1_data_7_in : in std_logic_vector(31 downto 0);
    B1_data_8_in : in std_logic_vector(31 downto 0);
    B1_data_9_in : in std_logic_vector(31 downto 0);
    B1_data_10_in : in std_logic_vector(31 downto 0);
    B1_1_mux_ctrl_in : in std_logic_vector(3 downto 0);
    B1_1_data_0_in : in std_logic_vector(31 downto 0);
    B1_1_data_1_in : in std_logic_vector(31 downto 0);
    B1_1_data_2_in : in std_logic_vector(31 downto 0);
    B1_1_data_3_in : in std_logic_vector(0 downto 0);
    B1_1_data_4_in : in std_logic_vector(IMEMADDRWIDTH-1 downto 0);
    B1_1_data_5_in : in std_logic_vector(31 downto 0);
    B1_1_data_6_in : in std_logic_vector(31 downto 0);
    B1_1_data_7_in : in std_logic_vector(31 downto 0);
    B1_1_data_8_in : in std_logic_vector(31 downto 0);
    B1_1_data_9_in : in std_logic_vector(31 downto 0);
    B1_1_data_10_in : in std_logic_vector(31 downto 0);
    simm_B1 : in std_logic_vector(31 downto 0);
    simm_cntrl_B1 : in std_logic_vector(0 downto 0);
    simm_B1_1 : in std_logic_vector(31 downto 0);
    simm_cntrl_B1_1 : in std_logic_vector(0 downto 0));

end ffaccel_interconn;

architecture comb_andor of ffaccel_interconn is

  signal databus_B1 : std_logic_vector(31 downto 0);
  signal databus_B1_1 : std_logic_vector(31 downto 0);

  component ffaccel_input_mux_2
    generic (
      BUSW_0 : integer := 32;
      BUSW_1 : integer := 32;
      DATAW : integer := 32);
    port (
      databus0 : in std_logic_vector(BUSW_0-1 downto 0);
      databus1 : in std_logic_vector(BUSW_1-1 downto 0);
      data : out std_logic_vector(DATAW-1 downto 0);
      databus_cntrl : in std_logic_vector(0 downto 0));
  end component;

  component ffaccel_input_mux_12
    generic (
      BUSW_0 : integer := 32;
      BUSW_1 : integer := 32;
      BUSW_2 : integer := 32;
      BUSW_3 : integer := 32;
      BUSW_4 : integer := 32;
      BUSW_5 : integer := 32;
      BUSW_6 : integer := 32;
      BUSW_7 : integer := 32;
      BUSW_8 : integer := 32;
      BUSW_9 : integer := 32;
      BUSW_10 : integer := 32;
      BUSW_11 : integer := 32;
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
      databus8 : in std_logic_vector(BUSW_8-1 downto 0);
      databus9 : in std_logic_vector(BUSW_9-1 downto 0);
      databus10 : in std_logic_vector(BUSW_10-1 downto 0);
      databus11 : in std_logic_vector(BUSW_11-1 downto 0);
      data : out std_logic_vector(DATAW-1 downto 0);
      databus_cntrl : in std_logic_vector(3 downto 0));
  end component;


begin -- comb_andor

  RF_i1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_RF_i1_data,
      databus_cntrl => socket_RF_i1_bus_cntrl);

  alu_comp_i1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_alu_comp_i1_data,
      databus_cntrl => socket_alu_comp_i1_bus_cntrl);

  alu_comp_i2 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_alu_comp_i2_data,
      databus_cntrl => socket_alu_comp_i2_bus_cntrl);

  bool_i1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 1)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_bool_i1_data,
      databus_cntrl => socket_bool_i1_bus_cntrl);

  gcu_i1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => IMEMADDRWIDTH)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_gcu_i1_data,
      databus_cntrl => socket_gcu_i1_bus_cntrl);

  gcu_i2 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => IMEMADDRWIDTH)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_gcu_i2_data,
      databus_cntrl => socket_gcu_i2_bus_cntrl);

  lsu_i1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 12)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_lsu_i1_data,
      databus_cntrl => socket_lsu_i1_bus_cntrl);

  lsu_i1_1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_lsu_i1_1_data,
      databus_cntrl => socket_lsu_i1_1_bus_cntrl);

  lsu_i1_1_1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_lsu_i1_1_1_data,
      databus_cntrl => socket_lsu_i1_1_1_bus_cntrl);

  lsu_i2 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_lsu_i2_data,
      databus_cntrl => socket_lsu_i2_bus_cntrl);

  lsu_i2_1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_lsu_i2_1_data,
      databus_cntrl => socket_lsu_i2_1_bus_cntrl);

  lsu_i2_1_1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 10)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_lsu_i2_1_1_data,
      databus_cntrl => socket_lsu_i2_1_1_bus_cntrl);

  lsu_i2_1_1_2_1 : ffaccel_input_mux_2
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      DATAW => 32)
    port map (
      databus0 => databus_B1,
      databus1 => databus_B1_1,
      data => socket_lsu_i2_1_1_2_1_data,
      databus_cntrl => socket_lsu_i2_1_1_2_1_bus_cntrl);

  B1_bus_mux_inst : ffaccel_input_mux_12
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      BUSW_2 => 32,
      BUSW_3 => 1,
      BUSW_4 => IMEMADDRWIDTH,
      BUSW_5 => 32,
      BUSW_6 => 32,
      BUSW_7 => 32,
      BUSW_8 => 32,
      BUSW_9 => 32,
      BUSW_10 => 32,
      BUSW_11 => 32,
      DATAW => 32)
    port map (
      databus0 => B1_data_0_in,
      databus1 => B1_data_1_in,
      databus2 => B1_data_2_in,
      databus3 => B1_data_3_in,
      databus4 => B1_data_4_in,
      databus5 => B1_data_5_in,
      databus6 => B1_data_6_in,
      databus7 => B1_data_7_in,
      databus8 => B1_data_8_in,
      databus9 => B1_data_9_in,
      databus10 => B1_data_10_in,
      databus11 => simm_B1,
      data => databus_B1,
      databus_cntrl => B1_mux_ctrl_in);

  B1_1_bus_mux_inst : ffaccel_input_mux_12
    generic map (
      BUSW_0 => 32,
      BUSW_1 => 32,
      BUSW_2 => 32,
      BUSW_3 => 1,
      BUSW_4 => IMEMADDRWIDTH,
      BUSW_5 => 32,
      BUSW_6 => 32,
      BUSW_7 => 32,
      BUSW_8 => 32,
      BUSW_9 => 32,
      BUSW_10 => 32,
      BUSW_11 => 32,
      DATAW => 32)
    port map (
      databus0 => B1_1_data_0_in,
      databus1 => B1_1_data_1_in,
      databus2 => B1_1_data_2_in,
      databus3 => B1_1_data_3_in,
      databus4 => B1_1_data_4_in,
      databus5 => B1_1_data_5_in,
      databus6 => B1_1_data_6_in,
      databus7 => B1_1_data_7_in,
      databus8 => B1_1_data_8_in,
      databus9 => B1_1_data_9_in,
      databus10 => B1_1_data_10_in,
      databus11 => simm_B1_1,
      data => databus_B1_1,
      databus_cntrl => B1_1_mux_ctrl_in);


end comb_andor;
