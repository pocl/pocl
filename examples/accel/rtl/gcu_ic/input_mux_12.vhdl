library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.tce_util.all;

entity ffaccel_input_mux_12 is

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

end ffaccel_input_mux_12;

architecture rtl of ffaccel_input_mux_12 is
begin

    -- If width of input bus is greater than width of output,
    -- using the LSB bits.
    -- If width of input bus is smaller than width of output,
    -- using zero extension to generate extra bits.

  sel : process (databus_cntrl, databus0, databus1, databus2, databus3, databus4, databus5, databus6, databus7, databus8, databus9, databus10, databus11)
  begin
    data <= (others => '0');
    case databus_cntrl is
      when "0000" =>
        data <= tce_ext(databus0, data'length);
      when "0001" =>
        data <= tce_ext(databus1, data'length);
      when "0010" =>
        data <= tce_ext(databus2, data'length);
      when "0011" =>
        data <= tce_ext(databus3, data'length);
      when "0100" =>
        data <= tce_ext(databus4, data'length);
      when "0101" =>
        data <= tce_ext(databus5, data'length);
      when "0110" =>
        data <= tce_ext(databus6, data'length);
      when "0111" =>
        data <= tce_ext(databus7, data'length);
      when "1000" =>
        data <= tce_ext(databus8, data'length);
      when "1001" =>
        data <= tce_ext(databus9, data'length);
      when "1010" =>
        data <= tce_ext(databus10, data'length);
      when others =>
        data <= tce_ext(databus11, data'length);
    end case;
  end process sel;
end rtl;
