library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.tce_util.all;

entity ffaccel_input_mux_2 is

  generic (
    BUSW_0 : integer := 32;
    BUSW_1 : integer := 32;
    DATAW : integer := 32);
  port (
    databus0 : in std_logic_vector(BUSW_0-1 downto 0);
    databus1 : in std_logic_vector(BUSW_1-1 downto 0);
    data : out std_logic_vector(DATAW-1 downto 0);
    databus_cntrl : in std_logic_vector(0 downto 0));

end ffaccel_input_mux_2;

architecture rtl of ffaccel_input_mux_2 is
begin

    -- If width of input bus is greater than width of output,
    -- using the LSB bits.
    -- If width of input bus is smaller than width of output,
    -- using zero extension to generate extra bits.

  sel : process (databus_cntrl, databus0, databus1)
  begin
    data <= (others => '0');
    case databus_cntrl is
      when "0" =>
        data <= tce_ext(databus0, data'length);
      when others =>
        data <= tce_ext(databus1, data'length);
    end case;
  end process sel;
end rtl;
