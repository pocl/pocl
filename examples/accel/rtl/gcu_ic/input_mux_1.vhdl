library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.tce_util.all;

entity tta_core_input_mux_1 is

  generic (
    BUSW_0 : integer := 32;
    DATAW : integer := 32);
  port (
    databus0 : in std_logic_vector(BUSW_0-1 downto 0);
    data : out std_logic_vector(DATAW-1 downto 0));

end tta_core_input_mux_1;

architecture rtl of tta_core_input_mux_1 is
begin

    -- If width of input bus is greater than width of output,
    -- using the LSB bits.
    -- If width of input bus is smaller than width of output,
    -- using zero extension to generate extra bits.

  process (databus0)
  begin
    data <= (others => '0');
    data <= tce_ext(databus0, data'length);
  end process;
end rtl;
