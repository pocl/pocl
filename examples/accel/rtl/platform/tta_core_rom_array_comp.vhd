library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use work.tta_core_imem_image.all;

entity tta_core_rom_array_comp is

  generic (
    addrw  : integer := 10;
    instrw : integer := 100);
  port (
    clock   : in  std_logic;
    en_x    : in std_logic; -- not used
    addr    : in  std_logic_vector(addrw-1 downto 0);
    dataout : out std_logic_vector(instrw-1 downto 0));
end tta_core_rom_array_comp;

architecture rtl of tta_core_rom_array_comp is

  subtype imem_index is integer range 0 to imem_array'length-1;
  constant imem : std_logic_imem_matrix(0 to imem_array'length-1) := imem_array;
  signal en_x_dummy : std_logic;

begin --rtl

  process
    variable imem_line : imem_index;
  begin -- process
    wait until clock'event and clock='1';
    imem_line := conv_integer(unsigned(addr));
    dataout <= imem(imem_line);
  end process;

  en_x_dummy <= en_x; -- dummy connection

end rtl;
