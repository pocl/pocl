-- Copyright (c) 2002-2009 Tampere University.
--
-- This file is part of TTA-Based Codesign Environment (TCE).
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a
-- copy of this software and associated documentation files (the "Software"),
-- to deal in the Software without restriction, including without limitation
-- the rights to use, copy, modify, merge, publish, distribute, sublicense,
-- and/or sell copies of the Software, and to permit persons to whom the
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
-- DEALINGS IN THE SOFTWARE.



library ieee;
use ieee.std_logic_1164.all;
package util is
  
  function flip_bits(in_vec : std_logic_vector)  -- make unconstrained
    return std_logic_vector;

  function int_to_str (InputInt : integer)
    return string;

  function bit_width (num : integer)
    return integer;

  component util_inverter is  
  port (
    data_in  : in  std_logic;
    data_out : out std_logic);
  end component;

end package util;

package body util is
  
  function flip_bits(in_vec : std_logic_vector)  -- make unconstrained
    return std_logic_vector is

    variable flipped_vec : std_logic_vector(in_vec'reverse_range);
  begin
    for i in in_vec'range loop
      flipped_vec(i) := in_vec(i);
    end loop;
    return flipped_vec;
  end flip_bits;

  -- ------------------------------------------------------------------------
  -- PROCEDURE NAME:  Int_To_Str
  --
  -- PARAMETERS    :  InputInt     - Integer to be converted to String.
  --                  ResultStr    - String buffer for converted Integer
  --                  AppendPos    - Position in buffer to place result
  --
  -- DESCRIPTION   :  This procedure is used to convert an input integer
  --                  into a string representation.  The converted string
  --                  may be placed at a specific position in the result
  --                  buffer.
  --
  -- ------------------------------------------------------------------------

  function int_to_str (InputInt : integer)
    return string is

    -- Look-up table.  Given an int, we can get the character.
    type     integer_table_type is array (0 to 9) of character;
    constant integer_table : integer_table_type :=
      (
        '0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9'
        ) ;

    --  Local variables used in this function.
    
    variable inpVal      : integer := inputInt;
    variable divisor     : integer := 10;
    variable tmpStrIndex : integer := 1;
    variable tmpStr      : string (1 to 256);
    variable ResultStr   : string (1 to 256);
    variable appendPos   : integer := 1;
    
  begin
    
    if (inpVal = 0) then
      tmpStr(tmpStrIndex) := integer_table (0);
      tmpStrIndex         := tmpStrIndex + 1;
    else
      while (inpVal > 0) loop
        tmpStr(tmpStrIndex) := integer_table (inpVal mod divisor);
        tmpStrIndex         := tmpStrIndex + 1;
        inpVal              := inpVal / divisor;
      end loop;
    end if;

    if (appendPos /= 1) then
      resultStr(appendPos) := ',';
      appendPos            := appendPos + 1;
    end if;

    for i in tmpStrIndex-1 downto 1 loop
      resultStr(appendPos) := tmpStr(i);
      appendPos            := appendPos + 1;
    end loop;

    return ResultStr;
    
  end int_to_str;

  function bit_width (num : integer) return integer is
    variable count : integer;
  begin
    count      := 1;
    if (num    <= 0) then return 0;
    elsif (num <= 2**10) then
      for i in 1 to 10 loop
        if (2**count >= num) then
          return i;
        end if;
        count := count + 1;
      end loop;
    elsif (num <= 2**20) then
      for i in 1 to 20 loop
        if (2**count >= num) then
          return i;
        end if;
        count := count + 1;
      end loop;
    elsif (num <= 2**30) then
      for i in 1 to 30 loop
        if (2**count >= num) then
          return i;
        end if;
        count := count + 1;
      end loop;
    else
      for i in 1 to num loop
        if (2**i >= num) then
          return i;
        end if;
      end loop;
    end if;
  end bit_width;

end package body util;

library ieee;
use ieee.std_logic_1164.all;

entity util_inverter is  
  port (
    data_in  : in  std_logic;
    data_out : out std_logic);
end util_inverter;

architecture rtl of util_inverter is
begin  -- rtl
  data_out <= not data_in;
end rtl;
