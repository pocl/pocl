-- Copyright (c) 2002-2015 Tampere University.
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
use ieee.numeric_std.to_integer;
use ieee.numeric_std.unsigned;

package tce_util is

  function flip_bits(in_vec : std_logic_vector)  -- make unconstrained
    return std_logic_vector;

  function int_to_str (InputInt : integer)
    return string;

  function bit_width (num : integer)
    return integer;

  function tce_ext (
    constant src    : std_logic_vector;
    constant dstlen : integer)
    return std_logic_vector;

  function tce_sxt (
    constant src    : std_logic_vector;
    constant dstlen : integer)
    return std_logic_vector;

  function to_int (
    constant val : boolean)
    return integer;

  function to_uint (
    constant interpret_as_unsigned : std_logic_vector)
    return natural;

  type integer_array is array (natural range <>) of integer;
  function return_highest(values : integer_array; numberOfValues : integer)
    return integer;

  component util_inverter is
    port (
      data_in  : in  std_logic;
      data_out : out std_logic);
  end component;

end package tce_util;

package body tce_util is

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

  -- Same as Synopsys' EXT function but set all bits to 'X'es if some bits
  -- in the src vector are 'U', 'X', 'W' or 'Z'.
  function tce_ext (
    constant src    : std_logic_vector;
    constant dstlen : integer)
    return std_logic_vector is
    -- Normalize src's slice i.e. 4 downto 2 -> 2 downto 0.
    variable tmp_src : std_logic_vector(src'length-1 downto 0) :=
      (others => '0');
    variable dst     : std_logic_vector(dstlen-1 downto 0) :=
      (others => '0');
    variable common_msb : integer := 0;
  begin
    tmp_src := src;
    if tmp_src'high < dst'high then
      common_msb := tmp_src'high;
    else
      common_msb := dst'high;
    end if;
    dst(common_msb downto 0) := tmp_src(common_msb downto 0);
    return dst;
  end tce_ext;

  -- Same as Synopsys' SXT function but set all bits to 'X'es if some bits
  -- in the src vector are 'U', 'X', 'W' or 'Z'.
  function tce_sxt (
    constant src    : std_logic_vector;
    constant dstlen : integer)
    return std_logic_vector is
    -- Normalize src's slice i.e. 4 downto 2 -> 2 downto 0.
    variable tmp_src    : std_logic_vector(src'length-1 downto 0) :=
      (others => '0');
    variable dst        : std_logic_vector(dstlen-1 downto 0);
    variable common_msb : integer := 0;
  begin
    tmp_src := src;
    if tmp_src'high < dst'high then
      common_msb                          := tmp_src'high;
      dst(dst'high downto tmp_src'length) :=
        (dst'high downto tmp_src'length => tmp_src(tmp_src'high));
    else
      common_msb := dst'high;
    end if;
    dst(common_msb downto 0) := tmp_src(common_msb downto 0);
    return dst;
  end tce_sxt;

  -- Converts boolean value to integer. Maps true as 1 and otherwise 0.
  function to_int (
    constant val : boolean)
    return integer is
  begin
    if val then
      return 1;
    else
      return 0;
    end if;
  end to_int;

  -- Converts std_logic_vector to natural while interpreting the vector
  -- as unsigned.
  function to_uint (
    constant interpret_as_unsigned : std_logic_vector)
    return natural is
  begin
    return to_integer(unsigned(interpret_as_unsigned));
  end to_uint;

  function return_highest(values : integer_array; numberOfValues : integer)
    return integer is

    variable highest : integer;

  begin

    highest := 0;
    for x in 0 to numberOfValues-1 loop
      if values(x) > highest then
        highest := values(x);
      end if;
    end loop;  -- x

    return highest;
  end return_highest;

end package body tce_util;

library ieee;
use ieee.std_logic_1164.all;

entity tce_util_inverter is
  port (
    data_in  : in  std_logic;
    data_out : out std_logic);
end tce_util_inverter;

architecture rtl of tce_util_inverter is
begin  -- rtl
  data_out <= not data_in;
end rtl;
