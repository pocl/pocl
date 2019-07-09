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

library IEEE;
use IEEE.Std_Logic_1164.all;

entity datapath_gate is
  generic (
    dataw : integer := 32);
  port (
    data_in  : in  std_logic_vector(dataw-1 downto 0);
    load_in  : in  std_logic;
    data_out : out std_logic_vector(dataw-1 downto 0);
    load_out : out std_logic);
end datapath_gate;

architecture rtl of datapath_gate is
begin  -- rtl
  data_out <= data_in when load_in = '1' else (others => '0');
  load_out <= load_in;
end rtl;

