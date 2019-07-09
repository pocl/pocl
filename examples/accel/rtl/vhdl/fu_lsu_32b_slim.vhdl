-- Copyright (c) 2017-2019 Tampere University of Technology.
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
-------------------------------------------------------------------------------
-- Title      : LSU for AlmaIF Integrator
-- Project    : Almarvi
-------------------------------------------------------------------------------
-- File       : fu_lsu_32b_slim.vhdl
-- Author     : Kati Tervo
-- Company    :
-- Created    : 2017-05-31
-- Last update: 2019-05-22
-- Platform   :
-------------------------------------------------------------------------------
-- Description: 32 bit wide LSU with parametric endianness
--              External ports:
--  | Signal     | Comment
--  ---------------------------------------------------------------------------
--  |            | Access channel
--  | avalid_out | LSU asserts avalid when it has a valid command for the memory
--  | aready_in  | and considers the command accepted when both avalid and
--  |            | aready are asserted on the same clock cycle
--  |            |
--  | aaddr_out  | Address of the access, word-addressed
--  | awren_out  | High for write, low for read
--  | astrb_out  | Bytewise write strobe
--  | adata_out  | Data to write, if any
--  |            |
--  |            | Response channel
--  | rvalid_in  | Works like avalid/aready, in the other direction. The memory
--  | rready_out | must keep read accesses in order.
--  |            |
--  | rdata_in   | Read data, if any.
--  ---------------------------------------------------------------------------
--
-- Revisions  :
-- Date        Version  Author  Description
-- 2017-05-31  1.0      katte   Created
-- 2019-05-22  1.0      katte   Removed sign-extending operations (LD16/LD8)
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity fu_lsu_32b_slim is
  generic (
    addrw_g           : integer := 11;
    -- This will be converted to slv later, but passed as integer because
    -- some tools only handle ints (esp. mixed-language simulation or synthesis)
    register_bypass_g : integer := 2;
    -- Ditto, converted to boolean
    little_endian_g   : integer := 1
  );
  port(
    clk           : in std_logic;
    rstx          : in std_logic;

    -- Control signals
    glock_in      : in std_logic;
    glockreq_out  : out std_logic;

    -- Address port, triggers
    t1_address_in : in  std_logic_vector(addrw_g-1 downto 0);
    t1_load_in    : in  std_logic;
    t1_opcode_in  : in  std_logic_vector(2 downto 0);
    -- Data operand and output ports
    o1_data_in    : in  std_logic_vector(32-1 downto 0);
    o1_load_in    : in  std_logic;
    r1_data_out   : out std_logic_vector(32-1 downto 0);
    -- external memory unit interface:
    -- Access channel
    avalid_out    : out std_logic_vector(0 downto 0);
    aready_in     : in  std_logic_vector(0 downto 0);
    aaddr_out     : out std_logic_vector(addrw_g-2-1 downto 0);
    awren_out     : out std_logic_vector(0 downto 0);
    astrb_out     : out std_logic_vector(4-1 downto 0);
    adata_out     : out std_logic_vector(32-1 downto 0);
    -- Read channel
    rvalid_in     : in  std_logic_vector(0 downto 0);
    rready_out    : out std_logic_vector(0 downto 0);
    rdata_in      : in  std_logic_vector(32-1 downto 0)
  );
end fu_lsu_32b_slim;

architecture rtl of fu_lsu_32b_slim is

  constant is_little_endian_c : boolean := little_endian_g /= 0;

  function pipeline_depth(reg_bypass : std_logic_vector) return integer is
  variable result : integer := 0;
  begin
    if reg_bypass(0) = '0' then
      return 3;
    else
      return 2;
    end if;
  end pipeline_depth;

  type operations_t is (LD32, LD16, LD8, STORE, NOP);
  type pipeline_state_t is record
      operation   : operations_t;
      addr_low    : std_logic_vector(2 - 1 downto 0);
  end record;

  -------------------------------------------------------------------------
  -- register_bypass_c: if element n is '1', bypass corresponding registers
  --  n  |  register name | description
  -- ----------------------------------------------------------------------
  --  0  |  rdata_r       | Registers rdata_in
  --  1  |  result_r      | FU output port register
  -------------------------------------------------------------------------
  constant register_bypass_c : std_logic_vector(3 - 1 downto 0)
              := std_logic_vector(to_unsigned(register_bypass_g, 3));
  constant data_width_c : integer := 32;
  constant byte_width_c : integer := data_width_c/8;

  type pipeline_regs_t is
                       array (pipeline_depth(register_bypass_c) - 1 downto 0)
                       of pipeline_state_t;
  signal pipeline_r    : pipeline_regs_t;

  signal o1_data_r      : std_logic_vector(data_width_c - 1 downto 0);
  signal write_data     : std_logic_vector(data_width_c - 1 downto 0);
  signal glockreq       : std_logic;
  signal fu_glock       : std_logic;

  -- Access channel registers
  signal avalid_r    : std_logic;
  signal aaddr_r     : std_logic_vector(aaddr_out'range);
  signal awren_r     : std_logic;
  signal astrb_r     : std_logic_vector(astrb_out'range);
  signal adata_r     : std_logic_vector(adata_out'range);
  signal rready_r    : std_logic;

  -- Signals between memory and LSU output
  signal read_data     : std_logic_vector(data_width_c - 1 downto 0);
  signal rdata_r       : std_logic_vector(data_width_c - 1 downto 0);
  signal result        : std_logic_vector(data_width_c - 1 downto 0);
  signal result_r      : std_logic_vector(data_width_c - 1 downto 0);

  ----------------------------------------------------------------------------
  -- Lookup table for converting between BE and LE
  --  --------------------------------------
  --  | LE OPC    | LE # | BE OPC   | BE # |
  --  --------------------------------------
  --  | OPC_LD32  | 0    | OPC_LDW  | 2    |
  --  | OPC_LDU16 | 1    | OPC_LDHU | 0    |
  --  | OPC_LDU8  | 2    | OPC_LDQU | 1    |
  --  | OPC_ST16  | 3    | OPC_STH  | 3    |
  --  | OPC_ST32  | 4    | OPC_STW  | 5    |
  --  | OPC_ST8   | 5    | OPC_STQ  | 4    |
  --  --------------------------------------
  ----------------------------------------------------------------------------
  type opcode_array is array (0 to 5, 0 to 1) of integer;
  constant opcode_lut : opcode_array := ((0, 0), (4, 1), (1, 2),
                                         (3, 3), (5, 4), (4, 5));
  -- LE opcodes for the switch statements
  constant OPC_LD32  : integer := 0;
  constant OPC_LDU16 : integer := 1;
  constant OPC_LDU8  : integer := 2;
  constant OPC_ST16  : integer := 3;
  constant OPC_ST32  : integer := 4;
  constant OPC_ST8   : integer := 5;
begin
  -- Design-wide assertions
  -- coverage off
  -- synthesis translate_off
  assert register_bypass_g < 4 and register_bypass_g >= 0
    report "register_bypass_g out of range"
    severity failure;

  assert little_endian_g = 0 or little_endian_g = 1
    report "little_endian_g should be either 0 or 1"
    severity failure;
  -- coverage on
  -- synthesis translate_on

  avalid_out(0) <= avalid_r;
  awren_out(0)  <= awren_r;
  aaddr_out     <= aaddr_r;
  astrb_out     <= astrb_r;
  adata_out     <= adata_r;
  rready_out(0) <= rready_r;

  shadow_registers_sync : process(clk, rstx)
  begin
    if rstx = '0' then
      o1_data_r      <= (others => '0');
      result_r       <= (others => '0');

    elsif rising_edge(clk) then

      if fu_glock = '0' then
        if o1_load_in = '1' then
          o1_data_r  <= o1_data_in;
        end if;

        result_r       <= result;
      end if;
    end if;
  end process;

  write_data <= o1_data_in when o1_load_in = '1' else o1_data_r;

  gen_lockreq : process(rready_r, rvalid_in, avalid_r, aready_in,
                        glock_in, glockreq)
  begin
    if    (rready_r = '1' and rvalid_in(0) = '0')
       or (avalid_r = '1' and aready_in(0) = '0') then
      glockreq <= '1';
    else
      glockreq <= '0';
    end if;

    fu_glock     <= glockreq or glock_in;
    glockreq_out <= glockreq;
  end process gen_lockreq;

-------------------------------------------------------------------------------
-- Byte shifts and enable signal logic based on most recent opcode
-------------------------------------------------------------------------------
  access_channel_sync : process(clk, rstx)
    variable opcode : integer range 0 to 7;
    variable addr_low : std_logic_vector(2 - 1 downto 0);
    variable byte_offset : integer range 3 downto 0;
  begin
    if rstx = '0' then
      pipeline_r  <= (others => (NOP, "00"));
      aaddr_r     <= (others => '0');
      astrb_r     <= (others => '0');
      adata_r     <= (others => '0');
      avalid_r    <= '0';
      awren_r     <= '0';
    elsif rising_edge(clk) then

      if avalid_r = '1' and aready_in(0) = '1' then
        avalid_r <= '0';
      end if;

      if fu_glock = '0' then
        pipeline_r(pipeline_r'high downto 1)
            <= pipeline_r(pipeline_r'high-1 downto 0);

        if t1_load_in = '1' then
          avalid_r <= '1';
          aaddr_r  <= t1_address_in(t1_address_in'high downto 2);
          opcode   := opcode_lut(to_integer(unsigned(t1_opcode_in)),
                                 little_endian_g);
          addr_low := t1_address_in(2 - 1 downto 0);
          case opcode is
            when OPC_LD32 =>
              pipeline_r(0)  <= (LD32, addr_low);
              awren_r        <= '0';

            when OPC_LDU16 =>
              pipeline_r(0)  <= (LD16, addr_low);
              awren_r        <= '0';
            when OPC_LDU8 =>
              pipeline_r(0)  <= (LD8,  addr_low);
              awren_r        <= '0';

            when OPC_ST32 =>
              pipeline_r(0)  <= (STORE, "00");
              awren_r        <= '1';
              astrb_r        <= "1111";
              adata_r        <= write_data;

            when OPC_ST16 =>
              pipeline_r(0)  <= (STORE, "00");
              awren_r        <= '1';

              adata_r        <= (others => '0');
              if addr_low(1) = '1' xnor is_little_endian_c then
                adata_r(32 - 1 downto 16) <= write_data(16 - 1 downto 0);
                astrb_r                   <= "1100";
              else
                adata_r(16 - 1 downto 0)  <= write_data(16 - 1 downto 0);
                astrb_r                   <= "0011";
              end if;

            when others => -- OPC_ST8
              pipeline_r(0)  <= (STORE, "00");
              awren_r        <= '1';

              if is_little_endian_c then
                byte_offset := to_integer(unsigned(addr_low));
              else
                byte_offset := 3 - to_integer(unsigned(addr_low));
              end if;

              adata_r  <= (others => '0');
              adata_r(8*byte_offset + 7 downto 8*byte_offset)
                          <= write_data(8 - 1 downto 0);
              astrb_r(byte_offset) <= '1';

          end case;
        else
          pipeline_r(0)  <= (NOP, "00");
          avalid_r       <= '0';
          awren_r        <= '0';
          aaddr_r        <= (others => '0');
          astrb_r        <= (others => '0');
          adata_r        <= (others => '0');
        end if;

      end if;
    end if;
  end process access_channel_sync;

  read_channel_sync : process(clk, rstx)
  begin
    if rstx = '0'then
      rready_r <= '0';
      rdata_r  <= (others => '0');
    elsif rising_edge(clk) then

      if rvalid_in = "1" and rready_r = '1' then
        rdata_r       <= rdata_in;
        rready_r      <= '0';
      end if;

      if pipeline_r(0).operation /= STORE and pipeline_r(0).operation /= NOP
         and fu_glock = '0' then
        rready_r <= '1';
      end if;
    end if;
  end process read_channel_sync;

  bypass_read_data_register : if register_bypass_c(0) = '1' generate
    read_data <= rdata_in when rready_r = '1' else rdata_r;
  end generate;

  use_mem_output_register : if register_bypass_c(0) = '0' generate
    read_data <= rdata_r;
  end generate;

-------------------------------------------------------------------------------
-- Read data formatting based on load type
-------------------------------------------------------------------------------
  shift_and_sext : process(pipeline_r, read_data, result_r)
    variable read_16 : std_logic_vector(16 - 1 downto 0);
    variable read_8  : std_logic_vector(8 - 1 downto 0);
    variable byte_offset : integer range 3 downto 0;
  begin
    case pipeline_r(pipeline_r'high).operation is

      when LD32 =>
        result   <= read_data;

      when LD16 =>
        if      pipeline_r(pipeline_r'high).addr_low(1) = '1'
           xnor is_little_endian_c then
          read_16   := read_data(31 downto 16);
        else
          read_16   := read_data(15 downto 0);
        end if;

        result <= (others => '0');
        result(16 - 1 downto 0) <= read_16;

      when LD8  =>
        if is_little_endian_c then
          byte_offset
              := to_integer(unsigned(pipeline_r(pipeline_r'high).addr_low));
        else
          byte_offset
              := 3 - to_integer(unsigned(pipeline_r(pipeline_r'high).addr_low));
        end if;
        read_8      := read_data(8*byte_offset + 7 downto 8*byte_offset);

        result <= (others => '0');
        result(8 - 1 downto 0) <= read_8;

      when others => -- NOP, STORE
        result <= result_r;
    end case;
  end process shift_and_sext;

  bypass_r1data_register : if register_bypass_c(1) = '1' generate
    r1_data_out <= result;
  end generate;

  use_r1data_register    : if register_bypass_c(1) = '0' generate
    r1_data_out <= result_r;
  end generate;

end rtl;
