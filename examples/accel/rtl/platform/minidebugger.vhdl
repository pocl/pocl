-- Copyright (c) 2017-2019 Tampere University
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
-- Title      : Minimal debugger, no breakpoint capabilities or bustrace
-- Project    :
-------------------------------------------------------------------------------
-- File       : minidebugger.vhdl
-- Author     : Kati Tervo
-- Company    : Tampere University
-- Created    : 2017-09-19
-- Last update: 2020-07-08
-- Platform   :
-- Standard   : VHDL'87
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author    Description
-- 2017-09-19  1.0      katte    Created
-- 2019-07-05  1.0      katte    Added AQL queue iterators
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
package register_pkg is
  -- Status registers
  constant TTA_STATUS        : integer := 0;
  constant TTA_PC            : integer := 1;
  constant TTA_CYCLECNT      : integer := 2;
  constant TTA_CYCLECNT_HIGH : integer := 3;
  constant TTA_LOCKCNT       : integer := 4;
  constant TTA_LOCKCNT_HIGH  : integer := 5;

  constant aql_addresspace_start_c : integer := 2**6;
  constant AQL_READ_IDX_LOW   : integer := 0 + aql_addresspace_start_c;
  constant AQL_READ_IDX_HIGH  : integer := 1 + aql_addresspace_start_c;
  constant AQL_WRITE_IDX_LOW  : integer := 2 + aql_addresspace_start_c;
  constant AQL_WRITE_IDX_HIGH : integer := 3 + aql_addresspace_start_c;

  constant control_addresspace_start_c : integer := 2**7;
  -- control register space
  constant TTA_DEBUG_CMD  : integer := 0 + control_addresspace_start_c;

  constant info_addresspace_start_c : integer := 2**7 + 2**6;
  -- info registers space: 0xC0..0xff
  constant TTA_DEVICECLASS      : integer := 0 + info_addresspace_start_c;
  constant TTA_DEVICE_ID        : integer := 1 + info_addresspace_start_c;
  constant TTA_INTERFACE_TYPE   : integer := 2 + info_addresspace_start_c;
  constant TTA_CORE_COUNT       : integer := 3 + info_addresspace_start_c;
  constant TTA_CTRL_SIZE        : integer := 4 + info_addresspace_start_c;
  constant TTA_IMEM_SIZE        : integer := 5 + info_addresspace_start_c;
  constant TTA_IMEM_START_LOW        : integer := 6 + info_addresspace_start_c;
  constant TTA_IMEM_START_HIGH        : integer := 7 + info_addresspace_start_c;

  constant TTA_CQMEM_SIZE_LOW        : integer := 8 + info_addresspace_start_c;
  constant TTA_CQMEM_SIZE_HIGH        : integer := 9 + info_addresspace_start_c;
  constant TTA_CQMEM_START_LOW        : integer := 10 + info_addresspace_start_c;
  constant TTA_CQMEM_START_HIGH        : integer := 11 + info_addresspace_start_c;

  constant TTA_DMEM_SIZE_LOW        : integer := 12 + info_addresspace_start_c;
  constant TTA_DMEM_SIZE_HIGH        : integer := 13 + info_addresspace_start_c;
  constant TTA_DMEM_START_LOW        : integer := 14 + info_addresspace_start_c;
  constant TTA_DMEM_START_HIGH        : integer := 15 + info_addresspace_start_c;

  constant TTA_FEATURE_FLAGS_LOW      : integer := 16 + info_addresspace_start_c;
  constant TTA_FEATURE_FLAGS_HIGH      : integer := 17 + info_addresspace_start_c;

  constant DEBUG_CMD_RESET    : integer := 0;
  constant DEBUG_CMD_CONTINUE : integer := 1;
  constant DEBUG_CMD_BREAK    : integer := 2;

  constant FF_AXI_MASTER : integer := 0;

end register_pkg;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.register_pkg.all;
use work.tce_util.all;

entity minidebugger is
  generic (
    data_width_g      : integer := 32;
    axi_addr_width_g  : integer;
    core_count_g      : integer;
    core_id_width_g   : integer;
    imem_data_width_g : integer;
    imem_addr_width_g : integer;
    dmem_data_width_g : integer;
    dmem_addr_width_g : integer;
    pmem_data_width_g : integer;
    pmem_addr_width_g : integer;
    axi_offset_high_g : integer;
    axi_offset_low_g  : integer;
    aql_queue_length_g : integer := 2
  );
  port (
    clk            : in std_logic;
    rstx           : in std_logic;
    -- AXI slave membus
    avalid_in      : in  std_logic;
    aready_out     : out std_logic;
    aaddr_in       : in  std_logic_vector(axi_addr_width_g-2-1 downto 0);
    awren_in       : in  std_logic;
    astrb_in       : in  std_logic_vector(data_width_g/8-1 downto 0);
    adata_in       : in  std_logic_vector(data_width_g-1 downto 0);
    rvalid_out     : out std_logic;
    rready_in      : in  std_logic;
    rdata_out      : out std_logic_vector(data_width_g-1 downto 0);
    core_sel_in    : in std_logic_vector(core_id_width_g-1 downto 0);

    tta_lockrq_out : out std_logic_vector(core_count_g-1 downto 0);
    tta_nreset_out : out std_logic_vector(core_count_g-1 downto 0);
    tta_pc_in      : in std_logic_vector(core_count_g*imem_addr_width_g-1
                                         downto 0);
    tta_locked_in  : in std_logic_vector(core_count_g-1 downto 0);
    tta_lockcnt_in : in std_logic_vector(core_count_g*64-1 downto 0);
    tta_cyclecnt_in : in std_logic_vector(core_count_g*64-1 downto 0);

    tta_read_idx_in         : in  std_logic_vector(64-1 downto 0);
    tta_read_idx_clear_out  : out std_logic_vector(0 downto 0);
    tta_write_idx_out       : out std_logic_vector(64-1 downto 0)

  );
end minidebugger;

architecture rtl of minidebugger is

  function mem_size(mem_addr_width:integer;mem_data_width:integer)
    return integer is
  begin
      if mem_data_width = 0 then
          return 0;
      else
          return 2**(mem_addr_width + bit_width(mem_data_width/8));
      end if;
  end function;

  constant device_class_c     : integer := 16#774#;
  constant device_id_c        : integer := 16#12345678#;
  constant interface_type_c   : integer := 2;
  constant aql_packet_size_c  : integer := 64;
  constant ctrl_size_c        : integer := 2**10;
  constant dmem_size_c        : integer := 2**(dmem_addr_width_g+
                                             bit_width(dmem_data_width_g/8));
  constant pmem_size_c        : integer := 2**(pmem_addr_width_g+
                                             bit_width(pmem_data_width_g/8));
  constant cqmem_size_c       : integer := aql_packet_size_c*aql_queue_length_g;
  constant imem_size_c        : integer := mem_size(imem_addr_width_g, imem_data_width_g);
  constant mem_sizes_c        : integer_array := (ctrl_size_c, dmem_size_c,
                                                  pmem_size_c, imem_size_c);
  constant segment_size_c     : integer := return_highest(mem_sizes_c, 4);

  constant cqmem_rel_offset_c : integer := dmem_size_c - cqmem_size_c;

  constant axi_offset_c : unsigned(63 downto 0) := unsigned(
            std_logic_vector(to_signed(axi_offset_high_g, 32)) &
            std_logic_vector(to_signed(axi_offset_low_g, 32)));
  constant imem_offset_c : unsigned(63 downto 0) := axi_offset_c +
                segment_size_c;
  constant cqmem_offset_c : unsigned(63 downto 0) := axi_offset_c + 
                2*ctrl_size_c + cqmem_rel_offset_c;
  constant buffermem_offset_c : unsigned(63 downto 0) := axi_offset_c + 
                4*ctrl_size_c;             

  constant feature_flags_c : std_logic_vector(63 downto 0) := (FF_AXI_MASTER => '1',
                                                               others => '0');

  signal tta_nreset_r, tta_nreset_rr : std_logic_vector(tta_nreset_out'range);
  signal tta_lockrq_r, tta_lockrq_rr : std_logic_vector(tta_lockrq_out'range);
  signal tta_locked_r, tta_locked_rr : std_logic_vector(tta_locked_in'range);
  signal tta_lockcnt_r : std_logic_vector(tta_lockcnt_in'range);
  signal tta_cyclecnt_r : std_logic_vector(tta_cyclecnt_in'range);
  signal tta_pc_r, tta_pc_rr : std_logic_vector(tta_pc_in'range);


  signal idx_reset_r             : std_logic;
  signal write_idx_r, read_idx_r : std_logic_vector(64-1 downto 0);

  signal rvalid_r : std_logic;
  signal rdata_r : std_logic_vector(rdata_out'range);
begin

  sync : process(clk, rstx)
    variable core_id_v : integer;
    variable command   : std_logic_vector(3 downto 0);
  begin
    if (rstx = '0') then
      tta_nreset_r   <= (others => '0');
      tta_nreset_rr  <= (others => '0');
      tta_lockrq_r   <= (others => '0');
      tta_lockrq_rr  <= (others => '0');
      tta_locked_r   <= (others => '0');
      tta_locked_rr  <= (others => '0');
      tta_lockcnt_r  <= (others => '0');
      tta_cyclecnt_r <= (others => '0');
      tta_pc_r       <= (others => '0');
      tta_pc_rr      <= (others => '0');

      rvalid_r      <= '0';
      rdata_r       <= (others => '0');

      read_idx_r    <= (others => '0');
      write_idx_r   <= (others => '0');
      idx_reset_r   <= '0';
    elsif rising_edge(clk) then
      idx_reset_r <= tta_nreset_r(0);
      if rready_in = '1' then
        rvalid_r <= '0';
      end if;
      -- Doubly registered to decouple the (high fanout) signals
      tta_locked_r <= tta_locked_in;
      tta_locked_rr <= tta_locked_r;

      tta_lockrq_rr <= tta_lockrq_r;
      tta_nreset_rr <= tta_nreset_r;

      tta_lockcnt_r  <= tta_lockcnt_in;
      tta_cyclecnt_r <= tta_cyclecnt_in;

      tta_pc_r  <= tta_pc_in;
      tta_pc_rr <= tta_pc_r;

      read_idx_r <= tta_read_idx_in;

      if avalid_in = '1' and rvalid_r = '0' then

        core_id_v := to_integer(unsigned(aaddr_in(aaddr_in'high downto 8)));

        if awren_in = '0' then

          rvalid_r <= '1';
          case to_integer(unsigned(aaddr_in(7 downto 0))) is
            when TTA_STATUS =>
              rdata_r <= (others => '0');
              if core_id_v < core_count_g then
                rdata_r(2 downto 0) <= (not tta_nreset_r(core_id_v))
                                       & tta_lockrq_r(core_id_v)
                                       & tta_locked_rr(core_id_v);
              end if;
            when TTA_PC =>
              if core_id_v < core_count_g then
                rdata_r                  <= (others => '0');
                rdata_r(imem_addr_width_g-1 downto 0) <=
                  tta_pc_rr(imem_addr_width_g*(core_id_v+1)-1 downto imem_addr_width_g*core_id_v);
              end if;
            when TTA_CYCLECNT =>
              if core_id_v< core_count_g then
                rdata_r <= tta_cyclecnt_r(64*core_id_v+32-1 downto core_id_v*64);
              end if;
            when TTA_CYCLECNT_HIGH =>
              if core_id_v< core_count_g then
                rdata_r <= tta_cyclecnt_r(64*core_id_v+64-1 downto core_id_v*64+32);
              end if;
            when TTA_LOCKCNT =>
              if core_id_v< core_count_g then
                rdata_r <= tta_lockcnt_r(64*core_id_v+32-1 downto core_id_v*64);
              end if;
            when TTA_LOCKCNT_HIGH =>
              if core_id_v< core_count_g then
                rdata_r <= tta_lockcnt_r(64*core_id_v+64-1 downto core_id_v*64+32);
              end if;

            when AQL_READ_IDX_LOW   =>
              rdata_r <= read_idx_r(32-1 downto 0);
            when AQL_READ_IDX_HIGH  =>
              rdata_r <= read_idx_r(64-1 downto 32);
            when AQL_WRITE_IDX_LOW  =>
              rdata_r <= write_idx_r(32-1 downto 0);
            when AQL_WRITE_IDX_HIGH =>
              rdata_r <= write_idx_r(64-1 downto 32);

            when TTA_DEVICECLASS =>
              rdata_r <= std_logic_vector(to_unsigned(device_class_c, 32));
            when TTA_DEVICE_ID =>
              rdata_r <= std_logic_vector(to_unsigned(device_id_c, 32));

            when TTA_INTERFACE_TYPE =>
              rdata_r <= std_logic_vector(to_unsigned(interface_type_c, 32));
            when TTA_CORE_COUNT =>
              rdata_r <= std_logic_vector(to_unsigned(core_count_g, 32));
            when TTA_CTRL_SIZE =>
              rdata_r <= std_logic_vector(to_unsigned(ctrl_size_c, 32));
            when TTA_IMEM_SIZE =>
              if imem_data_width_g = 0 then
                rdata_r <= std_logic_vector(to_unsigned(0, 32));
              else
                rdata_r <= std_logic_vector(to_unsigned(imem_size_c, 32));
              end if;
            when TTA_IMEM_START_LOW =>
                rdata_r <= std_logic_vector(imem_offset_c(31 downto 0));
            when TTA_IMEM_START_HIGH =>
                rdata_r <= std_logic_vector(imem_offset_c(63 downto 32));

            when TTA_CQMEM_SIZE_LOW =>
                rdata_r <= std_logic_vector(to_unsigned(cqmem_size_c,32));
            when TTA_CQMEM_SIZE_HIGH =>
                rdata_r <= (others => '0');
            when TTA_CQMEM_START_LOW =>
                rdata_r <= std_logic_vector(cqmem_offset_c(31 downto 0));
            when TTA_CQMEM_START_HIGH =>
                rdata_r <= std_logic_vector(cqmem_offset_c(63 downto 32));

            when TTA_DMEM_SIZE_LOW =>
                rdata_r <= std_logic_vector(to_unsigned(pmem_size_c,32));
            when TTA_DMEM_SIZE_HIGH =>
                rdata_r <= (others => '0');
            when TTA_DMEM_START_LOW =>
                rdata_r <= std_logic_vector(buffermem_offset_c(31 downto 0));
            when TTA_DMEM_START_HIGH =>
                rdata_r <= std_logic_vector(buffermem_offset_c(63 downto 32));

            when TTA_FEATURE_FLAGS_LOW =>
                rdata_r <= feature_flags_c(31 downto 0);
            when TTA_FEATURE_FLAGS_HIGH =>
                rdata_r <= feature_flags_c(63 downto 32);
            when others =>
              rdata_r <= (others => '0');
          end case;
        else
          if to_integer(unsigned(aaddr_in(7 downto 0))) = TTA_DEBUG_CMD then
            command := adata_in(command'range);
            if core_id_v < core_count_g then
              if command(DEBUG_CMD_CONTINUE) = '1'  then
                tta_nreset_r <= (others => '1');
                tta_lockrq_r <= (others => '0');
              end if;

              if command(DEBUG_CMD_BREAK) = '1' then
                tta_lockrq_r <= (others => '1');
              end if;

              if command(DEBUG_CMD_RESET) = '1' then
                tta_nreset_r <= (others => '0');
              end if;
            end if;
          elsif to_integer(unsigned(aaddr_in(7 downto 0))) = AQL_WRITE_IDX_LOW then
            write_idx_r <= std_logic_vector(unsigned(write_idx_r)
                                            + unsigned(adata_in));
          end if;
        end if;
      end if;
    end if;
  end process;

  aready_out <= not rvalid_r;
  rdata_out  <= rdata_r;
  rvalid_out <= rvalid_r;

  tta_lockrq_out <= tta_lockrq_rr;
  tta_nreset_out <= tta_nreset_rr;
  tta_write_idx_out <= write_idx_r;
  tta_read_idx_clear_out(0) <= not idx_reset_r;

end architecture rtl;
