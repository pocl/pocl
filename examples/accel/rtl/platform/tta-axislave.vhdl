-- Copyright (c) 2016 Nokia Research Center
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
-- Title      : AXI lite interface to TTA debugger and stream IO
-- Project    : 
-------------------------------------------------------------------------------
-- File       : tta-axislave.vhdl
-- Author     : Tommi Zetterman  <tommi.zetterman@nokia.com>
-- Company    : Nokia Research Center
-- Created    : 2014-06-23
-- Last update: 2017-06-01
-- Platform   : 
-- Standard   : VHDL'93
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- Copyright (c) 2014 Nokia Research Center
-------------------------------------------------------------------------------
-- Revisions  :
-- Date        Version  Author  Description
-- 2014-06-23  1.0      zetterma Created (as axi4dbgslave-rtl.vhdl
-- 2015-01-27  1.1      viitanet Modified into a processor wrapper
-- 2016-11-18  1.1      tervoa   Added full AXI4 interface
-- 2017-03-27  1.2      tervoa   Fixed burst transfer logic
-- 2017-04-25  1.3      tervoa   Combined entiy and architecture files
-- 2017-06-01  1.4      tervoa   Converted to memory buses with handshaking
-- 2017-06-01  1.5      tervoa   Fix address increment logic
-- 2018-07-30  1.6      tervoa   Support for optional sync reset
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tta_axislave is
  generic (
    -- Must be at least 2 + max(max(IMEMADDRWIDTH+IMEMWORDSEL,db_addr_width),
    --                          fu_LSU_addrw-2)
    -- where IMEMWORDSEL = bit_width((IMEMDATAWIDTH+31)/32)
    axi_addrw_g : integer := 17;
    axi_dataw_g  : integer := 32;
    axi_idw_g   : integer := 12;
    sync_reset_g : integer := 0
  );
  port (
    clk       : in std_logic;
    rstx      : in std_logic;
    -- Accelerator interface
    avalid_out : out std_logic;
    aready_in  : in  std_logic;
    aaddr_out  : out std_logic_vector(axi_addrw_g-2-1 downto 0);
    awren_out  : out std_logic;
    astrb_out  : out std_logic_vector(axi_dataw_g/8-1 downto 0);
    adata_out  : out std_logic_vector(axi_dataw_g-1 downto 0);
    rvalid_in  : in  std_logic;
    rready_out : out std_logic;
    rdata_in   : in  std_logic_vector(axi_dataw_g-1 downto 0);
    -- AXI slave port
    s_axi_awid     : in  STD_LOGIC_VECTOR (axi_idw_g-1 downto 0);
    s_axi_awaddr   : in  STD_LOGIC_VECTOR (axi_addrw_g-1 downto 0);
    s_axi_awlen    : in  STD_LOGIC_VECTOR (8-1 downto 0);
    s_axi_awsize   : in  STD_LOGIC_VECTOR (3-1 downto 0);
    s_axi_awburst  : in  STD_LOGIC_VECTOR (2-1 downto 0);
    s_axi_awvalid  : in  STD_LOGIC;
    s_axi_awready  : out STD_LOGIC;
    s_axi_wdata    : in  STD_LOGIC_VECTOR (31 downto 0);
    s_axi_wstrb    : in  STD_LOGIC_VECTOR (3 downto 0);
    s_axi_wvalid   : in  STD_LOGIC;
    s_axi_wready   : out STD_LOGIC;
    s_axi_bid      : out STD_LOGIC_VECTOR (axi_idw_g-1 downto 0);
    s_axi_bresp    : out STD_LOGIC_VECTOR (2-1 downto 0);
    s_axi_bvalid   : out STD_LOGIC;
    s_axi_bready   : in  STD_LOGIC;
    s_axi_arid     : in  STD_LOGIC_VECTOR (axi_idw_g-1 downto 0);
    s_axi_araddr   : in  STD_LOGIC_VECTOR (axi_addrw_g-1 downto 0);
    s_axi_arlen    : in  STD_LOGIC_VECTOR (8-1 downto 0);
    s_axi_arsize   : in  STD_LOGIC_VECTOR (3-1 downto 0);
    s_axi_arburst  : in  STD_LOGIC_VECTOR (2-1 downto 0);
    s_axi_arvalid  : in  STD_LOGIC;
    s_axi_arready  : out STD_LOGIC;
    s_axi_rid      : out STD_LOGIC_VECTOR (axi_idw_g-1 downto 0);
    s_axi_rdata    : out STD_LOGIC_VECTOR (31 downto 0);
    s_axi_rresp    : out STD_LOGIC_VECTOR (2-1 downto 0);
    s_axi_rlast    : out STD_LOGIC;
    s_axi_rvalid   : out STD_LOGIC;
    s_axi_rready   : in  STD_LOGIC
  );
end entity tta_axislave;

architecture rtl of tta_axislave is

  constant sync_reset_c : boolean := sync_reset_g /= 0;

  constant FIXED_BURST : std_logic_vector(1 downto 0) := "00";
  constant INCR_BURST  : std_logic_vector(1 downto 0) := "01";
  constant WRAP_BURST  : std_logic_vector(1 downto 0) := "10";

  constant AXI_OKAY    : std_logic_vector(1 downto 0) := "00";
  constant AXI_SLVERR  : std_logic_vector(1 downto 0) := "10";

  type u8_array is array (natural range <>) of unsigned(7 downto 0);
  constant burst_size_lut_c : u8_array(0 to 7) :=
          ("00000001", "00000010","00000100","00001000",
           "00010000", "00100000","01000000","10000000");

  type state_t is (S_READY, S_WRITE_DATA, S_FINISH_WRITE,
                   S_READ_DATA, S_FINISH_READ);
  signal state   : state_t;

  -- Output registers
  signal s_axi_awready_r : std_logic;
  signal s_axi_wready_r  : std_logic;
  signal s_axi_bid_r     : std_logic_vector(s_axi_bid'range);
  signal s_axi_bresp_r   : std_logic_vector(s_axi_bresp'range);
  signal s_axi_bvalid_r  : std_logic;
  signal s_axi_arready_r : std_logic;
  signal s_axi_rid_r     : std_logic_vector(s_axi_rid'range);
  signal s_axi_rdata_r   : std_logic_vector(s_axi_rdata'range);
  signal s_axi_rresp_r   : std_logic_vector(s_axi_rresp'range);
  signal s_axi_rlast_r   : std_logic;
  signal s_axi_rvalid_r  : std_logic;

  signal astrb_r, stall_strb_r    : std_logic_vector(astrb_out'range);
  signal adata_r, stall_data_r    : std_logic_vector(adata_out'range);
  signal aaddr_r, stall_addr_r    : std_logic_vector(aaddr_out'range);
  signal avalid_r, awren_r, stall_data_valid_r, rready_r : std_logic;

  signal burst_cnt_r     : unsigned(s_axi_arlen'range);
  signal read_cnt_r      : unsigned(s_axi_arlen'range);
  signal transaction_r   : std_logic_vector(s_axi_arid'range);

  signal burst_type_r : std_logic_vector(s_axi_arburst'range);
  signal burst_size_r : unsigned(7 downto 0);
  signal wrap_mask_r  : std_logic_vector(s_axi_araddr'range);
  signal axi_addr_r   : std_logic_vector(s_axi_araddr'range);

  signal axi_stall_r    : std_logic;


  function increment_addr(burst_type    : std_logic_vector(s_axi_arburst'range);
                          size          : unsigned;
                          wrap_mask     : std_logic_vector(axi_addrw_g-1 downto 0);
                          address       : std_logic_vector(axi_addrw_g-1 downto 0))
                          return std_logic_vector is
  variable address_tmp : std_logic_vector(axi_addrw_g-1 downto 0);
  begin
    case burst_type is
      when FIXED_BURST =>
        return address;
      when INCR_BURST =>
        return std_logic_vector(unsigned(address) + size);
      when WRAP_BURST => -- UNTESTED
        address_tmp := std_logic_vector(unsigned(address) + size);

        for I in address'range loop
          if wrap_mask(I) = '0' then
            address_tmp(I) := address(I);
          end if;
        end loop;

        return address_tmp;
      when others =>
        -- coverage off
        -- pragma translate_off
        assert false report "Unrecognized burst type" severity warning;
        -- pragma translate_on
        -- coverage on
        return address;
    end case;
  end function increment_addr;

  function wrap_mask(axsize : std_logic_vector(s_axi_arsize'range);
                     axlen  : std_logic_vector(s_axi_arlen'range))
                     return std_logic_vector is
  variable mask_temp : std_logic_vector(axi_addrw_g-1 downto 0);
  variable axsize_int : integer range 0 to 7
                      := to_integer(unsigned(axsize));
  begin
    for I in mask_temp'range loop
      if I < axsize_int then
        mask_temp(I) := '0';
      elsif I < axsize_int + axlen'high then
        mask_temp(I) := axlen(I - axsize_int);
      else
        mask_temp(I) := '0';
      end if;
    end loop;
    return mask_temp;
  end function wrap_mask;

begin

  sync : process(clk, rstx)
    variable axi_addr_v : std_logic_vector(axi_addr_r'range);
  begin
    if not sync_reset_c and rstx = '0' then
      s_axi_awready_r <= '0';
      s_axi_wready_r  <= '0';
      s_axi_bid_r     <= (others => '0');
      s_axi_bresp_r   <= (others => '0');
      s_axi_bvalid_r  <= '0';
      s_axi_arready_r <= '0';
      s_axi_rid_r     <= (others => '0');
      s_axi_rdata_r   <= (others => '0');
      s_axi_rresp_r   <= (others => '0');
      s_axi_rlast_r   <= '0';
      s_axi_rvalid_r  <= '0';

      avalid_r <= '0';
      aaddr_r  <= (others => '0');
      awren_r  <= '0';
      astrb_r  <= (others => '0');
      adata_r  <= (others => '0');
      rready_r <= '0';

      axi_addr_r          <= (others => '0');
      state               <= S_READY;
      burst_cnt_r         <= (others => '0');
      transaction_r       <= (others => '0');
      stall_strb_r        <= (others => '0');
      stall_data_r        <= (others => '0');
      stall_addr_r        <= (others => '0');
      stall_data_valid_r  <= '0';
    elsif rising_edge(clk) then
      if sync_reset_c and rstx = '0' then
        s_axi_awready_r <= '0';
        s_axi_wready_r  <= '0';
        s_axi_bid_r     <= (others => '0');
        s_axi_bresp_r   <= (others => '0');
        s_axi_bvalid_r  <= '0';
        s_axi_arready_r <= '0';
        s_axi_rid_r     <= (others => '0');
        s_axi_rdata_r   <= (others => '0');
        s_axi_rresp_r   <= (others => '0');
        s_axi_rlast_r   <= '0';
        s_axi_rvalid_r  <= '0';

        avalid_r <= '0';
        aaddr_r  <= (others => '0');
        awren_r  <= '0';
        astrb_r  <= (others => '0');
        adata_r  <= (others => '0');
        rready_r <= '0';

        axi_addr_r          <= (others => '0');
        state               <= S_READY;
        burst_cnt_r         <= (others => '0');
        transaction_r       <= (others => '0');
        stall_strb_r        <= (others => '0');
        stall_data_r        <= (others => '0');
        stall_addr_r        <= (others => '0');
        stall_data_valid_r  <= '0';
      else
        if s_axi_arready_r = '1' and s_axi_arvalid = '1' then
          s_axi_arready_r <= '0';
        end if;
        if s_axi_awready_r = '1' and s_axi_awvalid = '1' then
          s_axi_awready_r <= '0';
        end if;

        s_axi_wready_r  <= '0';
        rready_r        <= '0';
        s_axi_rlast_r   <= '0';

        -- valid_r(0)    <= '0';
        -- valid_r(2 downto 1) <= valid_r(1 downto 0);

        axi_stall_r   <= '0';
        case state is

          when S_READY =>
            if s_axi_awvalid = '1' then
              s_axi_awready_r <= '1';
              axi_addr_r      <= s_axi_awaddr;
              transaction_r   <= s_axi_awid;
              wrap_mask_r     <= wrap_mask(s_axi_awsize, s_axi_awlen);

              burst_size_r  <= burst_size_lut_c(to_integer(
                                                  unsigned(s_axi_awsize)));
              burst_type_r  <= s_axi_awburst;
              burst_cnt_r   <= unsigned(s_axi_awlen);
              state         <= S_WRITE_DATA;
            elsif s_axi_arvalid = '1' then
              s_axi_arready_r <= '1';
              transaction_r   <= s_axi_arid;
              wrap_mask_r     <= wrap_mask(s_axi_arsize, s_axi_arlen);

              axi_addr_r      <= s_axi_araddr;
              aaddr_r         <= s_axi_araddr(s_axi_araddr'high downto 2);
              avalid_r        <= '1';
              awren_r         <= '0';
              rready_r        <= '1';

              burst_size_r  <= burst_size_lut_c(to_integer(
                                                  unsigned(s_axi_arsize)));
              burst_cnt_r   <= unsigned(s_axi_arlen);
              read_cnt_r    <= unsigned(s_axi_arlen);
              burst_type_r  <= s_axi_arburst;
              state         <= S_READ_DATA;
            end if;

          when S_WRITE_DATA =>

            if avalid_r = '0' or stall_data_valid_r = '0' then
              s_axi_wready_r <= '1';
            else
              s_axi_wready_r <= '0';
            end if;

            if avalid_r = '1' and aready_in = '1' then
              if stall_data_valid_r = '1' then
                astrb_r <= stall_strb_r;
                adata_r <= stall_data_r;
                awren_r <= '1';
                aaddr_r <= stall_addr_r;
                stall_data_valid_r <= '0';
              else
                avalid_r <= '0';
              end if;
            end if;

            if s_axi_wvalid = '1' and s_axi_wready_r = '1' then
              if   (aready_in = '1' and stall_data_valid_r = '0')
                 or avalid_r = '0' then
                avalid_r <= '1';
                awren_r  <= '1';
                astrb_r  <= s_axi_wstrb;
                adata_r  <= s_axi_wdata;
                aaddr_r  <= axi_addr_r(axi_addr_r'high downto 2);
              else
                stall_data_valid_r <= '1';
                stall_data_r       <= s_axi_wdata;
                stall_strb_r       <= s_axi_wstrb;
                stall_addr_r       <= axi_addr_r(axi_addr_r'high downto 2);
              end if;

              if burst_cnt_r = 0 then
                s_axi_wready_r <= '0';
                s_axi_bresp_r  <= AXI_OKAY;
                s_axi_bvalid_r <= '1';
                s_axi_bid_r  <= transaction_r;
                state        <= S_FINISH_WRITE;
              else
                axi_addr_r <= increment_addr(burst_type_r, burst_size_r,
                                             wrap_mask_r, axi_addr_r);
                burst_cnt_r  <= burst_cnt_r - 1;
              end if;
            end if;


          when S_FINISH_WRITE =>
            if s_axi_bready = '1' then
              s_axi_bvalid_r <= '0';
            end if;

            if avalid_r = '1' and aready_in = '1' then
              if stall_data_valid_r = '1' then
                astrb_r <= stall_strb_r;
                adata_r <= stall_data_r;
                awren_r <= '1';

                stall_data_valid_r <= '0';
              else
                avalid_r <= '0';
                awren_r  <= '0';
              end if;
            end if;

            if avalid_r = '0' and s_axi_bvalid_r = '0' then
              state <= S_READY;
            end if;

          when S_READ_DATA =>
            if s_axi_rready = '1' and s_axi_rvalid_r = '1' then
              if stall_data_valid_r = '1' then
                s_axi_rdata_r      <= stall_data_r;
                stall_data_valid_r <= '0';
              else
                s_axi_rvalid_r <= '0';
              end if;
            end if;

            if avalid_r = '1' and aready_in = '1' then
              if read_cnt_r = 0 then
                avalid_r <= '0';
              else
                axi_addr_v := increment_addr(burst_type_r, burst_size_r,
                                             wrap_mask_r, axi_addr_r);
                axi_addr_r <= axi_addr_v;
                aaddr_r    <= axi_addr_v(axi_addr_v'high downto 2);
                read_cnt_r <= read_cnt_r - 1;
              end if;
            end if;

            if s_axi_rvalid_r = '0' or stall_data_valid_r = '0' then
              rready_r <= '1';
            else
              rready_r <= '0';
            end if;

            if rvalid_in = '1' and rready_r = '1' then
              if (s_axi_rready = '1'and stall_data_valid_r = '0')
                 or s_axi_rvalid_r = '0' then
                s_axi_rvalid_r <= '1';
                s_axi_rdata_r  <= rdata_in;
                s_axi_rresp_r  <= AXI_OKAY;
                s_axi_rid_r    <= transaction_r;
              else
                stall_data_valid_r <= '1';
                stall_data_r       <= rdata_in;
                rready_r           <= '0';
              end if;

              if burst_cnt_r = 0 then
                if (s_axi_rready = '1'and stall_data_valid_r = '0')
                   or s_axi_rvalid_r = '0' then
                   s_axi_rlast_r <= '1';
                end if;
                rready_r <= '0';
                state <= S_FINISH_READ;
              else
                burst_cnt_r  <= burst_cnt_r - 1;
              end if;
            end if;

          when S_FINISH_READ =>
            if s_axi_rready = '1' and s_axi_rvalid_r = '1' then
              if stall_data_valid_r = '1' then
                s_axi_rlast_r      <= '1';
                s_axi_rdata_r      <= stall_data_r;
                stall_data_valid_r <= '0';
              else
                s_axi_rvalid_r <= '0';
                s_axi_rlast_r  <= '0';
                state          <= S_READY;
              end if;
            end if;

        end case;
      end if;
    end if;
  end process;

  s_axi_awready  <= s_axi_awready_r;
  s_axi_wready   <= s_axi_wready_r;
  s_axi_bid      <= s_axi_bid_r;
  s_axi_bresp    <= s_axi_bresp_r;
  s_axi_bvalid   <= s_axi_bvalid_r;
  s_axi_arready  <= s_axi_arready_r;
  s_axi_rid      <= s_axi_rid_r;
  s_axi_rdata    <= s_axi_rdata_r;
  s_axi_rresp    <= s_axi_rresp_r;
  s_axi_rlast    <= s_axi_rlast_r;
  s_axi_rvalid   <= s_axi_rvalid_r;
  avalid_out     <= avalid_r;
  aaddr_out      <= aaddr_r;
  awren_out      <= awren_r;
  astrb_out      <= astrb_r;
  adata_out      <= adata_r;
  rready_out     <= rready_r;
end architecture rtl;
