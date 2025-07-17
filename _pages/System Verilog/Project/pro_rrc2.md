---
title: "project RRC-2" 
date: "2025-07-17"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# Synthesis 

## run_rrc_filter.dc
```
dc_shell -f rrc_filter.tcl | tee run.log
```
## rrc_filter.sdc
```verilog
#  case &  clock definition
#-----------------------------------------------------------------------
## FF to FF clock period margin
set CLK_MGN  0.7
## REGIN, REGOUT setup/hold margin
#set io_dly   0.15
set io_dly   0.05


#set per200  "5.00";  # ns -> 200 MHz
#set per5000  "5000.00";  # ps -> 200 MHz
#set per1000  "1000.00";  # ps -> 1000 MHz
set per1250 "800.00";  # ps -> 1250 MHz


#set dont_care   "2";
#set min_delay   "0.3";

#set clcon_clk_name "CLK"
#set cnt_clk_period "[expr {$per200*$CLK_MGN}]"
set cnt_clk_period "[expr {$per1250*$CLK_MGN}]"
set cnt_clk_period_h "[expr {$cnt_clk_period/2.0}]"

### I/O DELAY per clock speed
#set cnt_clk_delay         [expr "$per200 * $CLK_MGN * $io_dly"]
set cnt_clk_delay         [expr "$per1250 * $CLK_MGN * $io_dly"]

```
## rrc_filter.list
```
lappend search_path  ../verilog
set net_list "\
../verilog/rrc_filter.sv\
"
 analyze -format sverilog -library WORK $net_list
```
## run_rrc_filter.dc        
```
dc_shell -f rrc_filter.tcl | tee run.log
```

# synthesis 결과
확인 결과 Setup Violation 발생

## Timing Max -> Setup
```

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  shift_din_reg_15__3_/CLK (SC7P5T_DFFRQX4_S_CSC20L)      0.00       0.00 r
  shift_din_reg_15__3_/Q (SC7P5T_DFFRQX4_S_CSC20L)       50.61      50.61 r
  U999/Z (SC7P5T_INVX6_CSC20L)                           10.13      60.75 f
  U1035/S (SC7P5T_FAX6_CSC20L)                           52.11     112.86 r
  U683/Z (SC7P5T_XNR2X2_CSC20L)                          36.98     149.84 f
  U682/Z (SC7P5T_XNR2X2_CSC20L)                          34.38     184.23 r
  U736/CO (SC7P5T_FAX2_A_CSC20L)                         38.24     222.47 r
  U758/S (SC7P5T_FAX2_A_CSC20L)                          57.55     280.02 f
  U846/CO (SC7P5T_FAX2_A_CSC20L)                         35.60     315.62 f
  U910/S (SC7P5T_FAX2_A_CSC20L)                          53.26     368.88 r
  U255/Z (SC7P5T_XOR3X2_CSC20L)                          42.58     411.46 f
  U494/Z (SC7P5T_NR2X4_CSC20L)                           14.97     426.43 r
  U160/Z (SC7P5T_NR2X3_CSC20L)                            9.68     436.11 f
  U115/Z (SC7P5T_INVX3_CSC20L)                            7.99     444.10 r
  U256/Z (SC7P5T_NR2IAX2_CSC20L)                          7.34     451.45 f
  U237/Z (SC7P5T_NR2X2_MR_CSC20L)                        14.95     466.40 r
  U253/Z (SC7P5T_OAI21X2_CSC20L)                         15.18     481.58 f
  U239/Z (SC7P5T_XNR2X1_CSC20L)                          23.06     504.65 f
  U739/Z (SC7P5T_OAI21X2_CSC20L)                         11.82     516.47 r
  data_out_reg_4_/D (SC7P5T_SDFFRQX2_A_CSC20L)            0.00     516.47 r
  data arrival time                                                516.47

  clock cnt_clk (rise edge)                             560.00     560.00
  clock network delay (ideal)                             0.00     560.00
  clock uncertainty                                     -50.00     510.00
  data_out_reg_4_/CLK (SC7P5T_SDFFRQX2_A_CSC20L)          0.00     510.00 r
  library setup time                                    -39.62     470.38
  data required time                                               470.38
  --------------------------------------------------------------------------
  data required time                                               470.38
  data arrival time                                               -516.47
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                 -46.09
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 093211.png>)

## Timing Min -> Hold
```
  Path Type: min

  Point                                                   Incr       Path
  --------------------------------------------------------------------------
  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  shift_din_reg_5__6_/CLK (SC7P5T_DFFRQX4_S_CSC20L)       0.00       0.00 r
  shift_din_reg_5__6_/Q (SC7P5T_DFFRQX4_S_CSC20L)        48.48      48.48 f
  shift_din_reg_6__6_/D (SC7P5T_DFFRQX1_AS_CSC20L)        0.00      48.48 f
  data arrival time                                                 48.48

  clock cnt_clk (rise edge)                               0.00       0.00
  clock network delay (ideal)                             0.00       0.00
  clock uncertainty                                      50.00      50.00
  shift_din_reg_6__6_/CLK (SC7P5T_DFFRQX1_AS_CSC20L)      0.00      50.00 r
  library hold time                                      12.95      62.95
  data required time                                                62.95
  --------------------------------------------------------------------------
  data required time                                                62.95
  data arrival time                                                -48.48
  --------------------------------------------------------------------------
  slack (VIOLATED)                                                 -14.47
```
![alt text](<../../../assets/img/SystemVerilog/rrc2/스크린샷 2025-07-17 093723.png>)

