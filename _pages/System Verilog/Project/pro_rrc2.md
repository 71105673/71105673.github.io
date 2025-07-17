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