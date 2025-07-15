---
title: "Day-2 Sysyem Verilog" 
date: "2025-07-15"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

## Setup && Hold Time
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 093520.png>)

### 📌 1. Setup Time (설정 시간)

- **정의**: 클럭 상승/하강 에지 **이전**에 데이터 입력(D)이 **안정적으로 유지되어야 하는 최소 시간**.
- **의미**: 클럭 에지가 발생하기 전, 데이터를 미리 준비해야 하는 시간.
- **충족되지 않으면**: 레지스터가 잘못된 데이터를 캡처하거나 메타안정성(Metastability) 문제가 발생할 수 있음.

### 📌 2. Hold Time (유지 시간)

- **정의**: 클럭 상승/하강 에지 **이후**에도 데이터 입력(D)이 **안정적으로 유지되어야 하는 최소 시간**.
- **의미**: 클럭 에지가 발생한 후에도 잠시 동안 데이터를 유지해야 한다는 의미.
- **충족되지 않으면**: 잘못된 데이터가 캡처되거나 메타안정성 발생 가능.

### Recovery / Removal time
![alt text](<../../../assets/img/SystemVerilog/스크린샷 2025-07-15 092225.png>)
