---
title: "VLSI 1Week"
date: "2025-06-16~18"
thumbnail: "/assets/img/VLSI/스크린샷 2025-06-16 150820 - 복사본.png"
---

---
# Day 1~3 학습 내용 정리
***Synopsys Custom Compile***
---

**Day1: Tool의 기능 숙지를 위해 Inverter의 Schmatic 및 Simbol을 생성하고 HSPICE 시뮬레이션 동작**

**Day2: 이를 활용해 Un과 Up의 비율을 시뮬레이션의 파형 비교를 통해 이상적인 Width 값을 찾아냅니다.**

**Day3: 2NAND GATE를 만들고 같은 방법으로 Width를 구합니다.**

---
"Simulation Flow:
tool prime wave -> setup ->  model file -> 경로찾기 -> FF 선택 -> simulation -> option -> primesim HSPICE -> variables -> copy to design -> vin = 0 -> setup -> analysis -> dc -> Design Variable -> start stop size 등 시간 설정 -> click to add -> Schmatic에서 vin vout 선택 -> save_state -> 이름설정 -> simulation -> netlist and run -> plot"
---

Day 1
---
"Custom Compiler Open to MobaXterm"
![alt text](<../../../assets/img/VLSI/스크린샷 2025-06-16 150820.png>)

"Libray 생성"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151356.png>) 

"Tech PDK 32nm 설정"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151405.png>) 

"Inverter 만들기 (NOT)"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151428.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151458.png>) 

"Options 에서 Spacing 선택"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151525.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151544.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151607.png>) 

"인스턴스 추가"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 151920.png>) 

"인버터 생성 후 Q를 통해 Value 수정"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 153408.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 153621.png>) 

"Shift+X를 통해 룰체크 + 세이브"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 153957.png>)

"Symbol 생성"
 ![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 154441.png>) 
 ![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 154621.png>) 
 ![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 154624.png>) 

"Symbol 모양 수정" 
 ![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 160344.png>) 
 ![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 160734.png>)
 ![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 160916.png>)
 ![alt text](../../../assets/img/심볼.png)

"아날로그 모듈에서 Vdc 설정"
 ![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164350.png>) 

"시뮬레이션 모듈 (Test) 설정"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 163816.png>)
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 163827.png>) 

"경로 설정 tool prime wave -> setup ->  model file -> 경로찾기 -> FF 선택"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164127.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164136.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164406.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164458.png>) 

"시뮬레이션 옵션 설정 simulation -> option -> primesim HSPICE"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164538.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164554.png>) 

"디자인 카피 및 in 설정 variables -> copy to design -> vin = 0 -> setup"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164610.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164620.png>) 

"Analyses 설정 analysis -> dc -> Design Variable -> start stop size 등 시간 설정 -> click to add -> Schmatic에서 vin vout 선택 -> save_state -> 이름설정 -> simulation -> netlist and run -> plot"
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164634.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164658.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164857.png>) 
![text](<../../../assets/img/VLSI/스크린샷 2025-06-16 164909.png>) 

"시뮬레이션 결과"
![alt text](../../../assets/img/VLSI/시뮬레이션결과.png)


Day2
---
"여기서부터는 스윙을 통해 시뮬레이션을 진행합니다.

![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113540.png>)
"Variables 의 Design Parameterization 선택"

![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113609.png>)
![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113618.png>)
"ADD devices 선택 후 PMNOS를 선택"

![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113633.png>)
![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113705.png>)
"W를 WIDTH로 변수 설정한다."

![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113709.png>)
![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113714.png>)
![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113721.png>)
"Tool의 Parametic Analyses를 선택 후 Add new sweep 설정"

![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113727.png>)
![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113732.png>)
"WIDTH를 선택"

![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113821.png>)
![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113835.png>)
"WIDTH의 값을 설정한다"

![alt text](<../../../assets/img/VLSI/day3/스크린샷 2025-06-18 113851.png>)
"IN OUT을 Schmatic에서 선택한다."

![alt text](<../../../assets/img/VLSI/day3/1~2결과 및 Plot.png>)
![alt text](<../../../assets/img/VLSI/day3/값 1~2사이.png>)
"1~2사이 결과 값을 커서를 설정 후 보았을 때 1.1~1.2u 사이인 것을 확인"

![alt text](../../../assets/img/VLSI/day3/1.1~1.2설정.png)
"범위를 1.1~1.2u, 단위를 0.01u로 더 자세히 시뮬레이션"

![alt text](../../../assets/img/VLSI/day3/1.1~.1.2결과.png)
"1.15~1.16u사이 1.16u에 더 가까운 것을 확인, 소수 2번째 자리까지 유효 숫자로 계산하여 마무리"

![text](../../../assets/img/VLSI/day3/1.15~.16설정.png)
![text](../../../assets/img/VLSI/day3/1.15~1.16사이.png) 
![text](../../../assets/img/VLSI/day3/1.15~1.16사이2.png)
"추가로 1.15~1.16 사이 설정 후 시뮬레이션 진행, 자세한 값은 1.157u로 예상할 수 있습니다."

Day3
---
![alt text](<../../../assets/img/VLSI/day3/2NAND 핀까지.png>)
"2NAND의 Pin을 추가한 Schmatic을 그립니다."

![alt text](<../../../assets/img/VLSI/day3/한번에 여러 모듈 인스턴스 값 바꾸기.png>)
"해당 방법을 통해 여러 모듈 인스턴스 값을 한번에 바꿉니다."

![text](../../../assets/img/VLSI/day3/심볼만들기1.png)
![text](../../../assets/img/VLSI/day3/심볼만들기2.png)
"2NAND 심볼을 생성합니다.

![alt text](../../../assets/img/VLSI/day3/2nand_symbol.png)
"2NAND의 Symbol을 완성합니다.

![alt text](../../../assets/img/VLSI/day3/2nand_.png)
"Test Cellview를 만들어 in out 값을 넣어줍니다.

![alt text](../../../assets/img/VLSI/day3/0.5-1.5설정.png)
"![alt text](../../../assets/img/VLSI/day3/width0.9.png)
"설정 후 sim 결과 0.85~0.9u 사이 확인"

![alt text](../../../assets/img/VLSI/day3/0.85-0.95설정.png)
![alt text](../../../assets/img/VLSI/day3/0.9확정.png)
"0.85~0.95u 에서 0.01u단위로 측정하여 Width = 0.9u"

**추가 진행**

![text](../../../assets/img/VLSI/day3/0.9~0.95설정.png)
![text](<../../../assets/img/VLSI/day3/0.9~0.95 결과.png>) 
"0.9u~0.95u 사이를 0.005u 단위로 측정하여 0.9~0.905 사이 결정"

![text](../../../assets/img/VLSI/day3/0.9~0.91설정.png) 
![text](../../../assets/img/VLSI/day3/0.901결과.png)
"0.9~.0.91u 사이를 0.001u 단위로 측정하여 0.901u 값을 얻음"
---