---
title: "Day-1 Sysyem Verilog" 
date: "2025-07-14"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# SystemVerilog 기본 개념 정리

## 🧪 Testbench 관점에서의 용어

- **Driver**: DUT의 **입력**을 담당  
  → 출력 시에는 `FF(clk)` 처리 필요
- **Monitor**: DUT의 **출력**을 담당  
  → 출력 시에는 **비동기(non-clk)** 처리 필요
- **Sample**: Monitor가 DUT의 출력을 **샘플링**하는 것

---

## 🔧 task vs function

| 항목       | `task`                       | `function`                 |
|------------|------------------------------|----------------------------|
| 시간 흐름   | 있음 (`#`, `@` 사용 가능)       | 없음                        |
| 반환 값     | 없음                         | 있음                        |
| 사용 용도   | 동작 순서나 지연 포함 작업 처리 | 계산, 간단한 로직 등        |

---

## 🗂 파일 확장자

- `.sv`: **SystemVerilog** 파일 → **컴파일러 필요**
- `.v`: **Verilog** 파일

> ✅ `.sv`는 지원 가능한 시뮬레이터나 툴에서만 작동하므로 환경 설정 필요

---

## 🧱 Class 관련 개념

- Class는 **함수형 구조**를 가지며 `task` 사용 가능
- 그러나 내부에서는 `initial`, `always` 사용 불가  
  → 이유: **시간 흐름**이 없기 때문

---

## ⏱ Delay 표현 방법

- `#`: **timescale 기준의 지연**
- `##`: **clock 주기를 지난다는 의미**

```sv
##4  // clk을 4번 지나는 의미

// 어떤 클럭인지 명확하지 않기 때문에 아래처럼 작성하는 것이 안전:
repeat(4) @(posedge clk);
```

## ⚙️ 기본 데이터 타입
기본 Default 값:

- 값: x

- Signed: unsigned

- 상태: 4-state (0, 1, z, x)

| 타입        | 크기    | Signed | 상태      |
| --------- | ----- | ------ | ------- |
| `integer` | 32bit | Yes    | 4-state |
| `int`     | 32bit | No     | 2-state |

## 📐 Dynamic Array 관련

- 갯수 반드시 선언 필수 → 안 하면 컴파일 에러 발생

- 남는 부분은 X로 초기화됨

- delete() 사용 시 전부 X로 초기화