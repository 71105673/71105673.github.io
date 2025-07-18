---
title: "project FFT" 
date: "2025-07-17"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# 주어진 MATLAB Code
## test_fft_float_stu.m
```matlab
% Test fft function (fft_float) 
% Added on 2025/07/02 by jihan 
 fft_mode = 1; % '0': ifft, '1': fft
 N = 512;

 [cos_float, cos_fixed] = cos_in_gen(fft_mode, N);

 [fft_out, module2_out] = fft_float(1, cos_float); % Floating-point fft (fft) : Cosine 
```
## fft_float.m
```matlab
% Added on 2025/07/01 by jihan 
function [fft_out, module2_out] = fft_float(fft_mode, fft_in)

 shift = 0;
 SIM_FIX = 0; % 0: float, 1: fixed

 if (fft_mode==1) % fft
	din = fft_in;
 else % ifft
	din = conj(fft_in);
 end

 fac8_0 = [1, 1, 1, -j];
 fac8_1 = [1, 1, 1, -j, 1, 0.7071-0.7071j, 1, -0.7071-0.7071j];

 %-----------------------------------------------------------------------------
 % Module 0
 %-----------------------------------------------------------------------------
 % step0_0
 bfly00_out0 = din(1:256) + din(257:512); % <4,6>
 bfly00_out1 = din(1:256) - din(257:512);

 bfly00_tmp = [bfly00_out0, bfly00_out1];

 for nn=1:512
	bfly00(nn) = bfly00_tmp(nn)*fac8_0(ceil(nn/128));
 end

 % step0_1
 for kk=1:2
  for nn=1:128
	bfly01_tmp((kk-1)*256+nn) = bfly00((kk-1)*256+nn) + bfly00((kk-1)*256+128+nn);
	bfly01_tmp((kk-1)*256+128+nn) = bfly00((kk-1)*256+nn) - bfly00((kk-1)*256+128+nn);
  end
 end


 for nn=1:512
	bfly01(nn) = bfly01_tmp(nn)*fac8_1(ceil(nn/64));
 end

 % step0_2
 for kk=1:4
  for nn=1:64
	bfly02_tmp((kk-1)*128+nn) = bfly01((kk-1)*128+nn) + bfly01((kk-1)*128+64+nn);
	bfly02_tmp((kk-1)*128+64+nn) = bfly01((kk-1)*128+nn) - bfly01((kk-1)*128+64+nn);
  end
 end

 % Data rearrangement
 K3 = [0, 4, 2, 6, 1, 5, 3, 7];

 for kk=1:8
  for nn=1:64
	twf_m0((kk-1)*64+nn) = exp(-j*2*pi*(nn-1)*(K3(kk))/512);
  end
 end

 for nn=1:512
	bfly02(nn) = bfly02_tmp(nn)*twf_m0(nn);
 end

 %-----------------------------------------------------------------------------
 % Module 1
 %-----------------------------------------------------------------------------
 % step1_0
 for kk=1:8
  for nn=1:32
	bfly10_tmp((kk-1)*64+nn) = bfly02((kk-1)*64+nn) + bfly02((kk-1)*64+32+nn);
	bfly10_tmp((kk-1)*64+32+nn) = bfly02((kk-1)*64+nn) - bfly02((kk-1)*64+32+nn);
  end
 end

 for kk=1:8
  for nn=1:64
	bfly10((kk-1)*64+nn) = bfly10_tmp((kk-1)*64+nn)*fac8_0(ceil(nn/16));
  end
 end

 % step1_1
 for kk=1:16
  for nn=1:16
	bfly11_tmp((kk-1)*32+nn) = bfly10((kk-1)*32+nn) + bfly10((kk-1)*32+16+nn);
	bfly11_tmp((kk-1)*32+16+nn) = bfly10((kk-1)*32+nn) - bfly10((kk-1)*32+16+nn);
  end
 end

 for kk=1:8
  for nn=1:64
	bfly11((kk-1)*64+nn) = bfly11_tmp((kk-1)*64+nn)*fac8_1(ceil(nn/8));
  end
 end

 % step1_2 (16)
 for kk=1:32
  for nn=1:8
	bfly12_tmp((kk-1)*16+nn) = bfly11((kk-1)*16+nn) + bfly11((kk-1)*16+8+nn);
	bfly12_tmp((kk-1)*16+8+nn) = bfly11((kk-1)*16+nn) - bfly11((kk-1)*16+8+nn);
  end
 end

 % Data rearrangement
 K2 = [0, 4, 2, 6, 1, 5, 3, 7];

 for kk=1:8
  for nn=1:8
	twf_m1((kk-1)*8+nn) = exp(-j*2*pi*(nn-1)*(K2(kk))/64);
  end
 end

 for kk=1:8
  for nn=1:64
	bfly12((kk-1)*64+nn) = bfly12_tmp((kk-1)*64+nn)*twf_m1(nn);
  end
 end

 %-----------------------------------------------------------------------------
 % Module 2
 %-----------------------------------------------------------------------------
 % step2_0
 for kk=1:64
  for nn=1:4
	bfly20_tmp((kk-1)*8+nn) = bfly12((kk-1)*8+nn) + bfly12((kk-1)*8+4+nn);
	bfly20_tmp((kk-1)*8+4+nn) = bfly12((kk-1)*8+nn) - bfly12((kk-1)*8+4+nn);
  end
 end

 for kk=1:64
  for nn=1:8
	bfly20((kk-1)*8+nn) = bfly20_tmp((kk-1)*8+nn)*fac8_0(ceil(nn/2));
  end
 end

 % step2_1
 for kk=1:128
  for nn=1:2
	bfly21_tmp((kk-1)*4+nn) = bfly20((kk-1)*4+nn) + bfly20((kk-1)*4+2+nn);
	bfly21_tmp((kk-1)*4+2+nn) = bfly20((kk-1)*4+nn) - bfly20((kk-1)*4+2+nn);
  end
 end

 for kk=1:64
  for nn=1:8
	bfly21((kk-1)*8+nn) = bfly21_tmp((kk-1)*8+nn)*fac8_1(nn);
  end
 end

 % step2_2
 for kk=1:256
	bfly22_tmp((kk-1)*2+1) = bfly21((kk-1)*2+1) + bfly21((kk-1)*2+2);
	bfly22_tmp((kk-1)*2+2) = bfly21((kk-1)*2+1) - bfly21((kk-1)*2+2);
 end

 bfly22 = bfly22_tmp;

 %-----------------------------------------------------------------------------
 % Index 
 %-----------------------------------------------------------------------------
 fp=fopen('reorder_index.txt','w');
 for jj=1:512
	%kk = bitget(jj-1,9)*(2^0) + bitget(jj-1,8)*(2^1) + bitget(jj-1,7)*(2^2) + bitget(jj-1,6)*(2^3) + bitget(jj-1,5)*(2^4) + bitget(jj-1,4)*(2^5) + bitget(jj-1,3)*(2^6) + bitget(jj-1,2)*(2^7) + bitget(jj-1,1)*(2^8);
	kk = bitget(jj-1,9)*1 + bitget(jj-1,8)*2 + bitget(jj-1,7)*4 + bitget(jj-1,6)*8 + bitget(jj-1,5)*16 + bitget(jj-1,4)*32 + bitget(jj-1,3)*64 + bitget(jj-1,2)*128 + bitget(jj-1,1)*256;
	dout(kk+1) = bfly22(jj); % With reorder
	fprintf(fp, 'jj=%d, kk=%d, dout(%d)=%f+j%f\n',jj, kk,(kk+1),real(dout(kk+1)),imag(dout(kk+1)));
 end
 fclose(fp);

 if (fft_mode==1) % fft
	fft_out = dout;
	module2_out = bfly22;
 else % ifft
	fft_out = conj(dout)/512; 
	module2_out = conj(bfly22)/512;

 end

end
```

## cos_in_gen.m
```matlab
% Added on 2024/01/12 by jihan 
function [data_float, data_fixed] = cos_in_gen(fft_mode, num)
 N = num;

 for i=1:N
	data_float_re(i) = cos(2.0*pi*(i-1)/N);
	data_float_im(i) = 0.0;
	data_float(i) = data_float_re(i) + j*data_float_im(i);
 end

 for i=1:N
  if (data_float_re(i)==1.0)
   if (fft_mode==1) % FFT
	%data_fixed_re(i) = 127; % <2.7>
	data_fixed_re(i) = 63; % <3.6> % Modified on 2025/07/02 by jihan
   else % IFFT
	data_fixed_re(i) = 255; % <1.8>
	%data_fixed_re(i) = 127; % <2.7> % Modified on 2025/07/02 by jihan
   end
  else	
   if (fft_mode==1) % FFT
	%data_fixed_re(i) = round(data_float_re(i)*128); % <2.7>
	data_fixed_re(i) = round(data_float_re(i)*64); % <3.6> % Modified on 2025/07/02 by jihan
   else % IFFT
	data_fixed_re(i) = round(data_float_re(i)*256); % <1.8>
	%data_fixed_re(i) = round(data_float_re(i)*128); % <2.7> % Modified on 2025/07/02 by jihan
   end
  end

  if (data_float_im(i)==1.0)
   if (fft_mode==1) % FFT
	%data_fixed_im(i) = 127; % <2.7>
	data_fixed_im(i) = 63; % <3.6> % Modified on 2025/07/02 by jihan
   else % IFFT
	data_fixed_im(i) = 255; % <1.8>
	%data_fixed_im(i) = 127; % <2.7> % Modified on 2025/07/02 by jihan
   end
  else	
   if (fft_mode==1) % FFT
	%data_fixed_im(i) = round(data_float_im(i)*128); % <2.7>
	data_fixed_im(i) = round(data_float_im(i)*64); % <3.6> % Modified on 2025/07/02 by jihan
   else % IFFT
	data_fixed_im(i) = round(data_float_im(i)*256); % <1.8>
	%data_fixed_im(i) = 127; % <2.7> % Modified on 2025/07/02 by jihan
   end
  end

	data_fixed(i) = data_fixed_re(i) + j*data_fixed_im(i);
 end

end
```

## 일단 확인

```matlab
% Test fft function (fft_float) with cosine input
fft_mode = 1; % 1: FFT mode
N = 512;

% Generate cosine input (floating point)
[cos_float, ~] = cos_in_gen(fft_mode, N);

% Compute FFT using fft_float function (floating point FFT)
[fft_out, ~] = fft_float(fft_mode, cos_float);

% Time domain plot (input signal)
figure;

subplot(2,2,1);
plot(0:N-1, real(cos_float));
title('Input Signal (Time domain) - Real part');
xlabel('Sample index');
ylabel('Amplitude');
grid on;

subplot(2,2,2);
plot(0:N-1, imag(cos_float));
title('Input Signal (Time domain) - Imag part');
xlabel('Sample index');
ylabel('Amplitude');
grid on;

% Frequency domain plot (FFT output)
freq_axis = (0:N-1)*(1/N); % Normalized frequency axis

subplot(2,2,3);
plot(freq_axis, real(fft_out));
title('FFT Output (Frequency domain) - Real part');
xlabel('Normalized Frequency');
ylabel('Amplitude');
grid on;

subplot(2,2,4);
plot(freq_axis, imag(fft_out));
title('FFT Output (Frequency domain) - Imag part');
xlabel('Normalized Frequency');
ylabel('Amplitude');
grid on;
```
![alt text](<../../../assets/img/SystemVerilog/FFT/스크린샷 2025-07-17 171436.png>)


## 고찰

### Step0 
| Step     | 구조            | 하는 일                             | twiddle factor (짝수만 적용됨)                                        | 이유             |
| -------- | ------------- | -------------------------------- | --------------------------------------------------------------- | -------------- |
| step0\_0 | 512 → 256×2   | 홀짝 분할 + radix-4 butterfly        | fac8\_0 = \[1, 1, 1, -j]                                        | N=4 포인트 기준 FFT |
| step0\_1 | 256×2 → 128×4 | 다시 4개 그룹으로 쪼개고 radix-8 구조 준비     | fac8\_1 = \[1, 1, 1, -j, 1, 0.7071-0.7071j, 1, -0.7071-0.7071j] | N=8 포인트 FFT 기준 |
| step0\_2 | 128×4 → 64×8  | 최종 radix-8 그룹으로 정리 + 상위 비트 기준 정렬 | K3 = \[0 4 2 6 1 5 3 7]                                         | MSB 정렬 (k3 비트) |


### Step1
| Step     | 설명                         | 입력 크기      | 출력 크기      | butterfly 크기                  | twiddle factor 길이 및 역할                     |
| -------- | -------------------------- | ---------- | ---------- | ----------------------------- | ------------------------------------------ |
| step1\_0 | 64점 단위로 쪼갬, 32점씩 butterfly | 512(8×64)  | 512(8×64)  | 64-point → 32-point butterfly | fac8\_0 (길이 4)로 16개씩 그룹화해 곱함               |
| step1\_1 | 32점 단위 butterfly, 16점씩 쪼갬  | 512(16×32) | 512(16×32) | 32-point → 16-point butterfly | fac8\_1 (길이 8)로 8개씩 그룹화해 곱함                |
| step1\_2 | 16점 단위 butterfly, 8점씩 쪼갬   | 512(32×16) | 512(32×16) | 16-point → 8-point butterfly  | twf\_m1 (길이 64)로 각 64점마다 twiddle factor 곱함 |


### Step2
| Step     | 설명                      | 입력 크기       | 출력 크기       | butterfly 크기                | twiddle factor 길이 및 역할       |
| -------- | ----------------------- | ----------- | ----------- | --------------------------- | ---------------------------- |
| step2\_0 | 8점 단위 butterfly, 4점씩 쪼갬 | 512 (64×8)  | 512 (64×8)  | 8-point → 4-point butterfly | fac8\_2 (길이 16)로 4개씩 그룹화해 곱함 |
| step2\_1 | 4점 단위 butterfly, 2점씩 쪼갬 | 512 (128×4) | 512 (128×4) | 4-point → 2-point butterfly | fac8\_3 (길이 32)로 2개씩 그룹화해 곱함 |
| step2\_2 | 2점 단위 butterfly, 1점씩 쪼갬 | 512 (256×2) | 512 (256×2) | 2-point → 1-point butterfly | fac8\_4 (길이 64)로 1개씩 그룹화해 곱함 |
