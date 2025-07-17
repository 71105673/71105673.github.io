---
title: "project FFT" 
date: "2025-07-17"
thumbnail: "../../../assets/img/SystemVerilog/image.png"
---

# 일단 확인

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