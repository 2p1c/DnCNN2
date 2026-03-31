function filtered_signal = bandpass_filter(signal_data, fs, lowcut, highcut, order)
    % BANDPASS_FILTER 简单的带通滤波器实现
    %
    % 功能: 
    %   使用Butterworth滤波器进行带通滤波
    %   不依赖信号处理工具箱
    %
    % 语法:
    %   filtered_signal = bandpass_filter(signal_data, fs, lowcut, highcut, order)
    %
    % 输入参数:
    %   signal_data - 输入信号 (行向量或列向量)
    %   fs - 采样频率 (Hz)
    %   lowcut - 低频截止频率 (Hz)
    %   highcut - 高频截止频率 (Hz)
    %   order - 滤波器阶数 (默认: 4)
    %
    % 输出参数:
    %   filtered_signal - 滤波后的信号
    %
    % 示例:
    %   fs = 1e6;  % 1 MHz采样率
    %   t = 0:1/fs:1e-3;  % 1ms信号
    %   signal = sin(2*pi*100e3*t) + sin(2*pi*200e3*t);
    %   filtered = bandpass_filter(signal, fs, 50e3, 300e3, 4);
    
    if nargin < 5
        order = 4;
    end
    
    % 确保信号是列向量
    if size(signal_data, 1) == 1
        signal_data = signal_data(:);
        transpose_output = true;
    else
        transpose_output = false;
    end
    
    % 验证输入参数
    if lowcut >= highcut
        error('bandpass_filter:invalid_freq', '低频截止必须小于高频截止频率');
    end
    
    if highcut >= fs/2
        error('bandpass_filter:freq_exceeds_nyquist', '高频截止频率超过奈奎斯特频率 (%.2f Hz)', fs/2);
    end
    
    if lowcut <= 0
        error('bandpass_filter:negative_freq', '频率必须为正');
    end
    
    try
        % 方法1: 尝试使用信号处理工具箱
        if license('test', 'Signal_Processing_Toolbox')
            fprintf('  使用信号处理工具箱\n');
            
            % 归一化频率
            Wn = [lowcut, highcut] / (fs/2);
            
            % 设计Butterworth滤波器
            [b, a] = butter(order, Wn, 'bandpass');
            
            % 应用滤波 (前后向滤波以消除相位失真)
            filtered_signal = filtfilt(b, a, signal_data);
            fprintf('  ✓ 使用filtfilt进行零相位滤波\n');
            
        else
            % 方法2: 使用FFT-based滤波（不需要工具箱）
            fprintf('  使用FFT-based方法进行滤波\n');
            
            % FFT变换
            signal_fft = fft(signal_data);
            freq_axis = (0:length(signal_data)-1) * fs / length(signal_data);
            
            % 创建频率掩模
            mask = (freq_axis >= lowcut & freq_axis <= highcut) | ...
                   (freq_axis >= fs - highcut & freq_axis <= fs - lowcut);
            
            % 应用掩模
            signal_fft_filtered = signal_fft .* mask(:);
            
            % 逆FFT
            filtered_signal = real(ifft(signal_fft_filtered));
            fprintf('  ✓ 使用FFT方法进行滤波\n');
        end
        
    catch ME
        % 方法3: 备选方案 - 简单的butterworth滤波器实现
        fprintf('  使用备选滤波方法\n');
        
        % 创建Butterworth滤波器系数（手动计算）
        Wn = [lowcut, highcut] / (fs/2);
        
        % 验证归一化频率范围
        if Wn(1) <= 0 || Wn(2) >= 1
            error('bandpass_filter:normalized_freq_out_of_range', ...
                '归一化频率必须在(0,1)范围内');
        end
        
        % 使用一个简单的IIR滤波器实现
        % 先设计低通滤波器系数
        [b, a] = butter_design(order, Wn);
        
        % 应用一阶滤波（简单但稳定）
        filtered_signal = signal_data;
        for i = 1:2  % 应用两次以提高滤波效果
            filtered_signal = filter(b, a, filtered_signal);
            filtered_signal = filter(b, a, filtered_signal(end:-1:1));
            filtered_signal = filtered_signal(end:-1:1);
        end
        
        fprintf('  ✓ 使用自定义滤波实现\n');
    end
    
    % 恢复原始形状
    if transpose_output
        filtered_signal = filtered_signal.';
    end
end

function [b, a] = butter_design(n, Wn)
    % 简单的Butterworth滤波器设计（备选方案）
    % 这个函数实现了基础的Butterworth滤波器设计
    
    % 对于带通滤波，使用标准的bilinear变换
    % 这是一个简化实现，可能不如工具箱精确
    
    if length(Wn) == 2
        % 带通滤波
        w0 = Wn(1);
        w1 = Wn(2);
        
        % 计算带宽和中心频率
        bw = w1 - w0;
        wc = sqrt(w0 * w1);
        
        % 使用原型低通滤波器
        [b0, a0] = butter_lowpass(n, 1);
        
        % 通过频率变换得到带通滤波器
        [b, a] = freq_transform(b0, a0, wc, bw);
        
    else
        % 低通滤波
        [b, a] = butter_lowpass(n, Wn);
    end
end

function [b, a] = butter_lowpass(n, Wc)
    % 简单的Butterworth低通滤波器设计
    
    % 计算模拟Butterworth极点
    poles = exp(1i * pi * (2*(1:n) + n + 1) / (2*n));
    
    % 缩放到指定截止频率
    poles = poles * Wc;
    
    % 转换为数字滤波器（使用bilinear变换）
    zeros = -ones(n, 1);
    
    % bilinear变换
    [b, a] = bilinear_transform(poles, zeros, 1);
end

function [b, a] = bilinear_transform(poles, zeros, k)
    % Bilinear变换 s -> (2/Ts)*(z-1)/(z+1)
    % 这里Ts=1（归一化）
    
    c = 2;  % 2/Ts
    
    z = (1 + poles/c) ./ (1 - poles/c);
    z = [z; -ones(length(zeros), 1)];
    p = (1 + zeros/c) ./ (1 - zeros/c);
    
    % 构造多项式
    b = poly(p);
    a = poly(z);
    
    % 归一化
    b = real(b * k);
    a = real(a);
end

function [b, a] = freq_transform(b0, a0, wc, bw)
    % 频率变换：低通到带通
    % 这是一个简化实现
    
    % 对于带通滤波，使用级联高通和低通
    [bh, ah] = butter_highpass_transform(length(a0)-1, wc - bw/2);
    [bl, al] = butter_lowpass_transform(length(a0)-1, wc + bw/2);
    
    % 级联两个滤波器
    b = conv(bh, bl);
    a = conv(ah, al);
    
    % 归一化
    a = a / a(1);
    b = b / a(1);
end

function [b, a] = butter_highpass_transform(n, Wc)
    % 高通滤波器频率变换
    [b, a] = butter_lowpass(n, Wc);
    
    % 转换为高通
    b = [1, zeros(1, length(b)-1)] .* b;
end

function [b, a] = butter_lowpass_transform(n, Wc)
    % 低通滤波器频率变换
    [b, a] = butter_lowpass(n, Wc);
end
