classdef Filter
    % BANDPASSFILTER 带通滤波器类
    % 提供针对超声信号的带通滤波功能
    
    methods (Static)
        function filtered_signal = apply(signal, fs, center_freq, bandwidth, filter_order)
            % APPLY 应用带通滤波器到信号
            %
            % 输入参数:
            %   signal       - 输入信号 (向量或矩阵)
            %   fs           - 采样频率 (Hz)
            %   center_freq  - 中心频率 (Hz), 默认 100e3 (100 kHz)
            %   bandwidth    - 带宽 (Hz), 默认 40e3 (±20 kHz)
            %   filter_order - 滤波器阶数, 默认 4
            %
            % 输出参数:
            %   filtered_signal - 滤波后的信号
            %
            % 示例:
            %   % 使用默认参数 (80-120 kHz)
            %   filtered = BandpassFilter.apply(signal, fs);
            %
            %   % 自定义参数
            %   filtered = BandpassFilter.apply(signal, fs, 100e3, 40e3, 6);
            
            % 参数验证
            if nargin < 2
                error('至少需要提供信号和采样频率');
            end
            
            % 设置默认参数
            if nargin < 3 || isempty(center_freq)
                center_freq = 100e3;  % 默认中心频率 100 kHz
            end
            
            if nargin < 4 || isempty(bandwidth)
                bandwidth = 40e3;  % 默认带宽 ±20 kHz
            end
            
            if nargin < 5 || isempty(filter_order)
                filter_order = 4;  % 默认4阶滤波器
            end
            
            % 计算截止频率
            lowcut = center_freq - bandwidth/2;   % 低频截止
            highcut = center_freq + bandwidth/2;  % 高频截止
            
            % 参数验证
            nyquist_freq = fs / 2;
            if lowcut <= 0
                error('低频截止频率必须大于0 Hz');
            end
            if highcut >= nyquist_freq
                error('高频截止频率 (%.1f kHz) 必须小于奈奎斯特频率 (%.1f kHz)', ...
                      highcut/1e3, nyquist_freq/1e3);
            end
            
            % 设计巴特沃斯带通滤波器
            [b, a] = butter(filter_order, [lowcut, highcut] / nyquist_freq, 'bandpass');
            
            % 应用零相位滤波 (前向-后向滤波,保持相位不变)
            if isvector(signal)
                % 如果是向量,直接滤波
                filtered_signal = filtfilt(b, a, signal);
            else
                % 如果是矩阵,沿第一个维度滤波
                filtered_signal = filtfilt(b, a, signal);
            end
        end
        
        function printInfo(center_freq, bandwidth, filter_order, fs)
            % PRINTINFO 显示滤波器参数信息
            %
            % 输入参数:
            %   center_freq  - 中心频率 (Hz)
            %   bandwidth    - 带宽 (Hz)
            %   filter_order - 滤波器阶数
            %   fs           - 采样频率 (Hz)
            
            lowcut = center_freq - bandwidth/2;
            highcut = center_freq + bandwidth/2;
            
            fprintf('带通滤波器参数:\n');
            fprintf('  中心频率: %.1f kHz\n', center_freq/1e3);
            fprintf('  通带范围: %.1f - %.1f kHz\n', lowcut/1e3, highcut/1e3);
            fprintf('  滤波器阶数: %d\n', filter_order);
            fprintf('  采样频率: %.2f MHz\n', fs/1e6);
        end
        
        function denoised_signal = waveletDenoise(signal, wavelet_name, level, threshold_method)
            % WAVELETDENOISE 小波去噪
            %
            % 输入参数:
            %   signal            - 输入信号 (向量)
            %   wavelet_name      - 小波基名称, 默认 'db4' (Daubechies 4)
            %   level             - 分解层数, 默认 5
            %   threshold_method  - 阈值方法, 默认 'soft' (软阈值)
            %                       可选: 'soft' (软阈值) 或 'hard' (硬阈值)
            %
            % 输出参数:
            %   denoised_signal - 去噪后的信号
            %
            % 原理:
            %   1. 小波分解: 将信号分解为不同频率尺度的小波系数
            %   2. 阈值处理: 对高频系数应用阈值,抑制噪声
            %   3. 小波重构: 从处理后的系数重构信号
            %
            % 示例:
            %   % 使用默认参数 (db4小波, 5层分解, 软阈值)
            %   denoised = Filter.waveletDenoise(signal);
            %
            %   % 自定义参数
            %   denoised = Filter.waveletDenoise(signal, 'sym4', 6, 'soft');
            
            % 参数验证
            if nargin < 1
                error('至少需要提供信号');
            end
            
            % 设置默认参数
            if nargin < 2 || isempty(wavelet_name)
                wavelet_name = 'db4';  % Daubechies 4小波,适合超声信号
            end
            
            if nargin < 3 || isempty(level)
                level = 5;  % 默认5层分解
            end
            
            if nargin < 4 || isempty(threshold_method)
                threshold_method = 'soft';  % 软阈值,更平滑
            end
            
            % 确保信号是列向量
            signal = signal(:);
            
            % 步骤1: 小波分解
            % 使用离散小波变换(DWT)将信号分解为多个尺度
            [C, L] = wavedec(signal, level, wavelet_name);
            
            % 步骤2: 计算阈值
            % 使用通用阈值估计方法 (Donoho-Johnstone)
            % sigma = median(|cD1|) / 0.6745, 其中cD1是第一层细节系数
            cD1 = detcoef(C, L, 1);  % 提取第一层细节系数
            sigma = median(abs(cD1)) / 0.6745;  % 噪声标准差估计
            
            % 通用阈值: thr = sigma * sqrt(2 * log(N))
            N = length(signal);
            thr = sigma * sqrt(2 * log(N));
            
            % 步骤3: 阈值处理
            % 对每一层的细节系数应用阈值
            if strcmp(threshold_method, 'soft')
                % 软阈值: 系数向零收缩
                C_denoised = wthcoef('d', C, L, 1:level, thr, 'soft');
            else
                % 硬阈值: 小于阈值的系数置零
                C_denoised = wthcoef('d', C, L, 1:level, thr, 'hard');
            end
            
            % 步骤4: 小波重构
            % 从处理后的系数重构信号
            denoised_signal = waverec(C_denoised, L, wavelet_name);
            
            % 计算去噪效果指标
            noise_removed = signal - denoised_signal;
            snr_improvement = 10 * log10(var(signal) / var(noise_removed));
            
            % 输出去噪信息(可选,调试用)
            % fprintf('  小波去噪完成:\n');
            % fprintf('    小波基: %s\n', wavelet_name);
            % fprintf('    分解层数: %d\n', level);
            % fprintf('    阈值方法: %s\n', threshold_method);
            % fprintf('    阈值: %.6f\n', thr);
            % fprintf('    SNR改善: %.2f dB\n', snr_improvement);
        end
        
        function printWaveletInfo(wavelet_name, level, threshold_method)
            % PRINTWAVELETINFO 显示小波去噪参数信息
            %
            % 输入参数:
            %   wavelet_name      - 小波基名称
            %   level             - 分解层数
            %   threshold_method  - 阈值方法
            
            fprintf('小波去噪参数:\n');
            fprintf('  小波基: %s\n', wavelet_name);
            fprintf('  分解层数: %d\n', level);
            fprintf('  阈值方法: %s (软阈值)\n', threshold_method);
            
            % 小波基说明
            switch wavelet_name
                case 'db4'
                    fprintf('  说明: Daubechies 4小波,良好的时频局部化特性,适合超声导波\n');
                case 'sym4'
                    fprintf('  说明: Symlet 4小波,近似对称,相位失真小\n');
                case 'coif3'
                    fprintf('  说明: Coiflet 3小波,尺度函数和小波函数都有较好的正则性\n');
                otherwise
                    fprintf('  说明: 自定义小波基\n');
            end
        end
    end
end
