%% 单点时域信号分析程序
% 功能:
%   - 加载TXT格式的超声信号数据
%   - 带通滤波处理
%   - 时域波形显示
%   - 频谱分析（FFT）
%   - 可灵活调整滤波参数和坐标轴
%
% 使用方法:
%   1. 设置数据文件路径
%   2. 调整滤波参数（频段、阶数）
%   3. 调整显示参数（频谱限制、归一化等）
%   4. 运行脚本进行分析
%
% 作者: 信号处理团队
% 日期: 2024

clear; close all; clc;

%% ================== 用户配置区 ==================

% 数据文件配置
data_config = struct();
data_config.file_path = "C:\Users\123\Documents\Datasets\激光分布\光栅\2025.6.16 改变激光空间分布实验\Scan1000highpass__4.txt";  % 修改为你的数据文件路径
data_config.skip_rows = 0;           % 跳过行数（-1为自动检测）
data_config.delimiter = 'auto';      % 分隔符 ('auto', '\t', ' ', ',', ';')
data_config.time_col = 1;            % 时间列
data_config.signal_col = 2;          % 信号列

% 带通滤波参数
filter_config = struct();
filter_config.enable = true;         % 是否启用滤波
filter_config.lowcut = 190e3;         % 低频截止 (Hz) - 180 kHz
filter_config.highcut = 210e3;       % 高频截止 (Hz) - 220 kHz
filter_config.order = 4;             % 滤波器阶数
filter_config.method = 'butter';     % 滤波器类型 ('butter', 'cheby1', 'cheby2')

% 显示参数
display_config = struct();
display_config.freq_limit = 1e6;     % 频谱显示上限 (Hz) - 1 MHz
display_config.freq_start = 0;       % 频谱显示下限 (Hz)
display_config.db_limit = -60;       % dB显示下限 (dB)
display_config.normalize = true;     % 是否幅值归一化
display_config.time_unit = 'us';     % 时间单位 ('s', 'ms', 'us', 'ns')
display_config.freq_unit = 'MHz';    % 频率单位 ('Hz', 'kHz', 'MHz')
display_config.figure_width = 14;    % 图形宽度 (英寸)
display_config.figure_height = 8;    % 图形高度 (英寸)

% ================= 配置区结束 ================

%% 1. 加载数据
fprintf('\n========== 加载数据 ==========\n');

try
    % 首先尝试作为.mat文件加载
    if endsWith(data_config.file_path, '.mat')
        fprintf('从MAT文件加载: %s\n', data_config.file_path);
        mat_data = load(data_config.file_path);
        
        % 尝试找到信号和时间数据
        field_names = fieldnames(mat_data);
        
        if length(field_names) == 1
            data_struct = mat_data.(field_names{1});
            if isstruct(data_struct)
                signal_data = data_struct.signal(:);
                time_data = data_struct.time(:);
                if isfield(data_struct, 'sampling_rate')
                    fs = data_struct.sampling_rate;
                else
                    % 从时间数据推断采样率
                    fs = 1 / median(diff(time_data));
                end
            else
                error('无法自动提取数据，请检查MAT文件结构');
            end
        else
            error('MAT文件包含多个变量，请明确指定');
        end
    else
        % 使用txt_loader加载TXT文件
        fprintf('从TXT文件加载: %s\n', data_config.file_path);
        
        signal_obj = txt_loader.load_single(data_config.file_path, ...
            'skip_rows', data_config.skip_rows, ...
            'delimiter', data_config.delimiter, ...
            'time_col', data_config.time_col, ...
            'signal_col', data_config.signal_col, ...
            'remove_nan', true);
        
        signal_data = signal_obj.data;
        time_data = signal_obj.time;
        fs = signal_obj.fs;
    end
    
    % 验证数据
    assert(length(signal_data) == length(time_data), '时间和信号长度不匹配');
    
    fprintf('✓ 数据加载成功\n');
    fprintf('  数据点数: %d\n', length(signal_data));
    fprintf('  采样率: %.2f MHz\n', fs/1e6);
    fprintf('  时间范围: %.2f - %.2f μs\n', time_data(1)*1e6, time_data(end));
    fprintf('  时间跨度: %.2f μs\n', (time_data(end)-time_data(1)));
    fprintf('  信号范围: %.4f - %.4f V\n', min(signal_data), max(signal_data));
    
catch ME
    fprintf('✗ 数据加载失败: %s\n', ME.message);
    return;
end

%% 2. 预处理
fprintf('\n========== 数据预处理 ==========\n');

% 去除直流分量
signal_data = signal_data - mean(signal_data);

% 计算原始信号统计
rms_original = rms(signal_data);
fprintf('✓ 已去除直流分量\n');
fprintf('  信号RMS: %.6f V\n', rms_original);

%% 3. 带通滤波
signal_filtered = signal_data;  % 备份未滤波信号
rms_filtered = rms_original;

if filter_config.enable
    fprintf('\n========== 带通滤波 ==========\n');
    
    % 计算归一化频率
    nyq = 0.5 * fs;  % 奈奎斯特频率
    
    % 验证频率范围
    if filter_config.lowcut >= nyq || filter_config.highcut >= nyq
        warning('带通滤波频率超出奈奎斯特频率 (%.2f MHz)，已调整', nyq/1e6);
        filter_config.lowcut = min(filter_config.lowcut, nyq * 0.9);
        filter_config.highcut = min(filter_config.highcut, nyq * 0.99);
    end
    
    fprintf('设计滤波器\n');
    fprintf('  类型: 带通滤波\n');
    fprintf('  阶数: %d\n', filter_config.order);
    fprintf('  频段: %.2f - %.2f kHz\n', filter_config.lowcut/1e3, filter_config.highcut/1e3);
    
    % 应用滤波
    try
        signal_filtered = bandpass_filter(signal_data, fs, filter_config.lowcut, ...
            filter_config.highcut, filter_config.order);
        rms_filtered = rms(signal_filtered);
        
        fprintf('✓ 滤波成功\n');
        fprintf('  滤波后RMS: %.6f V\n', rms_filtered);
        fprintf('  幅值衰减: %.2f dB\n', 20*log10(rms_filtered/rms_original));
        
    catch ME
        warning('滤波失败: %s，使用未滤波信号', ME.message);
        signal_filtered = signal_data;
    end
else
    fprintf('\n滤波已禁用，使用原始信号\n');
end

%% 4. 频域分析
fprintf('\n========== 频域分析 ==========\n');

% FFT参数
nfft = 2^nextpow2(length(signal_filtered));
fft_result_filtered = fft(signal_filtered, nfft);
fft_result_original = fft(signal_data, nfft);
freqs = (0:nfft-1) * fs / nfft;

% 单边频谱
freq_idx = freqs <= fs/2;
freqs_single = freqs(freq_idx);
fft_single_filtered = 2 * abs(fft_result_filtered(freq_idx)) / length(signal_filtered);
fft_single_original = 2 * abs(fft_result_original(freq_idx)) / length(signal_data);

% 找到主频率
[~, peak_idx] = max(fft_single_filtered);
dominant_freq = freqs_single(peak_idx);

fprintf('✓ FFT分析完成\n');
fprintf('  FFT点数: %d\n', nfft);
fprintf('  频率分辨率: %.2f kHz\n', fs/nfft/1e3);
fprintf('  主频率: %.2f kHz\n', dominant_freq/1e3);
fprintf('  主频率幅值: %.4f V\n', fft_single_filtered(peak_idx));

%% 5. 绘制结果
fprintf('\n========== 绘制图形 ==========\n');

% 设置显示参数
time_scale = get_time_scale(display_config.time_unit);
freq_scale = get_freq_scale(display_config.freq_unit);

fig = figure('Position', [100, 100, display_config.figure_width*100, display_config.figure_height*100]);

% 子图1: 时域波形（原始信号）
ax1 = subplot(2, 3, 1);
plot(time_data * time_scale, signal_data, 'b-', 'LineWidth', 0.8);
grid on; grid minor;
xlabel(sprintf('时间 (%s)', display_config.time_unit), 'FontSize', 11);
ylabel('幅值 (V)', 'FontSize', 11);
title('原始信号时域波形', 'FontSize', 12, 'FontWeight', 'bold');
xlim([time_data(1)*time_scale, min(time_data(end)*time_scale, time_data(1)*time_scale + 100*time_scale)]);

% 子图2: 时域波形（滤波信号）
ax2 = subplot(2, 3, 2);
plot(time_data * time_scale, signal_filtered, 'r-', 'LineWidth', 0.8);
grid on; grid minor;
xlabel(sprintf('时间 (%s)', display_config.time_unit), 'FontSize', 11);
ylabel('幅值 (V)', 'FontSize', 11);
title('滤波后信号时域波形', 'FontSize', 12, 'FontWeight', 'bold');
if filter_config.enable
    title(sprintf('滤波后信号 (%.0f-%.0f kHz)', ...
        filter_config.lowcut/1e3, filter_config.highcut/1e3), ...
        'FontSize', 12, 'FontWeight', 'bold');
end
xlim([time_data(1)*time_scale, min(time_data(end)*time_scale, time_data(1)*time_scale + 100*time_scale)]);

% 子图3: 时域对比
ax3 = subplot(2, 3, 3);
hold on;
plot(time_data * time_scale, signal_data, 'b-', 'LineWidth', 0.5, 'DisplayName', '原始信号');
plot(time_data * time_scale, signal_filtered, 'r-', 'LineWidth', 0.8, 'DisplayName', '滤波信号');
grid on; grid minor;
xlabel(sprintf('时间 (%s)', display_config.time_unit), 'FontSize', 11);
ylabel('幅值 (V)', 'FontSize', 11);
title('时域信号对比', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
xlim([time_data(1)*time_scale, min(time_data(end)*time_scale, time_data(1)*time_scale + 100*time_scale)]);

% 子图4: 频谱（原始信号）
ax4 = subplot(2, 3, 4);
freq_show = freqs_single / freq_scale;
semilogy(freq_show, fft_single_original, 'b-', 'LineWidth', 0.8);
grid on; grid minor;
xlabel(sprintf('频率 (%s)', display_config.freq_unit), 'FontSize', 11);
ylabel('幅值 (V)', 'FontSize', 11);
title('原始信号频谱', 'FontSize', 12, 'FontWeight', 'bold');
xlim([display_config.freq_start/freq_scale, display_config.freq_limit/freq_scale]);

% 子图5: 频谱（滤波信号）
ax5 = subplot(2, 3, 5);
semilogy(freq_show, fft_single_filtered, 'r-', 'LineWidth', 0.8);
grid on; grid minor;
xlabel(sprintf('频率 (%s)', display_config.freq_unit), 'FontSize', 11);
ylabel('幅值 (V)', 'FontSize', 11);
title('滤波后信号频谱', 'FontSize', 12, 'FontWeight', 'bold');
xlim([display_config.freq_start/freq_scale, display_config.freq_limit/freq_scale]);

% 子图6: dB频谱
ax6 = subplot(2, 3, 6);
fft_db_filtered = 20 * log10(fft_single_filtered + 1e-10);
fft_db_original_plot = 20 * log10(fft_single_original + 1e-10);
semilogy(freq_show, 10.^(fft_db_filtered/10), 'r-', 'LineWidth', 0.8, 'DisplayName', '滤波信号');
hold on;
semilogy(freq_show, 10.^(fft_db_original_plot/10), 'b--', 'LineWidth', 0.6, 'DisplayName', '原始信号');
grid on; grid minor;
xlabel(sprintf('频率 (%s)', display_config.freq_unit), 'FontSize', 11);
ylabel('幅值 (dB ref 1V)', 'FontSize', 11);
title('频谱对比 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([display_config.freq_start/freq_scale, display_config.freq_limit/freq_scale]);
legend('Location', 'best', 'FontSize', 10);

sgtitle('单点时域信号分析', 'FontSize', 14, 'FontWeight', 'bold');
set(fig, 'Color', 'white');

fprintf('✓ 图形绘制完成\n\n');

%% 6. 输出统计信息
fprintf('========== 统计信息汇总 ==========\n');
fprintf('\n【时域特性】\n');
fprintf('  原始信号 RMS: %.6f V\n', rms_original);
fprintf('  滤波信号 RMS: %.6f V\n', rms_filtered);
fprintf('  峰值（原始）: %.6f V\n', max(abs(signal_data)));
fprintf('  峰值（滤波）: %.6f V\n', max(abs(signal_filtered)));
fprintf('  峰值因子（原始）: %.2f\n', max(abs(signal_data))/rms_original);
fprintf('  峰值因子（滤波）: %.2f\n', max(abs(signal_filtered))/rms_filtered);

fprintf('\n【频域特性】\n');
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  奈奎斯特频率: %.2f MHz\n', fs/2/1e6);
fprintf('  频率分辨率: %.2f kHz\n', fs/nfft/1e3);
fprintf('  主频率: %.2f kHz\n', dominant_freq/1e3);
fprintf('  主频率幅值: %.4f V\n', fft_single_filtered(peak_idx));

if filter_config.enable
    fprintf('\n【滤波参数】\n');
    fprintf('  滤波器类型: %s\n', filter_config.method);
    fprintf('  滤波器阶数: %d\n', filter_config.order);
    fprintf('  低频截止: %.2f kHz\n', filter_config.lowcut/1e3);
    fprintf('  高频截止: %.2f kHz\n', filter_config.highcut/1e3);
    fprintf('  幅值衰减: %.2f dB\n', 20*log10(rms_filtered/rms_original));
end

fprintf('\n========== 分析完成 ==========\n\n');

%% 辅助函数
function scale = get_time_scale(unit)
    % 获取时间缩放因子
    switch unit
        case 's'
            scale = 1;
        case 'ms'
            scale = 1e3;
        case 'us'
            scale = 1e6;
        case 'ns'
            scale = 1e9;
        otherwise
            scale = 1e6;  % 默认微秒
    end
end

function scale = get_freq_scale(unit)
    % 获取频率缩放因子
    switch unit
        case 'Hz'
            scale = 1;
        case 'kHz'
            scale = 1e3;
        case 'MHz'
            scale = 1e6;
        case 'GHz'
            scale = 1e9;
        otherwise
            scale = 1e6;  % 默认MHz
    end
end
