%% COMSOL数据频散曲线分析
% 对COMSOL超声模拟数据进行频散曲线计算
% 包括: 波数-频率曲线 和 群速度-频率曲线

clear; close all; clc;

%% 1. 加载数据
fprintf('加载COMSOL处理后的数据...\n');
load('processed_data.mat');  % 包含: x, y, data_points, time_ns

% 数据信息
[n_points, n_time_steps] = size(y);
fprintf('数据点数: %d\n', n_points);
fprintf('时间步数: %d\n', n_time_steps);
fprintf('时间范围: %.3f - %.3f μs\n', x(1)*1e6, x(end)*1e6);

% 提取空间坐标
x_coords = data_points(:, 1);  % X坐标 (mm)
y_coords = data_points(:, 2);  % Y坐标 (mm)

%% 2. 确定分析方向和数据组织
% 分析X方向的波传播（一行数据）
% 选择X方向17.5mm到22.5mm的11个点
x_min = -22.5;  % mm
x_max = -17.5;  % mm
% x_min = 17.5;  % mm
% x_max = 22.5;  % mm
n_points_target = 11;

fprintf('\n选择X方向 %.1f - %.1f mm 的 %d 个点\n', x_min, x_max, n_points_target);

% 检查是否有多行数据
unique_y_coords = unique(y_coords);
fprintf('Y方向坐标数: %d\n', length(unique_y_coords));

if length(unique_y_coords) == 1
    % 只有一行数据
    fprintf('检测到单行数据\n');
    row_data = y;
    row_x_coords = x_coords;
else
    % 多行数据，选择中间行
    middle_y = median(unique_y_coords);
    row_indices = find(abs(y_coords - middle_y) < 1e-6);
    fprintf('选择Y=%.2f mm的行\n', middle_y);
    
    row_data = y(row_indices, :);
    row_x_coords = x_coords(row_indices);
end

% 筛选X方向17.5-22.5mm范围内的点
x_range_idx = (row_x_coords >= x_min) & (row_x_coords <= x_max);
selected_x_coords = row_x_coords(x_range_idx);
selected_row_data = row_data(x_range_idx, :);

fprintf('范围内找到 %d 个点\n', length(selected_x_coords));

% 如果找不到点，报告实际范围并使用所有点
if isempty(selected_x_coords)
    fprintf('警告: 指定范围 %.1f-%.1f mm 内没有数据点\n', x_min, x_max);
    fprintf('实际X坐标范围: %.2f - %.2f mm\n', min(row_x_coords), max(row_x_coords));
    fprintf('使用所有可用点进行分析\n\n');
    
    selected_x_coords = row_x_coords;
    selected_row_data = row_data;
end

% 按X坐标排序
[spatial_coords, sort_idx] = sort(selected_x_coords);
selected_row_data = selected_row_data(sort_idx, :);

% 如果点数不等于11，进行插值或使用现有点
if length(spatial_coords) >= 2 && length(spatial_coords) ~= n_points_target
    fprintf('点数: %d，进行插值到 %d 个点\n', length(spatial_coords), n_points_target);
    
    % 生成均匀的11个点
    x_start = spatial_coords(1);
    x_end = spatial_coords(end);
    new_x_coords = linspace(x_start, x_end, n_points_target);
    new_data = zeros(n_points_target, size(selected_row_data, 2));
    
    % 对每个时间点进行空间插值
    for t_idx = 1:size(selected_row_data, 2)
        new_data(:, t_idx) = interp1(spatial_coords, selected_row_data(:, t_idx), new_x_coords, 'linear');
    end
    
    spatial_coords = new_x_coords(:);
    selected_row_data = new_data;
elseif length(spatial_coords) < 2
    error('数据点不足，至少需要2个点进行频散分析');
end

[n_spatial_points, ~] = size(selected_row_data);
fprintf('用于频散分析的数据: %d × %d\n', n_spatial_points, n_time_steps);

%% 3. 计算采样参数
dt = x(501) - x(500);  % 时间采样间隔 (s)
% fs = 10e6;  % 采样频率 (Hz)
% fs = 6.25e6;  % 采样频率 (Hz)
fs = 1/dt;  % 采样频率 (Hz)
dx = (spatial_coords(end) - spatial_coords(1)) / (n_spatial_points - 1);  % 空间采样间隔 (mm)
dx_m = dx * 1e-3;  % 转换为米

fprintf('\n采样参数:\n');
fprintf('  时间采样间隔: %.3e s (%.3f ns)\n', dt, dt*1e9);
fprintf('  采样频率: %.2f MHz\n', fs/1e6);
fprintf('  空间采样间隔: %.3f mm\n', dx);
fprintf('  空间范围: %.2f - %.2f mm\n', spatial_coords(1), spatial_coords(end));

%% 4. 数据预处理：去直流和带通滤波
fprintf('\n数据预处理...\n');
% 去直流分量
data_filtered = selected_row_data - mean(selected_row_data, 2);

% 应用100kHz-700kHz带通滤波
center_freq = 400e3;  % 中心频率 400 kHz
bandwidth = 600e3;    % 带宽 600 kHz (100-700 kHz)
filter_order = 4;     % 滤波器阶数

fprintf('  应用带通滤波: %.0f - %.0f kHz\n', ...
    (center_freq - bandwidth/2)/1e3, (center_freq + bandwidth/2)/1e3);
Filter.printInfo(center_freq, bandwidth, filter_order, fs);

% 对每个空间点应用滤波
for i = 1:n_spatial_points
    data_filtered(i, :) = Filter.apply(data_filtered(i, :), fs, center_freq, bandwidth, filter_order);
end
fprintf('  带通滤波完成\n');

% 应用小波去噪
wavelet_name = 'db4';      % Daubechies 4小波
wavelet_level = 2;         % 分解层数
threshold_method = 'soft'; % 软阈值

fprintf('  应用小波去噪并补偿高频能量...\n');
Filter.printWaveletInfo(wavelet_name, wavelet_level, threshold_method);

for i = 1:n_spatial_points
    % 小波去噪
    signal_denoised = Filter.waveletDenoise(data_filtered(i, :), wavelet_name, wavelet_level, threshold_method);
    
    % 高频能量补偿（补偿小波变换导致的高频衰减）
    nfft_comp = length(signal_denoised);
    signal_fft = fft(signal_denoised);
    freq_comp = (0:nfft_comp-1) * fs / nfft_comp;
    
    % 设计高频补偿滤波器：线性递增增益
    % 300kHz以下: 增益为1.0
    % 300kHz-700kHz: 线性增加到1.5倍
    freq_threshold = 300e3;  % 300 kHz开始补偿
    freq_max = 700e3;        % 700 kHz最大补偿
    
    gain = ones(1, nfft_comp);
    % 正频率部分
    idx_comp = (freq_comp >= freq_threshold) & (freq_comp <= freq_max);
    gain(idx_comp) = 1.0 + 0.5 * (freq_comp(idx_comp) - freq_threshold) / (freq_max - freq_threshold);
    idx_high = freq_comp > freq_max;
    gain(idx_high) = 1.5;
    
    % 负频率部分镜像处理
    if mod(nfft_comp, 2) == 0
        gain(nfft_comp/2+2:end) = fliplr(gain(2:nfft_comp/2));
    else
        gain((nfft_comp+3)/2:end) = fliplr(gain(2:(nfft_comp+1)/2));
    end
    
    % 确保gain和signal_fft形状匹配
    if size(signal_fft, 1) > 1
        gain = gain(:);  % 转为列向量
    end
    
    % 应用补偿
    signal_fft_comp = signal_fft .* gain;
    data_filtered(i, :) = real(ifft(signal_fft_comp));
end
fprintf('  小波去噪和高频补偿完成\n');

%% 5. 二维傅里叶变换 (2D FFT)
fprintf('执行二维傅里叶变换...\n');

% 使用零填充提高分辨率
% 大幅增加空间维度的零填充以显著提高波数分辨率
nfft_space = 2^(nextpow2(n_spatial_points) + 6);  % 空间维度FFT点数（大幅增加零填充）
nfft_time = 2^(nextpow2(n_time_steps) + 2);       % 时间维度FFT点数

fprintf('  FFT点数: 空间=%d, 时间=%d\n', nfft_space, nfft_time);

% 二维FFT
fft2_result = fft2(data_filtered, nfft_space, nfft_time);
fft2_shifted = fftshift(fft2_result, 1);  % 只对空间维度shift

% 计算幅值谱
amplitude_spectrum = abs(fft2_shifted);

%% 6. 生成频率和波数向量
% 频率向量 (Hz)
freq_vector = (0:nfft_time-1) * fs / nfft_time;

% 波数向量 (rad/m)
dk = 2*pi / (nfft_space * dx_m);
k_vector = ((-nfft_space/2):(nfft_space/2-1)) * dk;

% 只保留正频率
positive_freq_idx = freq_vector <= fs/2;
freq_positive = freq_vector(positive_freq_idx);
amplitude_positive = amplitude_spectrum(:, positive_freq_idx);

fprintf('  频率范围: 0 - %.2f MHz\n', max(freq_positive)/1e6);
fprintf('  波数范围: %.2f - %.2f rad/m\n', min(k_vector), max(k_vector));

%% 6.5 频散曲线增强处理
fprintf('\n频散曲线增强处理:\n');

% 基础归一化：按频率逐列归一化
amp_base = amplitude_positive;
max_base = max(amp_base, [], 1);
max_base(max_base == 0) = 1;
amp_base = amp_base ./ max_base;
fprintf('  基础方法: 按频率归一化\n');

% 对数尺度 + 自适应阈值（组合方法）⭐推荐
% 原理: 先用对数压缩，再用阈值去噪
amp_temp = 20*log10(amplitude_positive + eps);
amp_temp = amp_temp - min(amp_temp(:));
amp_enhanced = zeros(size(amp_temp));
threshold_factor_combined = 1.0;  % 组合方法的阈值因子
for i = 1:size(amp_temp, 2)
    col = amp_temp(:, i);
    threshold = median(col) + threshold_factor_combined * std(col);
    col(col < threshold) = 0;
    amp_enhanced(:, i) = col;
end
% 归一化
max_enhanced = max(amp_enhanced, [], 1);
max_enhanced(max_enhanced == 0) = 1;
amp_enhanced = amp_enhanced ./ max_enhanced;
fprintf('  增强方法: 对数+阈值组合 (阈值因子=%.1f) ⭐\n', threshold_factor_combined);

%% 7. 提取频散曲线
fprintf('\n提取频散曲线...\n');

%=========================================================================%
% 【核心算法1】频散曲线提取 - 从k-f域谱图中提取频散关系
% 原理: 对于每个频率f，找到能量最大的波数k，即频散曲线 k(f)
%=========================================================================%

% 分离正向波和反向波
k_positive_idx = k_vector >= 0;  % 正向传播 (+X方向)
k_negative_idx = k_vector < 0;   % 反向传播 (-X方向)

% 对于每个频率，分别找正向和反向波的最大幅值波数
dispersion_k_forward = zeros(size(freq_positive));   % 正向波
dispersion_k_backward = zeros(size(freq_positive));  % 反向波
amplitude_forward = zeros(size(freq_positive));
amplitude_backward = zeros(size(freq_positive));

for i = 1:length(freq_positive)
    % 正向波 (k >= 0)
    [amp_f, idx_f] = max(amplitude_positive(k_positive_idx, i));
    k_positive_vals = k_vector(k_positive_idx);
    dispersion_k_forward(i) = k_positive_vals(idx_f);
    amplitude_forward(i) = amp_f;
    
    % 反向波 (k < 0)
    [amp_b, idx_b] = max(amplitude_positive(k_negative_idx, i));
    k_negative_vals = k_vector(k_negative_idx);
    dispersion_k_backward(i) = k_negative_vals(idx_b);
    amplitude_backward(i) = amp_b;
end

% 判断主要传播方向并确保主要波的波数为正
if sum(amplitude_forward) > sum(amplitude_backward)
    fprintf('  检测到主要传播方向: +X方向（正向波）\n');
    main_wave_is_forward = true;
else
    fprintf('  检测到主要传播方向: -X方向（反向波）\n');
    fprintf('  将反向波的波数取绝对值，使其为正\n');
    main_wave_is_forward = false;
end

% 处理主要波数据（始终使用正波数）
if main_wave_is_forward
    % 主要波是正向波，直接使用
    amplitude_threshold = 0.1 * max(amplitude_forward);
    valid_idx = amplitude_forward > amplitude_threshold & freq_positive > 0;
    freq_valid = freq_positive(valid_idx);
    k_valid = dispersion_k_forward(valid_idx);
else
    % 主要波是反向波，取绝对值使波数为正
    amplitude_threshold = 0.1 * max(amplitude_backward);
    valid_idx = amplitude_backward > amplitude_threshold & freq_positive > 0;
    freq_valid = freq_positive(valid_idx);
    k_valid = abs(dispersion_k_backward(valid_idx));  % 取绝对值
end

fprintf('  主要波有效频率点数: %d / %d\n', sum(valid_idx), length(valid_idx));

%% 7.5 频散曲线插值（提高平滑度）
fprintf('\n对频散曲线进行插值...\n');

% 设置插值后的频率点数（增加密度）
n_interp = 1000;  % 插值后的点数
freq_interp = linspace(min(freq_valid), max(freq_valid), n_interp);

% 使用PCHIP（分段三次Hermite插值）或spline插值
% PCHIP保持单调性，避免过冲，更适合物理数据
k_interp = interp1(freq_valid, k_valid, freq_interp, 'pchip');

fprintf('  插值方法: PCHIP (分段三次Hermite插值)\n');
fprintf('  原始点数: %d -> 插值点数: %d\n', length(freq_valid), n_interp);

% 保存原始数据用于对比
freq_valid_original = freq_valid;
k_valid_original = k_valid;

% 使用插值后的数据
freq_valid = freq_interp(:);
k_valid = k_interp(:);

%% 8. 计算相速度和群速度

%=========================================================================%
% 【核心算法2】相速度计算
% 公式: vp = ω/k = 2πf/k
% 物理意义: 等相位面的传播速度
%=========================================================================%

% 主要波相速度
omega_valid = 2*pi * freq_valid;
vp_valid = omega_valid ./ k_valid;  % m/s

%=========================================================================%
% 【核心算法3】群速度计算
% 公式: vg = dω/dk = d(2πf)/dk
% 物理意义: 能量包络的传播速度
% 方法: 使用中心差分进行数值微分
%=========================================================================%

% 主要波群速度
vg_valid = zeros(size(freq_valid));
for i = 2:length(freq_valid)-1
    df = freq_valid(i+1) - freq_valid(i-1);
    dk_diff = k_valid(i+1) - k_valid(i-1);
    if abs(dk_diff) > 1e-10
        vg_valid(i) = 2*pi * df / dk_diff;  % 中心差分
    else
        vg_valid(i) = NaN;
    end
end
% 边界点使用单侧差分
if length(freq_valid) > 1
    vg_valid(1) = 2*pi * (freq_valid(2) - freq_valid(1)) / ...
                          (k_valid(2) - k_valid(1));
    vg_valid(end) = 2*pi * (freq_valid(end) - freq_valid(end-1)) / ...
                            (k_valid(end) - k_valid(end-1));
end

% 去除异常值
vg_valid(abs(vg_valid) > 10000) = NaN;  % 大于10 km/s视为异常
vg_valid(abs(vg_valid) < 100) = NaN;    % 小于100 m/s视为异常

fprintf('  主要波相速度范围: %.0f - %.0f m/s\n', min(vp_valid), max(vp_valid));
fprintf('  主要波群速度范围: %.0f - %.0f m/s\n', ...
    min(vg_valid(~isnan(vg_valid))), max(vg_valid(~isnan(vg_valid))));

%% 9. 可视化结果
fprintf('\n生成可视化图表...\n');

% 图1: 频散曲线增强方法对比
figure('Name', '频散曲线对比', 'Position', [30, 30, 1400, 600]);

% 基础归一化
subplot(1, 2, 1);
pcolor(freq_positive/1e6, k_vector, amp_base);
shading interp;
colormap(jet);
colorbar;
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('波数 (rad/m)', 'FontSize', 11);
title('基础: 按频率归一化', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, 1]);
ylim([min(k_vector), max(k_vector)]);
grid on;

% 对数+阈值组合（增强）
subplot(1, 2, 2);
pcolor(freq_positive/1e6, k_vector, amp_enhanced);
shading interp;
colormap(jet);
colorbar;
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('波数 (rad/m)', 'FontSize', 11);
title('增强: 对数+阈值组合 ⭐', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
xlim([0, 1]);
ylim([min(k_vector), max(k_vector)]);
grid on;

sgtitle('频散曲线对比 - COMSOL数据', 'FontSize', 14, 'FontWeight', 'bold');

% 图2: 使用增强后的频谱进行频散分析
figure('Name', '频散分析结果（增强版）', 'Position', [50, 50, 1600, 500]);

subplot(1, 3, 1);
pcolor(freq_positive/1e6, k_vector, amp_enhanced);  % 使用增强后的数据
shading interp;
colormap(jet);
colorbar;
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('波数 (rad/m)', 'FontSize', 11);
title('k-f域频谱（增强版）', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, 1]);  % 0-1 MHz
ylim([min(k_vector), max(k_vector)]);
grid on;

% 图2: 波数-频率频散曲线
subplot(1, 3, 2);
scatter(freq_valid/1e6, k_valid, 30, 'b', 'filled');
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('波数 (rad/m)', 'FontSize', 11);
title('波数-频率频散曲线', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 1]);  % 0-1 MHz

% 图3: 相速度-频率曲线
subplot(1, 3, 3);
scatter(freq_valid/1e6, vp_valid, 30, 'r', 'filled');
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('相速度 (m/s)', 'FontSize', 11);
title('相速度-频率曲线', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 1]);  % 0-1 MHz

sgtitle('COMSOL超声数据频散曲线分析（增强版）', 'FontSize', 15, 'FontWeight', 'bold');

fprintf('\n✓ 频散曲线分析完成！\n');
fprintf('  已生成两个对比图表:\n');
fprintf('    图1: 频散曲线对比（基础归一化 vs 对数+阈值组合）\n');
fprintf('    图2: 频散分析结果（使用增强后的数据）\n');
fprintf('  波数分辨率已提高（空间FFT零填充增加）\n');
