%% ============================================================
%  ANTIGRAVITY WAVENUMBER FILTERING FOR DEFECT DETECTION
%  波数域定向滤波缺陷检测系统
%  ============================================================
%  策略: 利用波数域分离正向波/反射波/散射波，增强缺陷可见性
%  - 反射波检测: 缺陷反射产生负k_y分量
%  - 散射波检测: 缺陷散射产生高波数分量
%  - 时域RMS成像: 滤波后反变换，RMS可视化
%  ============================================================

clear; clc; close all;

%% ========== MISSION PARAMETERS (配置区) ========== %%

% 数据源配置
data_file = "E:\数据\260127\kongou200k2\51_51.mat";
n_points = [51, 51];     % 点阵大小: [n_x, n_y]
spacing = 1e-3;          % 空间采样间隔 (m)

% 滤波模式选择
% 'forward'     - 仅保留正向传播波 (入射波)
% 'backward'    - 仅保留反向传播波 (反射波/缺陷信号)
% 'scattered'   - 仅保留散射波 (高波数分量)
% 'total'       - 提取反射波 + 散射波 (缺陷增强)
% 'lowpass'     - 低通滤波 (去噪)
filter_mode = 'scattered';

% 波传播方向 (根据实际设置)
% 'positive_y' - 波沿+Y方向传播 (源在Y=0)
% 'negative_y' - 波沿-Y方向传播 (源在Y=max)
% 'positive_x' - 波沿+X方向传播
propagation_direction = 'negative_y';

% ===========260128-v1.1.0效果较好===========
% enable_attenuation_compensation = true;  % 是否启用衰减补偿
% attenuation_coefficient = 0.5;          % 衰减系数 (1/mm)
% source_position = [25, 65] * 1e-3;       % 波源位置 [x, y] (米)
% min_compensation_distance = 10e-3;       % 最小补偿距离 (米)

% % 滤波器参数
% k_lowpass_ratio = 0.65;         % 低通截止 (占最大波数的比例)
% k_highpass_ratio = 0.3;        % 高通截止 (散射波提取用)
% taper_width = 0.15;            % 边缘过渡宽度

% % 异常波数滤除参数
% enable_outlier_removal = true;       % 是否启用异常值滤除
% outlier_threshold = 0.5;             % 异常阈值 (mean + N*std)
% outlier_replace_method = 'zero';     % 替换方法: 'zero', 'median', 'interpolate'

% % 时间窗口 (可选，减少计算量)
% enable_time_window = false;
% time_start_us = 0;             % 起始时间 (μs)
% time_end_us = 100;             % 结束时间 (μs)

% % ===========260128-v1.1.1效果较好，优化衰减补偿===========
% enable_attenuation_compensation = true;  % 是否启用衰减补偿
% attenuation_coefficient = 0.5;          % 衰减系数 (1/mm)
% source_position = [25, 100] * 1e-3;       % 波源位置 [x, y] (米)
% min_compensation_distance = 45e-3;       % 最小补偿距离 (米)

% % 滤波器参数
% k_lowpass_ratio = 0.65;         % 低通截止 (占最大波数的比例)
% k_highpass_ratio = 0.3;        % 高通截止 (散射波提取用)
% taper_width = 0.15;            % 边缘过渡宽度

% % 异常波数滤除参数
% enable_outlier_removal = true;       % 是否启用异常值滤除
% outlier_threshold = 0.5;             % 异常阈值 (mean + N*std)
% outlier_replace_method = 'zero';     % 替换方法: 'zero', 'median', 'interpolate'

% % 时间窗口 (可选，减少计算量)
% enable_time_window = false;
% time_start_us = 0;             % 起始时间 (μs)
% time_end_us = 100;             % 结束时间 (μs)

% ===========260128-v1.1.1效果较好===========
enable_attenuation_compensation = true;  % 是否启用衰减补偿
attenuation_coefficient = 0.1;          % 衰减系数 (1/mm)
source_position = [25, 100] * 1e-3;       % 波源位置 [x, y] (米)
min_compensation_distance = 45e-3;       % 最小补偿距离 (米)

% 滤波器参数
k_lowpass_ratio = 0.5;         % 低通截止 (占最大波数的比例)
k_highpass_ratio = 0.3;        % 高通截止 (散射波提取用)
taper_width = 0.15;            % 边缘过渡宽度

% 异常波数滤除参数
enable_outlier_removal = true;       % 是否启用异常值滤除
outlier_threshold = 0.5;             % 异常阈值 (mean + N*std)
outlier_replace_method = 'zero';     % 替换方法: 'zero', 'median', 'interpolate'

% 时间窗口 (可选，减少计算量)
enable_time_window = false;
time_start_us = 0;             % 起始时间 (μs)
time_end_us = 100;             % 结束时间 (μs)

% ========== 斜线衰减补偿参数 ==========
% enable_attenuation_compensation = true;  % 是否启用衰减补偿
% attenuation_coefficient = 4;          % 衰减系数 (1/mm)
% source_position = [40, 40] * 1e-3;       % 波源位置 [x, y] (米)
% min_compensation_distance = 10e-3;       % 最小补偿距离 (米)

% % 滤波器参数
% k_lowpass_ratio = 0.5;         % 低通截止 (占最大波数的比例)
% k_highpass_ratio = 0.1;        % 高通截止 (散射波提取用)
% taper_width = 0.15;            % 边缘过渡宽度

% % 异常波数滤除参数
% enable_outlier_removal = true;       % 是否启用异常值滤除
% outlier_threshold = 0.4;             % 异常阈值 (mean + N*std)
% outlier_replace_method = 'zero';     % 替换方法: 'zero', 'median', 'interpolate'

% % 时间窗口 (可选，减少计算量)
% enable_time_window = false;
% time_start_us = 0;             % 起始时间 (μs)
% time_end_us = 100;             % 结束时间 (μs)

% 可视化
colormap_style = 'jet';
interp_factor = 4;             % RMS图像插值倍数

% RMS异常值处理参数
enable_rms_outlier_removal = true;   % 是否启用RMS异常值滤除
rms_outlier_threshold = 5.0;         % 异常阈值 (mean + N*std)
edge_margin = 2;                     % 边缘忽略宽度

%% ========== PHASE 1: DATA LOADING ========== %%

fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   WAVENUMBER FILTERING FOR DEFECT DETECTION               ║\n');
fprintf('║   波数域定向滤波缺陷检测系统 v3.0                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

fprintf('▶ PHASE 1: 数据加载...\n');

% 解析点阵尺寸
if isscalar(n_points)
    n_x = n_points; n_y = n_points;
else
    n_x = n_points(1); n_y = n_points(2);
end

% 加载数据
data_struct = load(data_file, 'x', 'y');
time_vector = data_struct.x;
raw_data = data_struct.y;
n_time = length(time_vector);
fs = 1 / (time_vector(2) - time_vector(1));

% 重塑为3D波场 [Y, X, Time]
wavefield_3d = zeros(n_y, n_x, n_time);
for col = 1:n_x
    start_idx = (col-1) * n_y + 1;
    end_idx = col * n_y;
    wavefield_3d(:, col, :) = raw_data(start_idx:end_idx, :);
end

% 时间窗口裁剪
if enable_time_window
    time_us = time_vector * 1e6;
    t_start_idx = find(time_us >= time_start_us, 1, 'first');
    t_end_idx = find(time_us <= time_end_us, 1, 'last');
    wavefield_3d = wavefield_3d(:, :, t_start_idx:t_end_idx);
    time_vector = time_vector(t_start_idx:t_end_idx);
    n_time = size(wavefield_3d, 3);
    fprintf('  ├─ 时间窗口: %.1f - %.1f μs (%d samples)\n', time_start_us, time_end_us, n_time);
end

fprintf('  ├─ 波场尺寸: %d × %d × %d (Y × X × T)\n', n_y, n_x, n_time);
fprintf('  ├─ 采样率: %.2f MHz\n', fs/1e6);
fprintf('  └─ ✓ 数据加载完成\n\n');

% 空间坐标
dx = spacing; dy = spacing;
x_coords = (0:n_x-1) * dx;
y_coords = (0:n_y-1) * dy;

%% ========== PHASE 1.5: ATTENUATION COMPENSATION (衰减补偿) ========== %%

if enable_attenuation_compensation
    fprintf('▶ PHASE 1.5: 空间衰减补偿...\n');
    
    % 保存未补偿的原始数据
    wavefield_raw_uncompensated = wavefield_3d;
    
    fprintf('  ├─ 衰减系数: %.4f /mm\n', attenuation_coefficient);
    fprintf('  ├─ 波源位置: (%.1f, %.1f) mm\n', source_position(1)*1e3, source_position(2)*1e3);
    fprintf('  ├─ 最小补偿距离: %.1f mm\n', min_compensation_distance*1e3);
    
    % 创建空间增益图 (向量化)
    [X_grid, Y_grid] = meshgrid(x_coords, y_coords);
    distance_map = sqrt((X_grid - source_position(1)).^2 + (Y_grid - source_position(2)).^2);
    
    % 计算有效补偿距离（仅补偿超过最小距离的部分）
    effective_distance = max(0, distance_map - min_compensation_distance);
    
    % 计算补偿增益 (改进逻辑: Geometric Spreading / Power Law)
    % Gain ~ d^0.4 -> 相比sqrt(d)模型，在远场增益增加更平缓，避免过度放大边缘噪声
    gain_map = 1 + attenuation_coefficient * (effective_distance * 1e3).^0.5;
    
    fprintf('  ├─ 补偿模型: Geometric Spreading (Sqrt)\n');
    fprintf('  ├─ 距离范围: %.1f - %.1f mm\n', min(distance_map(:))*1e3, max(distance_map(:))*1e3);
    fprintf('  ├─ 增益范围: %.2f - %.2f\n', min(gain_map(:)), max(gain_map(:)));
    
    % 应用补偿到每个空间点 (向量化: 使用bsxfun或隐式扩展)
    % gain_map是 [n_y, n_x], wavefield_3d是 [n_y, n_x, n_time]
    wavefield_3d = wavefield_3d .* gain_map;  % MATLAB自动扩展第三维
    
    fprintf('  └─ ✓ 衰减补偿完成\n\n');
    
    % 可视化增益分布
    figure('Name', '空间衰减补偿增益分布', 'Position', [100, 100, 600, 500]);
    imagesc(x_coords*1e3, y_coords*1e3, gain_map);
    axis equal tight;
    colormap('jet');
    colorbar;
    xlabel('X (mm)'); ylabel('Y (mm)');
    title(sprintf('补偿增益 (Coeff=%.4f, Sqrt Model)', attenuation_coefficient), 'FontWeight', 'bold');
    hold on;
    plot(source_position(1)*1e3, source_position(2)*1e3, 'w*', 'MarkerSize', 15, 'LineWidth', 2);
    set(gca, 'YDir', 'normal');
else
    % 不补偿时，原始和补偿数据相同
    wavefield_raw_uncompensated = wavefield_3d;
    
    % 空间坐标 (不补偿时也需要)
    dx = spacing; dy = spacing;
    x_coords = (0:n_x-1) * dx;
    y_coords = (0:n_y-1) * dy;
end

%% ========== PHASE 2: 3D FFT (空间-时间联合变换) ========== %%

fprintf('▶ PHASE 2: 波数-频率域变换 (3D FFT)...\n');

% 空间坐标
dx = spacing; dy = spacing;
x_coords = (0:n_x-1) * dx;
y_coords = (0:n_y-1) * dy;

% === 3D FFT: (y, x, t) → (k_y, k_x, f) ===
% 向量化操作，无循环
tic;
spectrum_3d = fftshift(fftn(wavefield_3d));
fft_time = toc;

% 构建波数-频率坐标轴
dk_x = 2*pi / (n_x * dx);
dk_y = 2*pi / (n_y * dy);
df = fs / n_time;

k_x = (-floor(n_x/2):ceil(n_x/2)-1) * dk_x;
k_y = (-floor(n_y/2):ceil(n_y/2)-1) * dk_y;
freq = (-floor(n_time/2):ceil(n_time/2)-1) * df;

[K_X, K_Y, ~] = meshgrid(k_x, k_y, freq);
K_mag = sqrt(K_X.^2 + K_Y.^2);

fprintf('  ├─ 3D FFT完成 (%.2f秒)\n', fft_time);
fprintf('  ├─ 频率范围: %.1f - %.1f kHz\n', min(freq)/1e3, max(freq)/1e3);
fprintf('  ├─ 波数范围: k_x=[%.1f, %.1f], k_y=[%.1f, %.1f] rad/m\n', ...
        min(k_x), max(k_x), min(k_y), max(k_y));
fprintf('  └─ ✓ 变换完成\n\n');

%% ========== PHASE 2.5: OUTLIER REMOVAL (异常波数滤除) ========== %%

if enable_outlier_removal
    fprintf('▶ PHASE 2.5: 异常波数滤除...\n');
    
    % 计算幅度谱
    amplitude = abs(spectrum_3d);
    
    % 统计阈值检测
    amp_mean = mean(amplitude(:));
    amp_std = std(amplitude(:));
    threshold_value = amp_mean + outlier_threshold * amp_std;
    
    % 找到异常点
    outlier_mask = amplitude > threshold_value;
    n_outliers = sum(outlier_mask(:));
    outlier_ratio = n_outliers / numel(amplitude) * 100;
    
    fprintf('  ├─ 异常阈值: %.2e (mean + %.1f×std)\n', threshold_value, outlier_threshold);
    fprintf('  ├─ 检测到异常点: %d (%.4f%%)\n', n_outliers, outlier_ratio);
    
    if n_outliers > 0
        switch outlier_replace_method
            case 'zero'
                % 置零
                spectrum_3d(outlier_mask) = 0;
                fprintf('  ├─ 替换方法: 置零\n');
                
            case 'median'
                % 替换为中值
                median_val = median(amplitude(:));
                spectrum_3d(outlier_mask) = median_val * exp(1i * angle(spectrum_3d(outlier_mask)));
                fprintf('  ├─ 替换方法: 中值 (%.2e)\n', median_val);
                
            case 'interpolate'
                % 替换为阈值（保留相位）
                spectrum_3d(outlier_mask) = threshold_value * exp(1i * angle(spectrum_3d(outlier_mask)));
                fprintf('  ├─ 替换方法: 截断到阈值\n');
        end
    end
    
    fprintf('  └─ ✓ 异常滤除完成\n\n');
end

%% ========== PHASE 3: DIRECTIONAL FILTER DESIGN ========== %%

fprintf('▶ PHASE 3: 定向滤波器设计 [%s]...\n', filter_mode);

k_max = max(K_mag(:));
k_lowpass = k_lowpass_ratio * k_max;
k_highpass = k_highpass_ratio * k_max;
transition = taper_width * k_max;

% 初始化滤波器
filter_mask = ones(size(spectrum_3d));

switch filter_mode
    case 'forward'
        % 正向波: k_y > 0 (或根据传播方向)
        fprintf('  ├─ 模式: 提取正向传播波 (入射波)\n');
        switch propagation_direction
            case 'positive_y'
                % 正向是 k_y > 0
                filter_mask = create_directional_mask(K_Y, 'positive', transition/max(abs(k_y)));
            case 'negative_y'
                filter_mask = create_directional_mask(K_Y, 'negative', transition/max(abs(k_y)));
            case 'positive_x'
                filter_mask = create_directional_mask(K_X, 'positive', transition/max(abs(k_x)));
        end
        
    case 'backward'
        % 反向波: k_y < 0 (反射波)
        fprintf('  ├─ 模式: 提取反向传播波 (反射波/缺陷信号)\n');
        switch propagation_direction
            case 'positive_y'
                % 反射波是 k_y < 0
                filter_mask = create_directional_mask(K_Y, 'negative', transition/max(abs(k_y)));
            case 'negative_y'
                filter_mask = create_directional_mask(K_Y, 'positive', transition/max(abs(k_y)));
            case 'positive_x'
                filter_mask = create_directional_mask(K_X, 'negative', transition/max(abs(k_x)));
        end
        
    case 'scattered'
        % 散射波: 高波数分量
        fprintf('  ├─ 模式: 提取散射波 (高波数分量)\n');
        % 高通滤波
        filter_mask = 1 - exp(-((K_mag).^2) / (2 * k_highpass^2));
        % 加低通防止极高频噪声
        filter_mask = filter_mask .* exp(-((K_mag - k_lowpass).^2) / (2 * transition^2));
        filter_mask(K_mag < k_highpass) = 0;
        filter_mask(K_mag > k_lowpass) = 0;
        
    case 'total'
        % 缺陷增强: 反射波 + 散射波
        fprintf('  ├─ 模式: 缺陷增强 (反射波 + 散射波)\n');
        switch propagation_direction
            case 'positive_y'
                backward_mask = create_directional_mask(K_Y, 'negative', transition/max(abs(k_y)));
            case 'negative_y'
                backward_mask = create_directional_mask(K_Y, 'positive', transition/max(abs(k_y)));
            otherwise
                backward_mask = ones(size(K_Y));
        end
        % 高波数分量
        highk_mask = 1 - exp(-((K_mag - k_highpass).^2) / (2 * (k_highpass/2)^2));
        highk_mask(K_mag < k_highpass) = 0;
        % 组合
        filter_mask = max(backward_mask, highk_mask * 0.5);
        
    case 'lowpass'
        % 低通滤波 (去噪)
        fprintf('  ├─ 模式: 低通滤波 (去噪)\n');
        filter_mask = exp(-((K_mag).^2) / (2 * k_lowpass^2));
        
    otherwise
        error('未知滤波模式: %s', filter_mode);
end

% 计算滤波器特性
filter_energy = sum(filter_mask(:)) / numel(filter_mask) * 100;
fprintf('  ├─ 滤波器通过率: %.1f%%\n', filter_energy);
fprintf('  └─ ✓ 滤波器设计完成\n\n');

%% ========== PHASE 4: APPLY FILTER & INVERSE TRANSFORM ========== %%

fprintf('▶ PHASE 4: 应用滤波并反变换...\n');

% 应用滤波
tic;
filtered_spectrum = spectrum_3d .* filter_mask;

% 逆3D FFT
wavefield_filtered = real(ifftn(ifftshift(filtered_spectrum)));
ifft_time = toc;

fprintf('  ├─ 滤波 + IFFT完成 (%.2f秒)\n', ifft_time);

% 能量统计
energy_orig = sum(wavefield_3d(:).^2);
energy_filt = sum(wavefield_filtered(:).^2);
fprintf('  ├─ 原始能量: %.4e\n', energy_orig);
fprintf('  ├─ 滤波后能量: %.4e (%.1f%%)\n', energy_filt, energy_filt/energy_orig*100);
fprintf('  └─ ✓ 反变换完成\n\n');

%% ========== PHASE 5: RMS IMAGING ========== %%

fprintf('▶ PHASE 5: RMS成像...\n');

% === 向量化RMS计算 ===
% 沿时间轴计算RMS (无循环)
% 沿时间轴计算RMS (无循环)
rms_raw_uncomp = sqrt(mean(wavefield_raw_uncompensated.^2, 3));
rms_original = sqrt(mean(wavefield_3d.^2, 3)); % 这里的rms_original实际上是补偿后但在波数滤波前的
rms_filtered = sqrt(mean(wavefield_filtered.^2, 3));

% 差值图 (突出滤波效果)
% 差值图 (突出滤波效果)
rms_diff = rms_filtered - rms_original;
% 使用归一化幅值计算比值
rms_original_norm = rms_original / max(rms_original(:));
rms_filtered_norm = rms_filtered / max(rms_filtered(:));
rms_ratio = rms_filtered_norm ./ (rms_original_norm + eps);

fprintf('  ├─ RMS范围 (原始): [%.4e, %.4e]\n', min(rms_original(:)), max(rms_original(:)));
fprintf('  ├─ RMS范围 (滤波后): [%.4e, %.4e]\n', min(rms_filtered(:)), max(rms_filtered(:)));
fprintf('  └─ ✓ RMS计算完成\n\n');

%% ========== PHASE 5.5: RMS OUTLIER REMOVAL ========== %%

if enable_rms_outlier_removal
    fprintf('▶ PHASE 5.5: RMS异常值滤除...\n');
    
    % 定义内部区域 (排除边缘)
    inner_range_y = (edge_margin+1):(n_y-edge_margin);
    inner_range_x = (edge_margin+1):(n_x-edge_margin);
    
    % === 处理原始 RMS ===
    rms_inner = rms_original(inner_range_y, inner_range_x);
    rms_mean = mean(rms_inner(:));
    rms_std = std(rms_inner(:));
    lower_thresh = rms_mean - rms_outlier_threshold * rms_std;
    upper_thresh = rms_mean + rms_outlier_threshold * rms_std;
    
    outlier_mask_orig = (rms_original < lower_thresh) | (rms_original > upper_thresh);
    n_outliers_orig = sum(outlier_mask_orig(:));
    
    if n_outliers_orig > 0
        rms_original_clean = rms_original;
        rms_original_clean(outlier_mask_orig) = NaN;
        % 用邻域中值填充
        for ii = 1:n_y
            for jj = 1:n_x
                if isnan(rms_original_clean(ii, jj))
                    i_min = max(1, ii-1); i_max = min(n_y, ii+1);
                    j_min = max(1, jj-1); j_max = min(n_x, jj+1);
                    neighbors = rms_original_clean(i_min:i_max, j_min:j_max);
                    valid = neighbors(~isnan(neighbors));
                    if ~isempty(valid)
                        rms_original_clean(ii, jj) = median(valid);
                    else
                        rms_original_clean(ii, jj) = rms_mean;
                    end
                end
            end
        end
        rms_original = rms_original_clean;
    end
    
    % === 处理滤波后 RMS ===
    rms_inner_filt = rms_filtered(inner_range_y, inner_range_x);
    rms_mean_filt = mean(rms_inner_filt(:));
    rms_std_filt = std(rms_inner_filt(:));
    lower_thresh_filt = rms_mean_filt - rms_outlier_threshold * rms_std_filt;
    upper_thresh_filt = rms_mean_filt + rms_outlier_threshold * rms_std_filt;
    
    outlier_mask_filt = (rms_filtered < lower_thresh_filt) | (rms_filtered > upper_thresh_filt);
    n_outliers_filt = sum(outlier_mask_filt(:));
    
    if n_outliers_filt > 0
        rms_filtered_clean = rms_filtered;
        rms_filtered_clean(outlier_mask_filt) = NaN;
        for ii = 1:n_y
            for jj = 1:n_x
                if isnan(rms_filtered_clean(ii, jj))
                    i_min = max(1, ii-1); i_max = min(n_y, ii+1);
                    j_min = max(1, jj-1); j_max = min(n_x, jj+1);
                    neighbors = rms_filtered_clean(i_min:i_max, j_min:j_max);
                    valid = neighbors(~isnan(neighbors));
                    if ~isempty(valid)
                        rms_filtered_clean(ii, jj) = median(valid);
                    else
                        rms_filtered_clean(ii, jj) = rms_mean_filt;
                    end
                end
            end
        end
        rms_filtered = rms_filtered_clean;
    end
    
    % 重新计算比值 (使用归一化幅值)
    rms_original_norm = rms_original / max(rms_original(:));
    rms_filtered_norm = rms_filtered / max(rms_filtered(:));
    rms_ratio = rms_filtered_norm ./ (rms_original_norm + eps);
    
    fprintf('  ├─ 原始RMS异常点: %d\n', n_outliers_orig);
    fprintf('  ├─ 滤波RMS异常点: %d\n', n_outliers_filt);
    fprintf('  └─ ✓ RMS异常值滤除完成\n\n');
end

%% ========== PHASE 6: INTERPOLATION ========== %%

fprintf('▶ PHASE 6: %d倍插值...\n', interp_factor);

[X_orig, Y_orig] = meshgrid(x_coords, y_coords);
x_interp = linspace(x_coords(1), x_coords(end), n_x * interp_factor);
y_interp = linspace(y_coords(1), y_coords(end), n_y * interp_factor);
[X_interp, Y_interp] = meshgrid(x_interp, y_interp);



rms_raw_uncomp_interp = interp2(X_orig, Y_orig, rms_raw_uncomp, X_interp, Y_interp, 'linear');
rms_original_interp = interp2(X_orig, Y_orig, rms_original, X_interp, Y_interp, 'linear');
rms_filtered_interp = interp2(X_orig, Y_orig, rms_filtered, X_interp, Y_interp, 'linear');
rms_ratio_interp = interp2(X_orig, Y_orig, rms_ratio, X_interp, Y_interp, 'linear');

fprintf('  └─ ✓ 插值完成\n\n');

%% ========== PHASE 7: VISUALIZATION ========== %%

fprintf('▶ PHASE 7: 可视化...\n');

% 转换为mm
x_mm = x_interp * 1e3;
y_mm = y_interp * 1e3;

% === Figure 1: RMS对比 ===
figure('Name', sprintf('波数域滤波RMS对比 [%s]', filter_mode), ...
       'Position', [50, 50, 1600, 800]);

% 子图1: 原始未补偿
subplot(2, 2, 1);
imagesc(x_mm, y_mm, rms_raw_uncomp_interp);
axis equal tight;
colormap(gca, colormap_style);
colorbar;
xlabel('X (mm)'); ylabel('Y (mm)');
title('原始RMS (未补偿)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'YDir', 'normal');

% 子图2: 补偿后(滤波前)
subplot(2, 2, 2);
imagesc(x_mm, y_mm, rms_original_interp);
axis equal tight;
colormap(gca, colormap_style);
colorbar;
xlabel('X (mm)'); ylabel('Y (mm)');
title(sprintf('衰减补偿后RMS (Coeff=%.3f)', attenuation_coefficient), 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'YDir', 'normal');

% 子图3: 滤波后
subplot(2, 2, 3);
imagesc(x_mm, y_mm, rms_filtered_interp);
axis equal tight;
colormap(gca, colormap_style);
colorbar;
xlabel('X (mm)'); ylabel('Y (mm)');
title(sprintf('滤波后RMS [%s]', filter_mode), 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'YDir', 'normal');

% 子图4: 比值图
subplot(2, 2, 4);
imagesc(x_mm, y_mm, rms_ratio_interp);
axis equal tight;
colormap(gca, 'hot');
colorbar;
clim([0, 2]);  % 突出变化区域
xlabel('X (mm)'); ylabel('Y (mm)');
title('RMS归一化比值 (滤波/补偿后)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'YDir', 'normal');

sgtitle(sprintf('WAVENUMBER FILTERING RESULT [Mode: %s]', upper(filter_mode)), ...
        'FontSize', 16, 'FontWeight', 'bold');

% === Figure 2: 2D频谱切片 ===
figure('Name', '波数域频谱分析', 'Position', [100, 100, 1400, 500]);

% 选择中间频率切片
mid_freq_idx = round(n_time / 2);
spectrum_slice_orig = squeeze(abs(spectrum_3d(:, :, mid_freq_idx)));
spectrum_slice_filt = squeeze(abs(filtered_spectrum(:, :, mid_freq_idx)));
filter_slice = squeeze(filter_mask(:, :, mid_freq_idx));

% 转换为rad/mm显示
kx_mm = k_x / 1e3;
ky_mm = k_y / 1e3;

subplot(1, 3, 1);
imagesc(kx_mm, ky_mm, 20*log10(spectrum_slice_orig + eps));
axis equal tight;
colormap(gca, 'jet');
colorbar;
clim([max(20*log10(spectrum_slice_orig(:)))-60, max(20*log10(spectrum_slice_orig(:)))]);
xlabel('k_x (rad/mm)'); ylabel('k_y (rad/mm)');
title('原始波数谱 (dB)', 'FontSize', 13);
set(gca, 'YDir', 'normal');

subplot(1, 3, 2);
imagesc(kx_mm, ky_mm, filter_slice);
axis equal tight;
colormap(gca, 'gray');
colorbar;
xlabel('k_x (rad/mm)'); ylabel('k_y (rad/mm)');
title(sprintf('滤波器掩码 [%s]', filter_mode), 'FontSize', 13);
set(gca, 'YDir', 'normal');

subplot(1, 3, 3);
imagesc(kx_mm, ky_mm, 20*log10(spectrum_slice_filt + eps));
axis equal tight;
colormap(gca, 'jet');
colorbar;
clim([max(20*log10(spectrum_slice_orig(:)))-60, max(20*log10(spectrum_slice_orig(:)))]);
xlabel('k_x (rad/mm)'); ylabel('k_y (rad/mm)');
title('滤波后波数谱 (dB)', 'FontSize', 13);
set(gca, 'YDir', 'normal');

sgtitle('K-SPACE ANALYSIS @ f=0', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('  └─ ✓ 可视化完成\n\n');

%% ========== MISSION COMPLETE ========== %%
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                 MISSION COMPLETE ✓                        ║\n');
fprintf('║  建议: 尝试不同filter_mode观察缺陷对比效果                   ║\n');
fprintf('║  - ''backward'' : 反射波 (缺陷反射信号)                      ║\n');
fprintf('║  - ''scattered'': 散射波 (高波数分量)                        ║\n');
fprintf('║  - ''total''    : 综合增强                                   ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');

%% ========== HELPER FUNCTIONS ========== %%

function mask = create_directional_mask(K, direction, taper_ratio)
    % CREATE_DIRECTIONAL_MASK 创建定向滤波掩码
    % 输入:
    %   K - 波数矩阵 (k_x 或 k_y)
    %   direction - 'positive' 或 'negative'
    %   taper_ratio - 边缘过渡比例 (0-1)
    
    k_max = max(abs(K(:)));
    taper_width = taper_ratio * k_max;
    
    mask = zeros(size(K));
    
    switch direction
        case 'positive'
            % 保留 K > 0 的部分
            mask(K > taper_width) = 1;
            % 过渡区
            transition_zone = (K >= 0) & (K <= taper_width);
            mask(transition_zone) = 0.5 * (1 - cos(pi * K(transition_zone) / taper_width));
            
        case 'negative'
            % 保留 K < 0 的部分
            mask(K < -taper_width) = 1;
            % 过渡区
            transition_zone = (K <= 0) & (K >= -taper_width);
            mask(transition_zone) = 0.5 * (1 - cos(pi * abs(K(transition_zone)) / taper_width));
    end
end
