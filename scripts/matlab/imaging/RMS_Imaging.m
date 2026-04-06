%% RMS成像程序
% 计算每个点的RMS值并进行可视化

clear; clc; close all;

%% ========== 参数配置区 ========== %%
% 所有可调参数集中在此处,方便修改

% 数据文件路径
data_file = "/Users/zyt/ANW/DnCNN2/results/best/20260331_212623/denoised_20260331_213005_full.mat";

% 点阵参数
n_points = [41, 41];     % 点阵大小: 标量(正方形) 或 [n_x, n_y](矩形)
spacing = 1e-3;          % 物理间距 (m), 1mm

% 带通滤波器参数
center_freq = 200e3;     % 中心频率 (Hz)
bandwidth = 200e3;        % 带宽 (Hz)
filter_order = 2;        % 滤波器阶数

% 小波去噪参数
wavelet_name = 'coif3';    % 小波基: 'db4', 'sym4', 'coif3'
wavelet_level = 3;       % 分解层数
threshold_method = 'soft'; % 阈值方法: 'soft' 或 'hard'

% 异常值处理参数
edge_margin = 2;         % 排除边缘点的层数
threshold_factor = 3;    % 全局异常值检测阈值 (标准差倍数，仅用于原始RMS图)

% 自适应梯度去噪参数（用于滤波/小波/衰减补偿三个RMS图）
gradient_denoise_enable = true;    % 是否启用自适应梯度去噪
gradient_denoise_sigma = 2.0;      % 归一化残差异常判定阈值 (σ倍数)
gradient_denoise_nbands = 8;       % 距离分段数（用于计算各距离圈的局部σ）

% 插值参数
interp_factor = 4;       % 插值倍数 (用于提高显示分辨率)

% 二值化参数（作用于小波去噪+衰减补偿后的最终RMS图）
binarization_enable = true;          % 是否启用二值化处理
binarization_method = 'fixed';  % 阈值方法: 'percentile'(百分位数) | 'fixed'(归一化固定值)
binarization_percentile = 95;        % 百分位数阈值 (0~100), 仅 method='percentile' 时有效
binarization_fixed_threshold = 0.87;  % 归一化固定阈值 (0~1), 仅 method='fixed' 时有效
binarization_show_overlay = true;    % 是否在二值化图上叠加原图作为背景参考

% 衰减补偿参数
enable_attenuation_compensation = true;  % 是否启用衰减补偿
attenuation_method = 'spatial';          % 补偿方法: 'spatial'(基于空间), 'temporal'(基于时间), 'combined'(混合)
attenuation_coefficient = 0.03;          % 衰减系数 (1/mm 或 1/μs, 根据方法而定)
source_position = [5, 51] * 1e-3;       % 波源位置 [x, y] (米), 默认在Y轴下方中心
min_compensation_distance = 1e-3;       % 最小补偿距离 (米), 小于此距离的区域增益为1（不补偿）

% 时间段选择参数
enable_time_window = true;               % 是否启用时间段截取
time_window_start = 0;                   % 起始时间 (μs)
time_window_end = 160;                   % 结束时间 (μs)

%% ========== 数据预处理 ========== %%

% 加载原始数据
[data_xyt_full, data_time_full, data_x, data_y, fs] = mat_loader(data_file, n_points, spacing);

% 应用时间段截取
if enable_time_window
    fprintf('\n========== 应用时间段截取 ==========\n');
    fprintf('截取范围: %.1f - %.1f μs\n', time_window_start, time_window_end);
    
    % 找到时间窗口对应的索引
    time_us = data_time_full * 1e6;  % 转换为微秒
    idx_start = find(time_us >= time_window_start, 1, 'first');
    idx_end = find(time_us <= time_window_end, 1, 'last');
    
    if isempty(idx_start) || isempty(idx_end) || idx_start >= idx_end
        error('时间窗口范围无效！请检查 time_window_start 和 time_window_end 参数');
    end
    
    % 截取数据
    data_time = data_time_full(idx_start:idx_end);
    data_xyt = data_xyt_full(:, :, idx_start:idx_end);
    
    fprintf('原始时间点数: %d\n', length(data_time_full));
    fprintf('截取后时间点数: %d\n', length(data_time));
    fprintf('实际截取范围: %.2f - %.2f μs\n', data_time(1)*1e6, data_time(end)*1e6);
else
    data_time = data_time_full;
    data_xyt = data_xyt_full;
    fprintf('\n时间段截取已禁用，使用完整时间范围\n');
end

% 提取点阵尺寸
[n_y, n_x, ~] = size(data_xyt);

%% ========== 应用带通滤波器 ========== %%

fprintf('\n========== 应用带通滤波器 ==========\n');
Filter.printInfo(center_freq, bandwidth, filter_order, fs);

data_xyt_filtered = zeros(size(data_xyt));

for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt(i, j, :));
        data_xyt_filtered(i, j, :) = Filter.apply(point_signal, fs, center_freq, bandwidth, filter_order);
    end
end

fprintf('滤波完成，共处理 %d 个空间点\n', n_x * n_y);

%% ========== 应用小波去噪 ========== %%

fprintf('\n========== 应用小波去噪 ==========\n');
Filter.printWaveletInfo(wavelet_name, wavelet_level, threshold_method);

data_xyt_wavelet = zeros(size(data_xyt));

fprintf('正在对滤波后的信号进行小波去噪...\n');
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt_filtered(i, j, :));
        data_xyt_wavelet(i, j, :) = Filter.waveletDenoise(point_signal, wavelet_name, wavelet_level, threshold_method);
    end
end

fprintf('小波去噪完成，共处理 %d 个空间点\n', n_x * n_y);

%% ========== 应用衰减补偿 ========== %%

if enable_attenuation_compensation
    fprintf('\n========== 应用衰减补偿 ==========\n');
    fprintf('补偿方法: %s\n', attenuation_method);
    fprintf('衰减系数: %.4f\n', attenuation_coefficient);
    fprintf('波源位置: (%.1f, %.1f) mm\n', source_position(1)*1e3, source_position(2)*1e3);
    
    data_xyt_compensated = zeros(size(data_xyt_wavelet));
    
    switch attenuation_method
        case 'spatial'
            % 基于空间距离的补偿: gain(i,j) = exp(alpha * max(0, distance - min_distance))
            % 为每个空间点计算到波源的距离
            fprintf('使用空间距离补偿: gain = exp(α * max(0, d - d_min))\n');
            fprintf('最小补偿距离: %.1f mm\n', min_compensation_distance * 1e3);
            
            % 创建空间增益图
            gain_map = zeros(n_y, n_x);
            for i = 1:n_y

                for j = 1:n_x
                    % 计算当前点到波源的距离 (米)
                    dx = data_x(j) - source_position(1);
                    dy = data_y(i) - source_position(2);
                    distance = sqrt(dx^2 + dy^2);
                    
                    % 计算有效补偿距离（仅补偿超过最小距离的部分）
                    effective_distance = max(0, distance - min_compensation_distance);
                    
                    % 计算补偿增益 (距离转换为mm)
                    gain_map(i, j) = exp(attenuation_coefficient * effective_distance * 1e3);
                end
            end
            
            % 计算最大距离（使用meshgrid处理矩形点阵）
            [X_grid, Y_grid] = meshgrid(data_x, data_y);
            all_distances = sqrt((X_grid - source_position(1)).^2 + (Y_grid - source_position(2)).^2);
            fprintf('距离范围: %.1f - %.1f mm\n', min(all_distances(:))*1e3, max(all_distances(:))*1e3);
            fprintf('增益范围: %.2f - %.2f\n', min(gain_map(:)), max(gain_map(:)));
            
            % 对每个空间点应用对应的增益
            fprintf('正在对小波去噪后的信号进行空间衰减补偿...\n');
            for i = 1:n_y

                for j = 1:n_x
                    point_signal = squeeze(data_xyt_wavelet(i, j, :));
                    data_xyt_compensated(i, j, :) = point_signal * gain_map(i, j);
                end
            end
            
            % 显示空间增益分布图
            figure('Name', '空间衰减补偿增益分布', 'Position', [100, 100, 800, 600]);
            imagesc(data_x * 1e3, data_y * 1e3, gain_map);
            axis equal tight;
            colormap('jet');
            colorbar;
            xlabel('X 位置 (mm)', 'FontSize', 12);
            ylabel('Y 位置 (mm)', 'FontSize', 12);
            title(sprintf('空间补偿增益分布 (α=%.4f /mm)', attenuation_coefficient), ...
                  'FontSize', 14, 'FontWeight', 'bold');
            hold on;
            plot(source_position(1)*1e3, source_position(2)*1e3, 'w*', 'MarkerSize', 15, 'LineWidth', 2);
            text(source_position(1)*1e3, source_position(2)*1e3 + 2, '波源', ...
                 'Color', 'white', 'FontSize', 12, 'HorizontalAlignment', 'center');
            grid on;
            set(gca, 'YDir', 'normal');
            
        case 'temporal'
            % 基于时间的补偿 (原有方法)
            fprintf('使用时间补偿: gain = exp(α * t_μs)\n');
            time_vector_us = data_time * 1e6;
            gain = exp(attenuation_coefficient * time_vector_us);
            gain = gain(:);
            
            fprintf('时间范围: %.1f - %.1f μs, 增益范围: %.2f - %.2f\n', ...
                    time_vector_us(1), time_vector_us(end), gain(1), gain(end));
            
            fprintf('正在对小波去噪后的信号进行时间衰减补偿...\n');
            for i = 1:n_y

                for j = 1:n_x
                    point_signal = squeeze(data_xyt_wavelet(i, j, :));
                    point_signal = point_signal(:);
                    data_xyt_compensated(i, j, :) = point_signal .* gain;
                end
            end
            
            % 显示时间增益曲线
            figure('Name', '时间衰减补偿增益曲线', 'Position', [100, 100, 800, 400]);
            plot(data_time * 1e6, gain, 'b-', 'LineWidth', 2);
            xlabel('时间 (μs)', 'FontSize', 12);
            ylabel('补偿增益', 'FontSize', 12);
            title(sprintf('时间补偿增益函数 (α=%.4f /μs)', attenuation_coefficient), ...
                  'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            xlim([data_time(1), data_time(end)] * 1e6);
            
        case 'combined'
            % 混合补偿: 空间 × 时间
            fprintf('使用混合补偿: gain = exp(α * (distance + c*t))\n');
            
            % 计算声速 (假设)
            sound_speed = 3000; % m/s, 根据材料调整
            time_vector_us = data_time * 1e6;
            
            fprintf('正在对小波去噪后的信号进行混合衰减补偿...\n');
            for i = 1:n_y

                for j = 1:n_x
                    % 空间距离
                    dx = data_x(j) - source_position(1);
                    dy = data_y(i) - source_position(2);
                    distance = sqrt(dx^2 + dy^2);
                    
                    % 计算有效补偿距离（仅补偿超过最小距离的部分）
                    effective_distance_mm = max(0, distance - min_compensation_distance) * 1e3;
                    
                    % 空间增益（标量）
                    spatial_gain = exp(attenuation_coefficient * effective_distance_mm * 0.5);
                    
                    % 时间增益（列向量）
                    time_gain = exp(attenuation_coefficient * time_vector_us * 0.5);
                    time_gain = time_gain(:);
                    
                    point_signal = squeeze(data_xyt_wavelet(i, j, :));
                    point_signal = point_signal(:);
                    data_xyt_compensated(i, j, :) = point_signal .* time_gain * spatial_gain;
                end
            end
            
        otherwise
            error('未知的衰减补偿方法: %s (可选: spatial, temporal, combined)', attenuation_method);
    end
    
    fprintf('衰减补偿完成，共处理 %d 个空间点\n', n_x * n_y);
    
else
    fprintf('\n衰减补偿已禁用\n');
    data_xyt_compensated = data_xyt_wavelet;
end

%% ========== 计算RMS值 ========== %%

fprintf('\n正在计算RMS值...\n');

% 原始数据RMS
rms_image = zeros(n_y, n_x);
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt(i, j, :));
        rms_image(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 滤波后数据RMS
rms_image_filtered = zeros(n_y, n_x);
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt_filtered(i, j, :));
        rms_image_filtered(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 小波去噪后数据RMS
rms_image_wavelet = zeros(n_y, n_x);
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt_wavelet(i, j, :));
        rms_image_wavelet(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 衰减补偿后数据RMS
% 注意: 对于空间补偿 (attenuation_method='spatial')，rms_image_compensated
% 将在自适应去噪之后由 gain_map .* rms_image_wavelet 推导得到，
% 以确保去噪作用于自然衰减梯度的小波RMS图（幂律模型拟合更准确）。
% 对于时间/混合补偿，增益随时间变化不满足 RMS(g(t)*x(t))=g*RMS(x) 的等价性，
% 因此仍从补偿后的时域信号计算。
if enable_attenuation_compensation && strcmp(attenuation_method, 'spatial')
    % 空间补偿：延迟到去噪后计算（见下方自适应梯度去噪段）
    rms_image_compensated = zeros(n_y, n_x);  % 占位，后续覆盖
else
    % 时间/混合补偿，或未启用补偿：从时域补偿后信号计算RMS
    rms_image_compensated = zeros(n_y, n_x);
    for i = 1:n_y
        for j = 1:n_x
            point_signal = squeeze(data_xyt_compensated(i, j, :));
            rms_image_compensated(i, j) = sqrt(mean(point_signal.^2));
        end
    end
end

fprintf('RMS计算完成\n');

%% ========== 2D 空间去噪 (中值预滤波) ========== %%
fprintf('\n正在应用空间 2D 中值滤波，消除图像噪点...\n');
% 使用 3x3 邻域进行中值滤波，这种非线性滤波对椒盐噪声和孤立噪点极其有效
rms_image_filtered = medfilt2(rms_image_filtered, [3 3]);
rms_image_wavelet = medfilt2(rms_image_wavelet, [3 3]);

%% ========== 原始RMS图：全局阈值异常值检测 ========== %%

fprintf('\n正在对原始RMS图进行全局阈值异常值检测...\n');

inner_range_y = (edge_margin+1):(n_y-edge_margin);
inner_range_x = (edge_margin+1):(n_x-edge_margin);

rms_inner = rms_image(inner_range_y, inner_range_x);
rms_mean = mean(rms_inner(:));
rms_std = std(rms_inner(:));
lower_threshold = rms_mean - threshold_factor * rms_std;
upper_threshold = rms_mean + threshold_factor * rms_std;

outlier_mask = (rms_image < lower_threshold) | (rms_image > upper_threshold);
rms_image_processed = rms_image;
rms_image_processed(outlier_mask) = NaN;

if sum(outlier_mask(:)) > 0
    rms_image_repaired = rms_image_processed;
    for i = 1:n_y
        for j = 1:n_x
            if isnan(rms_image_processed(i, j))
                i_min = max(1, i-1); i_max = min(n_y, i+1);
                j_min = max(1, j-1); j_max = min(n_x, j+1);
                neighbors = rms_image_processed(i_min:i_max, j_min:j_max);
                valid_neighbors = neighbors(~isnan(neighbors));
                if ~isempty(valid_neighbors)
                    rms_image_repaired(i, j) = median(valid_neighbors);
                else
                    rms_image_repaired(i, j) = rms_mean;
                end
            end
        end
    end
    rms_image = rms_image_repaired;
else
    rms_image = rms_image_processed;
end
fprintf('原始RMS图异常值处理完成\n');

%% ========== 自适应梯度去噪（滤波/小波/衰减补偿三图）========== %%
% 算法原理：
%   1. 以波源为参考，计算所有网格点到波源的距离 d(i,j)
%   2. 用 Theil-Sen 鲁棒中位数斜率回归拟合 ln(RMS) ~ ln(d)
%      得到幂律衰减模型 RMS_ref(d) = A * d^(-beta)
%   3. 将扫描域按距离等分为 N 个圆环带，在每个圆环带内
%      计算残差 (RMS - RMS_ref) 的局部标准差 sigma_local
%   4. 归一化残差超过 gradient_denoise_sigma 倍 sigma_local
%      的点判定为异常点，用邻域中值修复

if gradient_denoise_enable
    fprintf('\n========== 自适应梯度去噪（补偿前） ==========\n');
    fprintf('衰减模型: 幂律  RMS_ref(d) = A * d^(-β)\n');
    fprintf('异常判定阈值: %.1f σ (局部)\n', gradient_denoise_sigma);
    fprintf('距离分段数: %d\n', gradient_denoise_nbands);
    
    % 计算各点到波源的距离矩阵
    [X_grid, Y_grid] = meshgrid(data_x, data_y);
    dist_map = sqrt((X_grid - source_position(1)).^2 + (Y_grid - source_position(2)).^2);
    
    % 对带通滤波图和小波去噪图进行自适应梯度去噪
    % （此时两图均处于自然衰减状态，幂律模型拟合最准确）
    rms_image_filtered = adaptiveGradientDenoise(rms_image_filtered, dist_map, ...
        gradient_denoise_sigma, gradient_denoise_nbands, '带通滤波RMS图');
    rms_image_wavelet  = adaptiveGradientDenoise(rms_image_wavelet,  dist_map, ...
        gradient_denoise_sigma, gradient_denoise_nbands, '小波去噪RMS图');
    
    % ---- 衰减补偿后的RMS图：在去噪后推导 ----
    % 对于空间补偿：RMS(gain * x) = gain * RMS(x)，
    % 因此 rms_image_compensated = gain_map .* rms_image_wavelet（已去噪）
    % 这等价于先补偿时域信号再取RMS，但去噪是在自然衰减图上完成的
    if enable_attenuation_compensation && strcmp(attenuation_method, 'spatial')
        fprintf('  [衰减补偿RMS图] 空间补偿：使用去噪后的小波RMS图推导（补偿前去噪）\n');
        rms_image_compensated = gain_map .* rms_image_wavelet;
    else
        % 时间/混合补偿或未启用补偿：对已有的rms_image_compensated直接去噪
        rms_image_compensated = adaptiveGradientDenoise(rms_image_compensated, dist_map, ...
            gradient_denoise_sigma, gradient_denoise_nbands, '衰减补偿RMS图(补偿后去噪)');
    end
    
    fprintf('自适应梯度去噪完成\n');
else
    fprintf('\n自适应梯度去噪已禁用，对三图使用全局阈值方法...\n');
    % 退回到原始全局阈值方法处理三图
    for rms_cell = {{'rms_image_filtered', rms_image_filtered}, ...
                    {'rms_image_wavelet',  rms_image_wavelet}, ...
                    {'rms_image_compensated', rms_image_compensated}}
        entry = rms_cell{1};
        rms_cur = entry{2};
        rms_inner_cur = rms_cur(inner_range_y, inner_range_x);
        rms_mean_cur  = mean(rms_inner_cur(:));
        rms_std_cur   = std(rms_inner_cur(:));
        omask = (rms_cur < rms_mean_cur - threshold_factor*rms_std_cur) | ...
                (rms_cur > rms_mean_cur + threshold_factor*rms_std_cur);
        rms_cur(omask) = NaN;
        if any(omask(:))
            [ny_c, nx_c] = size(rms_cur);
            rms_rep = rms_cur;
            for ii = 1:ny_c
                for jj = 1:nx_c
                    if isnan(rms_cur(ii,jj))
                        nb = rms_cur(max(1,ii-1):min(ny_c,ii+1), max(1,jj-1):min(nx_c,jj+1));
                        vn = nb(~isnan(nb));
                        if ~isempty(vn); rms_rep(ii,jj) = median(vn);
                        else;            rms_rep(ii,jj) = rms_mean_cur; end
                    end
                end
            end
            rms_cur = rms_rep;
        end
        eval([entry{1} ' = rms_cur;']);
    end
end

fprintf('异常值处理完成\n');

%% ========== 插值提高显示分辨率 ========== %%

fprintf('\n正在进行 %d 倍插值...\n', interp_factor);

[X_orig, Y_orig] = meshgrid(data_x, data_y);

x_interp = linspace(data_x(1), data_x(end), n_x * interp_factor - (interp_factor - 1));
y_interp = linspace(data_y(1), data_y(end), n_y * interp_factor - (interp_factor - 1));
[X_interp, Y_interp] = meshgrid(x_interp, y_interp);

rms_image_interp = interp2(X_orig, Y_orig, rms_image, X_interp, Y_interp, 'linear');
rms_image_filtered_interp = interp2(X_orig, Y_orig, rms_image_filtered, X_interp, Y_interp, 'linear');
rms_image_wavelet_interp = interp2(X_orig, Y_orig, rms_image_wavelet, X_interp, Y_interp, 'linear');
rms_image_compensated_interp = interp2(X_orig, Y_orig, rms_image_compensated, X_interp, Y_interp, 'linear');

fprintf('插值完成\n');

%% ========== 可视化: RMS成像对比 ========== %%

figure('Name', 'RMS成像对比 (原始 vs 滤波 vs 小波去噪 vs 衰减补偿)', 'Position', [50, 50, 1600, 900]);

% 左上图：原始RMS成像
subplot(2, 2, 1);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title('原始RMS成像', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

% 右上图：滤波后RMS成像
subplot(2, 2, 2);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_filtered_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title(sprintf('带通滤波 (%.0f-%.0f kHz)', ...
              (center_freq-bandwidth/2)/1e3, (center_freq+bandwidth/2)/1e3), ...
      'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

% 左下图：小波去噪后RMS成像
subplot(2, 2, 3);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_wavelet_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title(sprintf('带通滤波 + 小波去噪 (%s)', wavelet_name), ...
      'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

% 右下图：衰减补偿后RMS成像
subplot(2, 2, 4);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_compensated_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
if enable_attenuation_compensation
    title(sprintf('小波去噪 + 衰减补偿 (%s, α=%.2f)', attenuation_method, attenuation_coefficient), ...
          'FontSize', 14, 'FontWeight', 'bold');
else
    title('衰减补偿 (未启用)', 'FontSize', 14, 'FontWeight', 'bold');
end
grid on;
set(gca, 'YDir', 'normal');

%% ========== 可视化: 时域波形对比 ========== %%

fprintf('\n正在生成时域波形对比...\n');

% 随机选择一个点 (避免边缘点)
rng('shuffle');
random_i = randi([edge_margin+1, n_y-edge_margin]);
random_j = randi([edge_margin+1, n_x-edge_margin]);

fprintf('  随机选择的点: (%d, %d)\n', random_i, random_j);
fprintf('  物理位置: (%.2f mm, %.2f mm)\n', data_x(random_i)*1e3, data_y(random_j)*1e3);

% 提取该点的四种处理信号
signal_original = squeeze(data_xyt(random_i, random_j, :));
signal_filtered = squeeze(data_xyt_filtered(random_i, random_j, :));
signal_wavelet = squeeze(data_xyt_wavelet(random_i, random_j, :));
signal_compensated = squeeze(data_xyt_compensated(random_i, random_j, :));

% 创建时域对比图
figure('Name', '随机点时域波形对比 (四种处理方法)', 'Position', [100, 100, 1600, 800]);

% 原始信号
subplot(2, 2, 1);
plot(data_time * 1e6, signal_original, 'b-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('原始信号 [位置: (%.1f, %.1f) mm]', data_x(random_i)*1e3, data_y(random_j)*1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 滤波后信号
subplot(2, 2, 2);
plot(data_time * 1e6, signal_filtered, 'r-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('带通滤波 (%.0f-%.0f kHz)', ...
              (center_freq-bandwidth/2)/1e3, (center_freq+bandwidth/2)/1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 小波去噪后信号
subplot(2, 2, 3);
plot(data_time * 1e6, signal_wavelet, 'g-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('带通滤波 + 小波去噪 (%s)', wavelet_name), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 衰减补偿后信号
subplot(2, 2, 4);
plot(data_time * 1e6, signal_compensated, 'm-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
if enable_attenuation_compensation
    title(sprintf('小波去噪 + 衰减补偿 (%s, α=%.2f)', attenuation_method, attenuation_coefficient), ...
          'FontSize', 13, 'FontWeight', 'bold');
else
    title('衰减补偿 (未启用)', 'FontSize', 13, 'FontWeight', 'bold');
end
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

fprintf('\n成像完成！\n');

%% ========== 二值化阈值处理与可视化 ========== %%

if binarization_enable
    fprintf('\n========== 二值化阈值处理 ==========\n');
    
    % 计算二值化阈值
    valid_pixels = rms_image_compensated_interp(isfinite(rms_image_compensated_interp));
    
    switch binarization_method
        case 'percentile'
            binarization_threshold_val = prctile(valid_pixels, binarization_percentile);
            threshold_label = sprintf('百分位数法 (P=%.0f%%)', binarization_percentile);
        case 'fixed'
            v_min = min(valid_pixels);
            v_max = max(valid_pixels);
            binarization_threshold_val = v_min + binarization_fixed_threshold * (v_max - v_min);
            threshold_label = sprintf('归一化固定值法 (threshold=%.2f)', binarization_fixed_threshold);
        otherwise
            error('未知的二值化方法: %s (可选: percentile, fixed)', binarization_method);
    end
    
    fprintf('二值化方法: %s\n', threshold_label);
    fprintf('实际阈值: %.6f\n', binarization_threshold_val);
    
    % 生成二值化图像 (1=高RMS区域, 0=低RMS区域)
    binary_image = rms_image_compensated_interp >= binarization_threshold_val;
    
    n_high = sum(binary_image(:));
    n_total = numel(binary_image);
    fprintf('高RMS区域像素数: %d / %d (%.1f%%)\n', n_high, n_total, n_high/n_total*100);
    
    % ---- 独立可视化 ----
    if binarization_show_overlay
        % 双子图模式：左为独立二值图，右为叠加背景的对照图
        figure('Name', '二值化结果 (最终RMS图: 小波去噪 + 衰减补偿)', ...
               'Position', [150, 150, 1400, 560]);
        
        % 左图：纯二值化图
        subplot(1, 2, 1);
        imagesc(x_interp * 1e3, y_interp * 1e3, binary_image);
        axis equal tight;
        colormap(gca, [0.12 0.12 0.18; 0.98 0.45 0.15]);  % 深色背景 + 橙色高值区
        colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'低RMS (0)', '高RMS (1)'});
        xlabel('X 位置 (mm)', 'FontSize', 12);
        ylabel('Y 位置 (mm)', 'FontSize', 12);
        title({'二值化结果', threshold_label}, 'FontSize', 13, 'FontWeight', 'bold');
        grid on;
        set(gca, 'YDir', 'normal');
        
        % 右图：在原RMS图上叠加二值化轮廓
        subplot(1, 2, 2);
        imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_compensated_interp);
        axis equal tight;
        colormap(gca, 'jet');
        colorbar;
        hold on;
        % 叠加二值化边界轮廓（白色）
        contour(x_interp * 1e3, y_interp * 1e3, double(binary_image), [0.5 0.5], ...
                'Color', 'white', 'LineWidth', 1.5);
        % 叠加高RMS区域的半透明红色遮罩
        overlay_rgb = cat(3, ones(size(binary_image)), zeros(size(binary_image)), zeros(size(binary_image)));
        h_overlay = imshow(overlay_rgb, 'XData', x_interp * 1e3, 'YData', y_interp * 1e3);
        set(h_overlay, 'AlphaData', double(binary_image) * 0.30);
        hold off;
        xlabel('X 位置 (mm)', 'FontSize', 12);
        ylabel('Y 位置 (mm)', 'FontSize', 12);
        title({'衰减补偿RMS图 + 二值化区域叠加', '(白色轮廓 + 红色半透明遮罩)'}, ...
              'FontSize', 13, 'FontWeight', 'bold');
        grid on;
        set(gca, 'YDir', 'normal');
        
        sgtitle(sprintf('二值化阈值处理 | %s | 阈值=%.4e | 高RMS占比=%.1f%%', ...
                        threshold_label, binarization_threshold_val, n_high/n_total*100), ...
                'FontSize', 12);
    else
        % 仅显示纯二值化图
        figure('Name', '二值化结果 (最终RMS图: 小波去噪 + 衰减补偿)', ...
               'Position', [150, 150, 700, 560]);
        imagesc(x_interp * 1e3, y_interp * 1e3, binary_image);
        axis equal tight;
        colormap([0.12 0.12 0.18; 0.98 0.45 0.15]);
        colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'低RMS (0)', '高RMS (1)'});
        xlabel('X 位置 (mm)', 'FontSize', 12);
        ylabel('Y 位置 (mm)', 'FontSize', 12);
        title({sprintf('二值化结果 | %s', threshold_label), ...
               sprintf('阈值=%.4e | 高RMS占比=%.1f%%', binarization_threshold_val, n_high/n_total*100)}, ...
              'FontSize', 13, 'FontWeight', 'bold');
        grid on;
        set(gca, 'YDir', 'normal');
    end
    
    fprintf('二值化处理完成\n');
end


%% ========== 本地函数 ========== %%

function rms_out = adaptiveGradientDenoise(rms_in, dist_map, sigma_thresh, n_bands, label)
% adaptiveGradientDenoise  基于幂律衰减参考模型的自适应梯度去噪
%
% 输入:
%   rms_in      - 待处理的 RMS 图像矩阵 [n_y x n_x]
%   dist_map    - 各点到波源的距离矩阵 [n_y x n_x]（与 rms_in 同尺寸）
%   sigma_thresh- 异常判定阈值（局部 sigma 的倍数），推荐 3.0
%   n_bands     - 距离分段数（用于估计局部 sigma），推荐 8
%   label       - 日志标签字符串
%
% 输出:
%   rms_out     - 去噪后的 RMS 图像矩阵

[n_y, n_x] = size(rms_in);
rms_out = rms_in;

% ---- 步骤1：Theil-Sen 鲁棒拟合幂律衰减参数 ----
% 模型: RMS(d) = A * d^(-beta)  =>  ln(RMS) = ln(A) - beta*ln(d)
% 只使用距离 > 0 且 RMS > 0 的有效点
valid_mask = (dist_map > 0) & (rms_in > 0) & isfinite(rms_in);
d_vec   = dist_map(valid_mask);   % 有效点距离向量
rms_vec = rms_in(valid_mask);     % 有效点RMS向量

ln_d   = log(d_vec);
ln_rms = log(rms_vec);

% Theil-Sen 中位数斜率估计
% 为加速计算，当点数 > 2000 时随机抽样 2000 个点对
n_pts = length(ln_d);
max_pairs = 2000;
if n_pts*(n_pts-1)/2 > max_pairs
    % 随机选 max_pairs 个点对
    rng_pairs = randi(n_pts, max_pairs, 2);
    same = (rng_pairs(:,1) == rng_pairs(:,2));
    rng_pairs(same, 2) = mod(rng_pairs(same, 2), n_pts) + 1;
    slopes = (ln_rms(rng_pairs(:,2)) - ln_rms(rng_pairs(:,1))) ./ ...
             (ln_d(rng_pairs(:,2))   - ln_d(rng_pairs(:,1)));
    slopes = slopes(isfinite(slopes));
else
    % 全量计算
    n_pairs = n_pts*(n_pts-1)/2;
    slopes = zeros(n_pairs, 1);
    k = 0;
    for p = 1:n_pts-1
        for q = p+1:n_pts
            dd = ln_d(q) - ln_d(p);
            if dd ~= 0
                k = k + 1;
                slopes(k) = (ln_rms(q) - ln_rms(p)) / dd;
            end
        end
    end
    slopes = slopes(1:k);
end

beta = -median(slopes);          % 幂律指数 (正值代表衰减)
% 用中位数估计截距: ln(A) = median(ln_rms + beta*ln_d)
ln_A = median(ln_rms + beta * ln_d);
A    = exp(ln_A);

fprintf('  [%s] 拟合结果: A = %.4e, β = %.4f\n', label, A, beta);

% ---- 步骤2：计算参考值和残差 ----
% 对距离为0的点（正好在波源处）直接用均值代替
ref_map = zeros(n_y, n_x);
ref_map(valid_mask)  = A .* d_vec .^ (-beta);
ref_map(~valid_mask) = mean(rms_vec);   % 兜底

residual_map = rms_in - ref_map;        % 实测 - 参考

% ---- 步骤3：按距离分段估计局部 sigma ----
d_min = min(dist_map(valid_mask));
d_max = max(dist_map(valid_mask));
band_edges    = linspace(d_min, d_max, n_bands + 1);
sigma_local_map = zeros(n_y, n_x);

for b = 1:n_bands
    if b < n_bands
        band_mask = valid_mask & (dist_map >= band_edges(b)) & (dist_map < band_edges(b+1));
    else
        band_mask = valid_mask & (dist_map >= band_edges(b)) & (dist_map <= band_edges(b+1));
    end
    if sum(band_mask(:)) > 3
        sigma_local_map(band_mask) = std(residual_map(band_mask));
    else
        % 本段点数太少，用全局残差 std 代替
        sigma_local_map(band_mask) = std(residual_map(valid_mask));
    end
end
% 无效点（距离=0等）也给个兜底 sigma
global_sigma = std(residual_map(valid_mask));
sigma_local_map(~valid_mask) = global_sigma;

% ---- 步骤4：检测并修复异常点 ----
% 归一化残差 > sigma_thresh 且 残差为正（即偏高）的点判为异常亮斑
norm_residual = residual_map ./ (sigma_local_map + eps);
outlier_mask  = (norm_residual > sigma_thresh) & valid_mask;

n_outliers = sum(outlier_mask(:));
fprintf('  [%s] 检测到异常亮斑点数: %d / %d (%.1f%%)\n', ...
        label, n_outliers, n_x*n_y, n_outliers/(n_x*n_y)*100);

if n_outliers > 0
    rms_work = rms_in;
    rms_work(outlier_mask) = NaN;
    
    for i = 1:n_y
        for j = 1:n_x
            if outlier_mask(i, j)
                i_min = max(1, i-2); i_max = min(n_y, i+2);
                j_min = max(1, j-2); j_max = min(n_x, j+2);
                neighbors = rms_work(i_min:i_max, j_min:j_max);
                valid_nb   = neighbors(~isnan(neighbors));
                if ~isempty(valid_nb)
                    rms_out(i, j) = median(valid_nb);
                else
                    % 若邻域全为异常点，退回参考模型值
                    rms_out(i, j) = ref_map(i, j);
                end
            end
        end
    end
end

end
