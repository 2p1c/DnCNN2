% filepath: /Users/zyt/ANW/5mm/dispersion_wavelet.m

%% 小波变换信号处理与频散曲线分析
% 本程序使用小波变换对单点信号进行去噪处理，并计算不同分解层数对频散曲线的影响

clear; clc; close all;

%% ========== 第一部分：数据预处理（与 dispersion.m 相同）==========

% 加载原始数据
load("/Volumes/ESD-ISO/数据/260125/Laser/23_25.mat"); % 包含变量 x (1×2500) 和 y (465×2500)

% 计算采样率和时间向量
data_time = x; % 时间向量
fs = 6.25e6; % 采样率 (Hz)

% 设置点阵参数 - 矩形点阵
n_cols = 23;      % x方向列数
n_rows = 25;    % y方向行数
spacing = 5e-4;  % 物理间距 0.5mm = 0.0005m

% 生成坐标向量
data_x = (0:n_cols-1) * spacing; % x方向坐标 (m)
data_y = (0:n_rows-1) * spacing; % y方向坐标 (m)

% 验证数据点数
total_points = n_cols * n_rows;
assert(size(y, 1) == total_points, ...
    sprintf('数据点数不匹配: 期望 %d×%d=%d, 实际 %d', ...
    n_cols, n_rows, total_points, size(y, 1)));

% 将蛇形扫描数据重塑为 n_cols×n_rows×2500 的三维数组
data_xyt = zeros(n_cols, n_rows, length(data_time));

for col = 1:n_cols
    start_idx = (col-1) * n_rows + 1;
    end_idx = col * n_rows;
    col_data = y(start_idx:end_idx, :);
    if mod(col, 2) == 0
        col_data = flipud(col_data); % 翻转偶数列
    end
    data_xyt(col, :, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d列 × %d行\n', n_cols, n_rows);
fprintf('  物理尺寸: %.2f mm × %.2f mm\n', ...
    (n_cols-1)*spacing*1e3, (n_rows-1)*spacing*1e3);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  数据形状: %s\n', mat2str(size(data_xyt)));

%% ========== 第二部分：单点信号小波分析 ==========

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('第二部分: 单点信号小波分析\n');
fprintf('%s\n', repmat('=', 1, 70));

% 随机选择一个点
rng(42);  % 设置随机种子保证可重现性
rand_x = randi(n_cols);
rand_y = randi(n_rows);

% 提取该点的时域信号
point_signal = squeeze(data_xyt(rand_x, rand_y, :));

fprintf('\n随机选择的点: (%d, %d)\n', rand_x, rand_y);
fprintf('物理坐标: (%.2f mm, %.2f mm)\n', data_x(rand_x)*1e3, data_y(rand_y)*1e3);

% 应用带通滤波
center_freq = 3e5;    % 中心频率 300 kHz
bandwidth = 4e5;      % 带宽 400 kHz
filter_order = 2;     % 滤波器阶数

filtered_signal = Filter.apply(point_signal, fs, center_freq, bandwidth, filter_order);

fprintf('\n滤波参数: 中心频率=%.0f kHz, 带宽=%.0f kHz\n', center_freq/1e3, bandwidth/1e3);

% ========== 2.1: 多个分解层数的小波去噪 ==========

fprintf('\n--- 小波去噪处理 ---\n');

wavelet_name = 'db4';      % Daubechies 4小波
levels_to_test = [1, 2, 3, 4];  % 要测试的分解层数
num_levels = length(levels_to_test);

% 存储不同分解层数的去噪结果
denoised_signals = cell(num_levels, 1);
denoising_params = struct();

for idx = 1:num_levels
    level = levels_to_test(idx);
    fprintf('  处理分解层数 = %d...\n', level);
    
    % 使用 'soft' 软阈值进行去噪
    denoised_signals{idx} = Filter.waveletDenoise(filtered_signal, wavelet_name, level, 'soft');
    
    % 计算去噪效果指标
    rms_original = rms(filtered_signal);
    rms_denoised = rms(denoised_signals{idx});
    
    denoising_params.(['level_', num2str(level)]) = struct(...
        'level', level, ...
        'rms_original', rms_original, ...
        'rms_denoised', rms_denoised, ...
        'energy_preserved', (sum(denoised_signals{idx}.^2) / sum(filtered_signal.^2)) * 100 ...
    );
    
    fprintf('    RMS: %.4e → %.4e (能量保留: %.2f%%)\n', ...
        rms_original, rms_denoised, denoising_params.(['level_', num2str(level)]).energy_preserved);
end

fprintf('  小波去噪完成\n');

% ========== 2.2: 多种阈值方法对比（用第2层为例） ==========

fprintf('\n--- 阈值方法对比 (分解层数=2) ---\n');

level_for_method_test = 2;
threshold_methods = {'soft', 'hard', 'rigrsure', 'sqtwolog'};
num_methods = length(threshold_methods);

denoised_methods = cell(num_methods, 1);

for m_idx = 1:num_methods
    method = threshold_methods{m_idx};
    fprintf('  测试阈值方法: %s...\n', method);
    
    denoised_methods{m_idx} = Filter.waveletDenoise(filtered_signal, wavelet_name, level_for_method_test, method);
    
    rms_denoised = rms(denoised_methods{m_idx});
    energy_preserved = (sum(denoised_methods{m_idx}.^2) / sum(filtered_signal.^2)) * 100;
    
    fprintf('    RMS: %.4e (能量保留: %.2f%%)\n', rms_denoised, energy_preserved);
end

fprintf('  阈值方法对比完成\n');

% ========== 2.3: 计算频谱 ==========

nfft = 2^nextpow2(length(point_signal));
freq_vector = (0:nfft-1) * fs / nfft;
half_idx = 1:nfft/2;
freq_vector_pos = freq_vector(half_idx);

% 原始信号频谱
spec_original = abs(fft(point_signal, nfft));
spec_original = spec_original(half_idx);

% 滤波后信号频谱
spec_filtered = abs(fft(filtered_signal, nfft));
spec_filtered = spec_filtered(half_idx);

% 不同分解层数的频谱
spec_denoised = cell(num_levels, 1);
for idx = 1:num_levels
    spec_denoised{idx} = abs(fft(denoised_signals{idx}, nfft));
    spec_denoised{idx} = spec_denoised{idx}(half_idx);
end

% 不同阈值方法的频谱
spec_methods = cell(num_methods, 1);
for m_idx = 1:num_methods
    spec_methods{m_idx} = abs(fft(denoised_methods{m_idx}, nfft));
    spec_methods{m_idx} = spec_methods{m_idx}(half_idx);
end

% ========== 2.4: 小波系数可视化 ==========

fprintf('\n--- 小波系数分析 ---\n');

% 进行小波多层分解（以第3层为例）
example_level = 3;
fprintf('  进行 %d 层小波分解用于可视化...\n', example_level);

[C, L] = wavedec(filtered_signal, example_level, wavelet_name);

% 提取各层系数
approx_coeff = appcoef(C, L, wavelet_name);
detail_coeffs = cell(example_level, 1);
for i = 1:example_level
    detail_coeffs{i} = detcoef(C, L, i);
end

fprintf('  分解完成: 近似系数长度=%d, 细节系数长度=%d\n', ...
    length(approx_coeff), length(detail_coeffs{1}));

% ========== 2.5: 时域波形可视化 ==========

fprintf('\n绘制时域波形对比图...\n');

figure('Name', '单点信号时域波形 - 多层小波去噪对比', ...
    'Position', [50, 400, 1400, 900]);

% 原始信号
subplot(3, 2, 1);
plot(data_time * 1e6, point_signal, 'b-', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('原始信号', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
xlim([data_time(1)*1e6, data_time(100)*1e6]);  % 显示前100个时间点

% 滤波后信号
subplot(3, 2, 2);
plot(data_time * 1e6, filtered_signal, 'r-', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title(sprintf('带通滤波 (%.0f-%.0f kHz)', (center_freq-bandwidth/2)/1e3, (center_freq+bandwidth/2)/1e3), ...
    'FontSize', 11, 'FontWeight', 'bold');
grid on;
xlim([data_time(1)*1e6, data_time(100)*1e6]);

% 不同分解层数的小波去噪结果
for idx = 1:num_levels
    subplot(3, 2, 2+idx);
    plot(data_time * 1e6, denoised_signals{idx}, 'g-', 'LineWidth', 1);
    xlabel('时间 (μs)', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    title(sprintf('小波去噪 (层数=%d, 软阈值)', levels_to_test(idx)), ...
        'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    xlim([data_time(1)*1e6, data_time(100)*1e6]);
end

sgtitle(sprintf('单点信号时域波形对比 - 位置: (%d, %d)', rand_x, rand_y), ...
    'FontSize', 13, 'FontWeight', 'bold');

% ========== 2.6: 频谱对比 ==========

fprintf('绘制频谱对比图...\n');

figure('Name', '单点信号频谱 - 多层小波去噪对比', ...
    'Position', [50, 50, 1400, 900]);

% 原始频谱
subplot(3, 2, 1);
plot(freq_vector_pos / 1e3, spec_original, 'b-', 'LineWidth', 1);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('原始信号频谱', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
xlim([0, 1000]);

% 滤波后频谱
subplot(3, 2, 2);
plot(freq_vector_pos / 1e3, spec_filtered, 'r-', 'LineWidth', 1);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('滤波后信号频谱', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
xlim([0, 1000]);

% 不同分解层数的频谱
for idx = 1:num_levels
    subplot(3, 2, 2+idx);
    plot(freq_vector_pos / 1e3, spec_denoised{idx}, 'g-', 'LineWidth', 1);
    xlabel('频率 (kHz)', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    title(sprintf('小波去噪 (层数=%d)', levels_to_test(idx)), ...
        'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    xlim([0, 1000]);
end

sgtitle('单点信号频谱对比', 'FontSize', 13, 'FontWeight', 'bold');

% ========== 2.7: 阈值方法对比 ==========

fprintf('绘制阈值方法对比图...\n');

figure('Name', '阈值方法对比 (层数=2)', ...
    'Position', [500, 400, 1400, 900]);

% 时域对比
for m_idx = 1:num_methods
    subplot(2, 4, m_idx);
    plot(data_time * 1e6, denoised_methods{m_idx}, 'LineWidth', 1);
    xlabel('时间 (μs)', 'FontSize', 9);
    ylabel('幅值', 'FontSize', 9);
    title(sprintf('时域: %s阈值', threshold_methods{m_idx}), 'FontSize', 10, 'FontWeight', 'bold');
    grid on;
    xlim([data_time(1)*1e6, data_time(100)*1e6]);
end

% 频域对比
for m_idx = 1:num_methods
    subplot(2, 4, 4+m_idx);
    plot(freq_vector_pos / 1e3, spec_methods{m_idx}, 'LineWidth', 1);
    xlabel('频率 (kHz)', 'FontSize', 9);
    ylabel('幅值', 'FontSize', 9);
    title(sprintf('频域: %s阈值', threshold_methods{m_idx}), 'FontSize', 10, 'FontWeight', 'bold');
    grid on;
    xlim([0, 1000]);
end

sgtitle(sprintf('阈值方法对比 (小波分解层数=%d)', level_for_method_test), ...
    'FontSize', 13, 'FontWeight', 'bold');

% ========== 2.8: 小波系数热力图 ==========

fprintf('绘制小波系数可视化图...\n');

figure('Name', sprintf('小波系数分析 (层数=%d)', example_level), ...
    'Position', [950, 50, 1200, 900]);

% 近似系数
subplot(2, 2, 1);
plot(approx_coeff, 'b-', 'LineWidth', 1);
xlabel('样本点', 'FontSize', 10);
ylabel('系数值', 'FontSize', 10);
title(sprintf('近似系数 cA%d (长度=%d)', example_level, length(approx_coeff)), ...
    'FontSize', 11, 'FontWeight', 'bold');
grid on;

% 细节系数热力图
subplot(2, 2, 2);
detail_matrix = [];
for i = 1:example_level
    % 通过插值使所有细节系数长度相同以便可视化
    if i == 1
        max_len = length(detail_coeffs{i});
    end
    detail_matrix = [detail_matrix; detail_coeffs{i}(1:min(end, max_len))];
end
imagesc(detail_matrix);
colorbar;
ylabel('分解层级', 'FontSize', 10);
xlabel('样本点', 'FontSize', 10);
title('细节系数热力图 (cD1-cD3)', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'YTickLabel', {sprintf('cD%d', example_level), ...
    sprintf('cD%d', example_level-1), sprintf('cD%d', 1)});

% 各层细节系数幅值谱
subplot(2, 2, 3);
hold on;
colors = {'r', 'g', 'b', 'k'};
for i = 1:example_level
    detail_spec = abs(fft(detail_coeffs{i}, 1024));
    freq_detail = (0:511) * fs / 1024;
    plot(freq_detail / 1e3, detail_spec(1:512), colors{i}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('cD%d', i));
end
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('细节系数频谱分析', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
xlim([0, 2000]);

% 能量分布
subplot(2, 2, 4);
energies = [];
labels_energy = {};
energy_approx = sum(approx_coeff.^2);
energies = energy_approx;
labels_energy = {sprintf('cA%d', example_level)};

for i = 1:example_level
    energies = [energies, sum(detail_coeffs{i}.^2)];
    labels_energy = [labels_energy, {sprintf('cD%d', i)}];
end

bar(energies, 'FaceColor', 'c', 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', labels_energy);
ylabel('能量 (信号平方和)', 'FontSize', 10);
title('各层系数能量分布', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

sgtitle(sprintf('小波系数详细分析 (小波=%s, 层数=%d)', wavelet_name, example_level), ...
    'FontSize', 13, 'FontWeight', 'bold');

%% ========== 第三部分：整列数据频散曲线计算 ==========

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('第三部分: 整列数据频散曲线计算\n');
fprintf('%s\n', repmat('=', 1, 70));

% 提取中间列数据
if n_cols == 1
    middle_col_index = 1;
else
    middle_col_index = ceil(n_cols / 2);
end

data_yt = permute(data_xyt(middle_col_index, :, :), [2, 3, 1]);  % [y空间 × 时间]

fprintf('\n使用第 %d 列数据 (共 %d 列)\n', middle_col_index, n_cols);
fprintf('分析y方向的波传播特性\n');

% ========== 3.1: 对不同分解层数进行频散曲线计算 ==========

fprintf('\n--- 计算不同分解层数的频散曲线 ---\n');

levels_for_dispersion = [1, 2, 3, 4];  % 要计算的分解层数
num_levels_disp = length(levels_for_dispersion);

% 存储结果
dispersion_data = struct();

for lv_idx = 1:num_levels_disp
    level = levels_for_dispersion(lv_idx);
    fprintf('  处理分解层数 = %d...\n', level);
    
    % 滤波
    data_yt_filtered = zeros(size(data_yt));
    for i = 1:n_rows
        data_yt_filtered(i, :) = Filter.apply(data_yt(i, :), fs, center_freq, bandwidth, filter_order);
    end
    
    % 小波去噪
    for i = 1:n_rows
        data_yt_filtered(i, :) = Filter.waveletDenoise(data_yt_filtered(i, :), wavelet_name, level, 'soft');
    end
    
    % FFT参数
    nfft_space = 2^(nextpow2(n_rows) + 1);
    nfft_time = 2^(nextpow2(length(data_time)) + 1);
    
    % 二维傅里叶变换
    kf_spectrum = fftn(data_yt_filtered, [nfft_space, nfft_time]);
    kf_shifted = fftshift(kf_spectrum, 1);
    
    % 频率和波数向量
    freq_vector_full = (0:nfft_time-1) * fs / nfft_time;
    delta_y = data_y(2) - data_y(1);
    ky_vector = ((-round(nfft_space/2) + 1 : round(nfft_space/2)) / nfft_space) * 2*pi / delta_y;
    
    % 截取感兴趣的频率范围
    max_freq = 1e6;
    [~, freq_max_index] = min(abs(freq_vector_full - max_freq));
    
    % 提取幅值谱
    amp = abs(kf_shifted(:, 1:freq_max_index));
    
    % 按频率逐列归一化
    max_amp = max(amp, [], 1);
    max_amp(max_amp == 0) = 1;
    amp = amp ./ max_amp;
    
    % 存储结果
    dispersion_data.(['level_', num2str(level)]) = struct(...
        'amp', amp, ...
        'freq', freq_vector_full(1:freq_max_index), ...
        'ky', ky_vector, ...
        'level', level ...
    );
    
    fprintf('    频散曲线计算完成\n');
end

fprintf('所有分解层数的频散曲线计算完成\n');

% filepath: /Users/zyt/ANW/5mm/dispersion_wavelet.m

% ...existing code...

% ========== 3.2: 绘制频散曲线对比 ==========

fprintf('绘制频散曲线对比图...\n');

figure('Name', '频散曲线 - 不同小波分解层数对比', ...
    'Position', [100, 50, 1600, 1000]);

for idx = 1:num_levels_disp
    level = levels_for_dispersion(idx);
    data_level = dispersion_data.(['level_', num2str(level)]);
    
    subplot(2, 2, idx);
    surf(data_level.freq / 1e3, -data_level.ky / 1e3, data_level.amp);
    shading interp;
    colormap(jet);
    colorbar;
    view([0, 90]);
    xlabel('频率 (kHz)', 'FontSize', 11);
    ylabel('波数 (rad/mm)', 'FontSize', 11);
    title(sprintf('小波分解层数 = %d', level), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    zlim([0, 1]);
end

sgtitle('频散曲线：不同小波分解层数对比 (按频率归一化)', ...
    'FontSize', 14, 'FontWeight', 'bold');

% ========== 3.2.5: 绘制未归一化的频散曲线对比 ==========

fprintf('计算未归一化的频散曲线...\n');

% 重新计算所有分解层数的频散曲线，但保留原始未归一化的幅值
dispersion_data_unnormalized = struct();

for lv_idx = 1:num_levels_disp
    level = levels_for_dispersion(lv_idx);
    fprintf('  处理未归一化数据 - 分解层数 = %d...\n', level);
    
    % 滤波
    data_yt_filtered = zeros(size(data_yt));
    for i = 1:n_rows
        data_yt_filtered(i, :) = Filter.apply(data_yt(i, :), fs, center_freq, bandwidth, filter_order);
    end
    
    % 小波去噪
    for i = 1:n_rows
        data_yt_filtered(i, :) = Filter.waveletDenoise(data_yt_filtered(i, :), wavelet_name, level, 'soft');
    end
    
    % FFT参数
    nfft_space = 2^(nextpow2(n_rows) + 1);
    nfft_time = 2^(nextpow2(length(data_time)) + 1);
    
    % 二维傅里叶变换
    kf_spectrum = fftn(data_yt_filtered, [nfft_space, nfft_time]);
    kf_shifted = fftshift(kf_spectrum, 1);
    
    % 频率和波数向量
    freq_vector_full = (0:nfft_time-1) * fs / nfft_time;
    delta_y = data_y(2) - data_y(1);
    ky_vector = ((-round(nfft_space/2) + 1 : round(nfft_space/2)) / nfft_space) * 2*pi / delta_y;
    
    % 截取感兴趣的频率范围
    max_freq = 1e6;
    [~, freq_max_index] = min(abs(freq_vector_full - max_freq));
    
    % 提取幅值谱 - 未进行任何归一化
    amp_unnorm = abs(kf_shifted(:, 1:freq_max_index));
    
    % 存储未归一化的结果
    dispersion_data_unnormalized.(['level_', num2str(level)]) = struct(...
        'amp', amp_unnorm, ...
        'freq', freq_vector_full(1:freq_max_index), ...
        'ky', ky_vector, ...
        'level', level ...
    );
    
    fprintf('    未归一化频散曲线计算完成\n');
end

% 计算全局最大值（用于统一颜色条和z轴限制）
global_max = 0;
for idx = 1:num_levels_disp
    level = levels_for_dispersion(idx);
    amp_data = dispersion_data_unnormalized.(['level_', num2str(level)]).amp;
    global_max = max(global_max, max(amp_data(:)));
end

fprintf('未归一化全局最大值: %.4e\n', global_max);

% 绘制未归一化的频散曲线对比（全局归一化）
fprintf('绘制未归一化频散曲线对比图...\n');

figure('Name', '频散曲线 - 未归一化对比 (全局归一化)', ...
    'Position', [100, 50, 1600, 1000]);

for idx = 1:num_levels_disp
    level = levels_for_dispersion(idx);
    data_level = dispersion_data_unnormalized.(['level_', num2str(level)]);
    
    % 全局归一化：用全局最大值归一化所有数据
    amp_globally_normalized = data_level.amp / global_max;
    
    subplot(2, 2, idx);
    surf(data_level.freq / 1e3, -data_level.ky / 1e3, amp_globally_normalized);
    shading interp;
    colormap(jet);
    colorbar;
    view([0, 90]);
    xlabel('频率 (kHz)', 'FontSize', 11);
    ylabel('波数 (rad/mm)', 'FontSize', 11);
    title(sprintf('小波分解层数 = %d', level), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    zlim([0, 1]);  % 统一的z轴限制
end

sgtitle('频散曲线：不同小波分解层数对比 (未归一化 - 全局归一化显示)', ...
    'FontSize', 14, 'FontWeight', 'bold');

% ========== 3.3: 频散曲线对比总结 ==========

fprintf('\n%s\n', repmat('-', 1, 70));
fprintf('频散曲线对比总结\n');
fprintf('%s\n', repmat('-', 1, 70));

fprintf('归一化方式对比:\n');
fprintf('  按频率归一化: 每一频率独立归一化，突出同频率下的相对模态分布\n');
fprintf('    - 物理意义: 哪些波数在该频率最强\n');
fprintf('    - 优点: 突出每个频率的导波模态\n');
fprintf('    - 缺点: 看不出不同频率间的能量差异\n\n');
fprintf('  未归一化(全局归一化): 保留不同频率的绝对能量差异\n');
fprintf('    - 物理意义: 哪些频率的总能量最大\n');
fprintf('    - 优点: 反映真实的能量分布，看出主要频率范围\n');
fprintf('    - 缺点: 低能量频率的细节可能不清晰\n\n');
fprintf('不同分解层数对频散曲线的影响:\n');
fprintf('  层数=1: 去噪效果弱，保留更多细节信息和噪声\n');
fprintf('  层数=2: 在细节和去噪之间取得较好平衡\n');
fprintf('  层数=3: 去噪效果显著，可能损失部分有用信息\n');
fprintf('  层数=4: 去噪效果最强，但可能过度平滑信号\n');
fprintf('\n推荐: 根据实际噪声水平和分析目的选择:\n');
fprintf('  - 观察导波模态分布 → 使用按频率归一化版本\n');
fprintf('  - 观察能量分布和主频率范围 → 使用未归一化版本\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('程序执行完成！\n');
fprintf('已生成以下图形:\n');
fprintf('  1. 单点信号时域波形 - 多层小波去噪对比\n');
fprintf('  2. 单点信号频谱 - 多层小波去噪对比\n');
fprintf('  3. 阈值方法对比 (层数=2)\n');
fprintf('  4. 小波系数分析 (层数=%d)\n', example_level);
fprintf('  5. 频散曲线 - 按频率归一化对比\n');
fprintf('  6. 频散曲线 - 未归一化(全局归一化)对比 ⭐新增\n');
fprintf('%s\n', repmat('=', 1, 70));