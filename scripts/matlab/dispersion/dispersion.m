%% 数据预处理：加载并重塑蛇形扫描数据
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
% 扫描方式: 先扫描y方向(列方向),再移动到下一列x方向
data_xyt = zeros(n_cols, n_rows, length(data_time));

for col = 1:n_cols
    % 计算当前列在y数组中的起始和结束索引
    start_idx = (col-1) * n_rows + 1;
    end_idx = col * n_rows;
    
    % 提取当前列的数据
    col_data = y(start_idx:end_idx, :);
    
    % 根据列数决定是否翻转（偶数列从下到上扫描）
    if mod(col, 2) == 0
        col_data = flipud(col_data); % 翻转偶数列
    end
    
    % 存储到三维数组中: data_xyt(x位置, y位置, 时间)
    data_xyt(col, :, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d列 × %d行\n', n_cols, n_rows);
fprintf('  物理尺寸: %.2f mm × %.2f mm\n', ...
    (n_cols-1)*spacing*1e3, (n_rows-1)*spacing*1e3);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  数据形状: %s\n', mat2str(size(data_xyt)));

%% 随机选择一个点并分析其时域和频域特性
% 随机选择点的坐标
rand_x = randi(n_cols);
rand_y = randi(n_rows);

% 提取该点的时域信号
point_signal = squeeze(data_xyt(rand_x, rand_y, :));

% 计算频谱
nfft = 2^nextpow2(length(point_signal)); % FFT点数
freq_spectrum = fft(point_signal, nfft);
freq_vector = (0:nfft-1) * fs / nfft; % 频率向量

% 只保留正频率部分
half_idx = 1:nfft/2;
freq_vector_pos = freq_vector(half_idx);
amplitude_spectrum = abs(freq_spectrum(half_idx));

fprintf('\n随机点分析:\n');
fprintf('  选择的点: (%d, %d)\n', rand_x, rand_y);
fprintf('  物理坐标: (%.2f mm, %.2f mm)\n', ...
    data_x(rand_x)*1e3, data_y(rand_y)*1e3);
fprintf('  信号RMS: %.4e\n', rms(point_signal));
fprintf('  信号峰值: %.4e\n', max(abs(point_signal)));
[~, max_freq_idx] = max(amplitude_spectrum);
fprintf('  主频: %.2f MHz\n', freq_vector_pos(max_freq_idx)/1e6);

%% 对随机点信号施加滤波并对比
% 设置滤波参数
center_freq = 3e5;    % 中心频率 300 kHz
bandwidth = 4e5;      % 带宽 400 kHz
filter_order = 2;     % 滤波器阶数

% 显示滤波器信息
fprintf('\n滤波处理:\n');
Filter.printInfo(center_freq, bandwidth, filter_order, fs);

% 应用带通滤波器
filtered_signal = Filter.apply(point_signal, fs, center_freq, bandwidth, filter_order);

% 计算滤波后的频谱
filtered_spectrum = fft(filtered_signal, nfft);
filtered_amplitude = abs(filtered_spectrum(half_idx));

% 计算滤波效果指标
original_energy = sum(point_signal.^2);
filtered_energy = sum(filtered_signal.^2);
energy_ratio = filtered_energy / original_energy * 100;

fprintf('  滤波后信号能量保留: %.2f%%\n', energy_ratio);
fprintf('  滤波后信号RMS: %.4e\n', rms(filtered_signal));
fprintf('  滤波后信号峰值: %.4e\n', max(abs(filtered_signal)));

% 可视化:滤波前后对比
figure('Name', '滤波前后信号对比', 'Position', [100, 100, 1400, 800]);

% 时域信号对比
subplot(2, 2, 1);
plot(data_time * 1e6, point_signal, 'b-', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title(sprintf('原始时域信号 - 位置: (%d, %d)', rand_x, rand_y), 'FontSize', 12);
grid on;
legend('原始信号', 'Location', 'best');

subplot(2, 2, 2);
plot(data_time * 1e6, filtered_signal, 'r-', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title('滤波后时域信号', 'FontSize', 12);
grid on;
legend('滤波信号', 'Location', 'best');

% 频域信号对比
subplot(2, 2, 3);
plot(freq_vector_pos / 1e6, amplitude_spectrum, 'b-', 'LineWidth', 1);
hold on;
% 标注通带范围
lowcut = (center_freq - bandwidth/2) / 1e6;
highcut = (center_freq + bandwidth/2) / 1e6;
xline(lowcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', lowcut));
xline(highcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', highcut));
hold off;
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title('原始频谱', 'FontSize', 12);
grid on;
xlim([0, fs/6/1e6]);
legend('原始频谱', 'Location', 'best');

subplot(2, 2, 4);
plot(freq_vector_pos / 1e6, filtered_amplitude, 'r-', 'LineWidth', 1);
hold on;
xline(lowcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', lowcut));
xline(highcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', highcut));
hold off;
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title('滤波后频谱', 'FontSize', 12);
grid on;
xlim([0, fs/6/1e6]);
legend('滤波频谱', 'Location', 'best');

% 添加总标题
sgtitle('随机点信号滤波前后对比分析', 'FontSize', 14, 'FontWeight', 'bold');

%% 频散曲线计算（f-k域分析）
% 1. 提取中间一行数据进行空间-时间二维分析
% 根据实际列数选择合适的列
if n_cols == 1
    middle_col_index = 1;  % 只有1列时使用第1列
else
    middle_col_index = ceil(n_cols / 2);  % 多列时选择中间列
end

data_yt = permute(data_xyt(middle_col_index, :, :), [2, 3, 1]);  % [y空间 × 时间]

fprintf('\n频散曲线计算:\n');
fprintf('  使用第 %d 列数据 (共 %d 列)\n', middle_col_index, n_cols);
fprintf('  分析y方向的波传播特性\n');

% 2. 对整列数据应用滤波
fprintf('  对整列数据应用滤波...\n');
data_yt_filtered = zeros(size(data_yt));
for i = 1:n_rows
    data_yt_filtered(i, :) = Filter.apply(data_yt(i, :), fs, center_freq, bandwidth, filter_order);
end
fprintf('  滤波完成\n');

% 2.5 应用小波去噪
wavelet_name = 'db4';      % Daubechies 4小波
wavelet_level = 1;         % 分解层数
threshold_method = 'soft'; % 软阈值

fprintf('\n  应用小波去噪...\n');
Filter.printWaveletInfo(wavelet_name, wavelet_level, threshold_method);

for i = 1:n_rows
    data_yt_filtered(i, :) = Filter.waveletDenoise(data_yt_filtered(i, :), wavelet_name, wavelet_level, threshold_method);
end
fprintf('  小波去噪完成\n');

% 2.6 应用 SVD 降噪 (新增：提取相干波动特征)
fprintf('  应用 SVD 奇异值分解降噪...\n');
[U, S, V] = svd(data_yt_filtered, 'econ');
s = diag(S);
% 保留前 k 个奇异值能量最高的成分 (超声波信号的前 3-5 个分量通常包含 90% 以上能量)
k_keep = min(5, length(s));
S_clean = zeros(size(S));
S_clean(1:k_keep, 1:k_keep) = S(1:k_keep, 1:k_keep);
data_yt_filtered = U * S_clean * V';

% 3. 设置FFT参数
% 使用零填充提高分辨率
nfft_space = 2^(nextpow2(n_rows) + 1);          % 空间维度FFT点数
nfft_time = 2^(nextpow2(length(data_time)) + 1); % 时间维度FFT点数

fprintf('  FFT点数: 空间=%d, 时间=%d\n', nfft_space, nfft_time);

% 4. 对原始数据和滤波数据分别进行二维傅里叶变换
% 原始数据
kf_spectrum_original = fftn(data_yt, [nfft_space, nfft_time]);
kf_shifted_original = fftshift(kf_spectrum_original, 1);

% 滤波数据
kf_spectrum_filtered = fftn(data_yt_filtered, [nfft_space, nfft_time]);
kf_shifted_filtered = fftshift(kf_spectrum_filtered, 1);

% 5. 生成频率和波数向量
% 频率向量 (Hz)
freq_vector_full = (0:nfft_time-1) * fs / nfft_time;

% 波数向量 (rad/m) - 沿y方向
delta_y = data_y(2) - data_y(1);  % y方向空间采样间隔
ky_vector = ((-round(nfft_space/2) + 1 : round(nfft_space/2)) / nfft_space) ...
            * 2*pi / delta_y;

% 6. 选择感兴趣的频率范围 (0 到 1 MHz)
max_freq = 1e6;  % 最大显示频率 (Hz)
[~, freq_max_index] = min(abs(freq_vector_full - max_freq));

% 截取数据
data_kf_original = kf_shifted_original(:, 1:freq_max_index);
data_kf_filtered = kf_shifted_filtered(:, 1:freq_max_index);
freq_display = freq_vector_full(1:freq_max_index);
ky_display = ky_vector;

fprintf('  显示频率范围: 0 - %.2f MHz\n', max_freq/1e6);
fprintf('  波数范围: %.2f - %.2f rad/mm\n', min(ky_vector)/1e3, max(ky_vector)/1e3);

% 7. 计算幅值谱
amp_original = abs(data_kf_original);
amp_filtered = abs(data_kf_filtered);

% 按频率逐列归一化（基础方法）
% 矩阵结构为 A(k, f)，对每一个固定频率(列)，使用最大值归一化该频率下的所有波数幅值
% 这有助于突出每个频率下的主要导波模态，忽略不同频率间的能量差异
max_original = max(amp_original, [], 1);
max_original(max_original == 0) = 1; % 防止除以零
amp_original = amp_original ./ max_original;

max_filtered = max(amp_filtered, [], 1);
max_filtered(max_filtered == 0) = 1; % 防止除以零
amp_filtered = amp_filtered ./ max_filtered;

%% 多种增强方法对比
fprintf('\n频散曲线增强处理:\n');

% 方法1: 对数尺度（dB）
% 原理: 压缩动态范围，让弱信号也能显示
amp_method1 = 20*log10(amp_filtered + eps);  % 加eps防止log(0)
amp_method1 = amp_method1 - min(amp_method1(:));  % 平移到非负
amp_method1 = amp_method1 / max(amp_method1(:));  % 归一化到[0,1]
fprintf('  方法1: 对数尺度(dB) - 压缩动态范围\n');

% 方法2: 自适应阈值
% 原理: 对每个频率单独设置阈值，去除背景噪声
amp_method2 = amp_filtered;
threshold_factor = 1.5;  % 阈值因子（可调节，越大越严格）
for i = 1:size(amp_filtered, 2)
    col = amp_filtered(:, i);
    % 使用中位数+倍数*标准差作为阈值
    threshold = median(col) + threshold_factor * std(col);
    col(col < threshold) = 0;
    amp_method2(:, i) = col;
end
% 重新归一化
max_method2 = max(amp_method2, [], 1);
max_method2(max_method2 == 0) = 1;
amp_method2 = amp_method2 ./ max_method2;
fprintf('  方法2: 自适应阈值 - 去除背景噪声 (阈值因子=%.1f)\n', threshold_factor);

% 方法3: 能量占比筛选
% 原理: 只保留贡献了主要能量的波数分量
amp_method3 = amp_filtered;
energy_threshold = 0.3;  % 能量阈值（保留幅值>最大值*阈值的分量）
for i = 1:size(amp_filtered, 2)
    col = amp_filtered(:, i);
    max_val = max(col);
    col(col < max_val * energy_threshold) = 0;
    amp_method3(:, i) = col;
end
fprintf('  方法3: 能量占比筛选 - 保留主要模态 (阈值=%.0f%%)\n', energy_threshold*100);

% 方法4: 对数尺度 + 自适应阈值（组合方法）
% 原理: 先用对数压缩，再用阈值去噪
amp_temp = 20*log10(amp_filtered + eps);
amp_temp = amp_temp - min(amp_temp(:));
amp_method4 = zeros(size(amp_temp));
threshold_factor_combined = 1;  % 组合方法的阈值因子
for i = 1:size(amp_temp, 2)
    col = amp_temp(:, i);
    threshold = median(col) + threshold_factor_combined * std(col);
    col(col < threshold) = 0;
    amp_method4(:, i) = col;
end
% 归一化
max_method4 = max(amp_method4, [], 1);
max_method4(max_method4 == 0) = 1;
amp_method4 = amp_method4 ./ max_method4;
fprintf('  方法4: 对数+阈值组合 - 综合增强 (阈值因子=%.1f)\n', threshold_factor_combined);

% 方法5: 平方增强 + 阈值
% 原理: 平方运算突出强信号，抑制弱信号
amp_method5 = amp_filtered.^2;  % 平方增强
% 归一化
max_method5 = max(amp_method5, [], 1);
max_method5(max_method5 == 0) = 1;
amp_method5 = amp_method5 ./ max_method5;
% 应用阈值
for i = 1:size(amp_method5, 2)
    col = amp_method5(:, i);
    threshold = 0.1;  % 简单的固定阈值
    col(col < threshold) = 0;
    amp_method5(:, i) = col;
end
fprintf('  方法5: 平方增强+阈值 - 突出强信号\n');

% 方法6: 对数+阈值+形态学处理（利用连续性）
% 原理: 导波模态在f-k域是连续曲线，噪声是随机分布的孤立点
% 先应用方法4的结果，再进行形态学增强
amp_method6 = amp_method4;  % 从方法4开始

% 二值化：将数据转换为0-1
threshold_binary = 0.15;  % 二值化阈值（可调节）
amp_binary = amp_method6 > threshold_binary;

% 形态学处理：去除孤立噪声点，保留连续结构
% 定义结构元素：椭圆形，沿频率方向（横向）拉长
se_size_k = 3;    % 波数方向尺寸（纵向）
se_size_f = 5;    % 频率方向尺寸（横向），更大以捕捉连续性
se = strel('rectangle', [se_size_k, se_size_f]);

% 形态学开运算：先腐蚀后膨胀，去除小的孤立噪声点
amp_opened = imopen(amp_binary, se);

% 形态学闭运算：先膨胀后腐蚀，连接断裂的模态
amp_closed = imclose(amp_opened, se);

% 连通域分析：只保留较大的连通区域（真正的模态）
min_area = 1;  % 最小连通区域面积（像素数）
amp_cleaned = bwareaopen(amp_closed, min_area);

% 将二值掩模应用到原始幅值数据
amp_method6 = amp_method6 .* double(amp_cleaned);

% 可选：沿频率方向进行中值滤波，进一步平滑
median_filter_size = [3, 5];  % [k方向, f方向]
for i = 1:size(amp_method6, 1)
    amp_method6(i, :) = medfilt1(amp_method6(i, :), median_filter_size(2));
end

% 重新归一化
max_method6 = max(amp_method6, [], 1);
max_method6(max_method6 == 0) = 1;
amp_method6 = amp_method6 ./ max_method6;

fprintf('  方法6: 对数+阈值+形态学 - 利用连续性增强 ⭐⭐推荐\n');
fprintf('    形态学参数: 结构元素[%d×%d], 最小区域=%d像素\n', ...
    se_size_k, se_size_f, min_area);

% 8. 绘制频散曲线对比 (滤波前后)
figure('Name', '频散曲线对比', 'Position', [100, 100, 1400, 600]);

% 原始频散曲线
subplot(1, 2, 1);
surf(freq_display/1e3, -ky_vector/1e3, amp_original);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);  % 俯视图
xlabel('频率 (kHz)', 'FontSize', 12);
ylabel('波数 (rad/mm)', 'FontSize', 12);
title('原始频散曲线 (按频率归一化)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% 滤波后频散曲线
subplot(1, 2, 2);
surf(freq_display/1e3, -ky_vector/1e3, amp_filtered);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);  % 俯视图
xlabel('频率 (kHz)', 'FontSize', 12);
ylabel('波数 (rad/mm)', 'FontSize', 12);
title(sprintf('滤波后频散曲线 (按频率归一化, %.0f-%.0f kHz)', lowcut*1e3, highcut*1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;

% 添加总标题
sgtitle('频散曲线滤波前后对比', ...
        'FontSize', 15, 'FontWeight', 'bold');

%% 9. 绘制多种增强方法对比
figure('Name', '频散曲线增强方法对比', 'Position', [50, 50, 1600, 1000]);

% 方法0: 基础归一化（参考）
subplot(3, 2, 1);
surf(freq_display/1e3, -ky_vector/1e3, amp_filtered);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('波数 (rad/mm)', 'FontSize', 10);
title('基础: 按频率归一化', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

% 方法1: 对数尺度
subplot(3, 2, 2);
surf(freq_display/1e3, -ky_vector/1e3, amp_method1);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('波数 (rad/mm)', 'FontSize', 10);
title('方法1: 对数尺度(dB)', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

% 方法2: 自适应阈值
subplot(3, 2, 3);
surf(freq_display/1e3, -ky_vector/1e3, amp_method2);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('波数 (rad/mm)', 'FontSize', 10);
title(sprintf('方法2: 自适应阈值 (因子=%.1f)', threshold_factor), 'FontSize', 11, 'FontWeight', 'bold');
grid on;

% 方法3: 能量占比筛选
subplot(3, 2, 4);
surf(freq_display/1e3, -ky_vector/1e3, amp_method3);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('波数 (rad/mm)', 'FontSize', 10);
title(sprintf('方法3: 能量占比 (阈值=%.0f%%)', energy_threshold*100), 'FontSize', 11, 'FontWeight', 'bold');
grid on;

% 方法4: 对数+阈值组合
subplot(3, 2, 5);
surf(freq_display/1e3, -ky_vector/1e3, amp_method4);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('波数 (rad/mm)', 'FontSize', 10);
title('方法4: 对数+阈值组合 ⭐', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.8, 0.4, 0]);
grid on;

% 方法6: 对数+阈值+形态学（利用连续性）
subplot(3, 2, 6);
surf(freq_display/1e3, -ky_vector/1e3, amp_method6);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);
xlabel('频率 (kHz)', 'FontSize', 10);
ylabel('波数 (rad/mm)', 'FontSize', 10);
title('方法6: 形态学增强(连续性) ⭐⭐最佳', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r');
grid on;

% 添加总标题
sgtitle('频散曲线增强方法对比 - 突出导波模态', ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n可视化完成！请对比不同方法的效果，选择最适合的方法。\n');
fprintf('推荐: 方法6（形态学增强）利用了导波模态的连续性特征，效果最好\n');
fprintf('备选: 方法4（对数+阈值）计算简单，效果也不错\n');