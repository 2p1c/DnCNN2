%% 读取COMSOL导出的超声信号数据
% 该脚本读取txt文件中的COMSOL超声模拟结果，并转换为dispersion.m可以读取的mat格式
% 
% 输入: 点源有缺陷.txt (COMSOL导出文件)
% 输出: processed_data.mat (包含x和y变量)

clear; close all; clc;

%% 1. 读取文件
fprintf('正在读取COMSOL数据文件...\n');
filename = "C:\Users\123\Desktop\无损_铝板表面.txt";

% 打开文件
fid = fopen(filename, 'r');
if fid == -1
    error('无法打开文件: %s', filename);
end

% 跳过前8行注释
for i = 1:8
    fgetl(fid);
end

% 读取第9行（列头行，以%开头）
header_line = fgetl(fid);
fprintf('列头行: %s\n', header_line(1:min(100, length(header_line))));

% 解析列头以提取时间信息
% 列头格式: "% X    Y    solid.disp (mm) @ t=0  solid.disp (mm) @ t=1  ..."
% 使用正则表达式提取时间值 t=xxx
time_pattern = 't=(\d+(?:\.\d+)?)';
time_matches = regexp(header_line, time_pattern, 'tokens');
n_time_steps = length(time_matches);

if n_time_steps == 0
    error('未能从列头中提取时间步信息。列头内容: %s', header_line(1:min(200, length(header_line))));
end

fprintf('检测到 %d 个时间步\n', n_time_steps);

% 提取时间向量 (单位: ns，需要转换为秒)
time_values = zeros(1, n_time_steps);
for i = 1:n_time_steps
    time_values(i) = str2double(time_matches{i}{1});
end

% 确定时间单位并转换为秒
% 根据用户描述: range(0, 1[ns], 200[ns]) range(200[ns]+50[ns], t_total)
% 所以时间值是纳秒(ns)
time_ns = time_values;  % 时间 (ns)
x_time_s = time_ns * 1e-9;  % 转换为秒
x_time_us = time_ns * 1e-3;  % 转换为微秒(用于显示)

% 手动设置采样频率 (Hz)
fs_manual = 6.25e6;  % 6.25 MHz

fprintf('时间范围: %.1f - %.1f ns (%.3f - %.3f μs)\n', ...
    min(time_ns), max(time_ns), min(time_ns)*1e-3, max(time_ns)*1e-3);
fprintf('采样频率(手动设置): %.2f MHz\n', fs_manual/1e6);

% 准备数据存储
data_points = [];
all_signals = [];

% 读取数据行
line_count = 10;
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(strtrim(line))
        % 分割数据
        values = strsplit(strtrim(line));
        
        % 提取第一个和第二个值(X和Y坐标)
        if length(values) >= 2 + n_time_steps
            x_coord = str2double(values{1});
            y_coord = str2double(values{2});
            
            % 提取位移数据(从第3列开始)
            displacements = zeros(1, n_time_steps);
            for i = 1:n_time_steps
                displacements(i) = str2double(values{2 + i});
            end
            
            data_points = [data_points; x_coord, y_coord];
            all_signals = [all_signals; displacements];
            
            line_count = line_count + 1;
        end
    end
end
fclose(fid);

n_points = size(data_points, 1);
fprintf('成功读取 %d 个数据点\n', n_points);

%% 2. 数据整理与验证
fprintf('\n数据信息:\n');
fprintf('  数据点数: %d\n', n_points);
fprintf('  时间步数: %d\n', n_time_steps);
fprintf('  数据形状: [%d × %d]\n', size(all_signals, 1), size(all_signals, 2));

% 检查X坐标信息
unique_x = unique(data_points(:, 1));
unique_y = unique(data_points(:, 2));
fprintf('  X坐标: %.1f - %.1f (共 %d 个)\n', min(unique_x), max(unique_x), length(unique_x));
fprintf('  Y坐标: %.1f - %.1f (共 %d 个)\n', min(unique_y), max(unique_y), length(unique_y));

%% 3. 准备dispersion.m格式的输出
% dispersion.m期望的变量:
% - x: 时间向量 (1 × n_time_steps)
% - y: 位移数据矩阵 (n_points × n_time_steps)

x = x_time_s;  % 时间向量 (秒)
y = all_signals;  % 位移数据矩阵 (mm)

fprintf('\n准备保存数据:\n');
fprintf('  x (时间向量): 1 × %d, 范围: %.3e - %.3e s\n', ...
    length(x), min(x), max(x));
fprintf('  y (位移数据): %d × %d, 范围: %.3e - %.3e mm\n', ...
    size(y, 1), size(y, 2), min(y(:)), max(y(:)));

%% 4. 可视化预览
figure('Name', '数据预览', 'Position', [100, 100, 1400, 600]);

% 显示所有信号的均方根(RMS)
subplot(1, 3, 1);
rms_values = rms(y, 2);
scatter(data_points(:, 1), rms_values, 50, rms_values, 'filled');
colorbar;
xlabel('X坐标 (mm)');
ylabel('信号RMS');
title('各点信号强度分布');
grid on;

% 显示第一个点的时域信号
subplot(1, 3, 2);
plot(x_time_us, y(1, :), 'b-', 'LineWidth', 1.5);
xlabel('时间 (μs)');
ylabel('位移 (mm)');
title('第1个点的时域信号');
grid on;

% 显示第一个点的频域信号
subplot(1, 3, 3);
if n_time_steps > 0
    nfft = 2^nextpow2(n_time_steps);
    fft_signal = fft(y(1, :), nfft);
    % 使用手动设置的采样频率
    freq_vector = (0:nfft-1) * fs_manual / nfft;
    plot(freq_vector(1:nfft/2)/1e6, abs(fft_signal(1:nfft/2)), 'r-', 'LineWidth', 1);
    xlabel('频率 (MHz)');
    ylabel('幅值');
    title('第1个点的频谱');
    grid on;
    xlim([0, fs_manual/2/1e6]);
else
    text(0.5, 0.5, '无数据', 'HorizontalAlignment', 'center');
end

sgtitle('COMSOL超声数据预览', 'FontSize', 14, 'FontWeight', 'bold');

%% 5. 保存为MAT文件
output_file = 'processed_data.mat';
save(output_file, 'x', 'y', 'data_points', 'time_ns', '-v7.3');

fprintf('\n数据已保存到: %s\n', output_file);
fprintf('包含变量:\n');
fprintf('  - x: 时间向量 (秒)\n');
fprintf('  - y: 位移矩阵 (mm)\n');
fprintf('  - data_points: XY坐标 (mm)\n');
fprintf('  - time_ns: 时间向量 (纳秒)\n');

fprintf('\n✓ 数据处理完成！可以在dispersion.m中使用该数据\n');
