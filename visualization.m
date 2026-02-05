%% 波场可视化程序
% 显示超声信号在空间上的传播过程

clear; clc; close all;

%% 数据预处理：加载并重塑蛇形扫描数据
% 加载原始数据
load("/Volumes/ESD-ISO/数据/260126/Laser/41_41.mat"); 

% 计算采样率和时间向量
data_time = x; % 时间向量
fs = 6.25e6; % 采样率 (Hz)

% 设置点阵参数
n_cols = 41;      % x方向列数
n_rows = 41;    % y方向行数
spacing = 5e-4;  % 物理间距 0.5mm
data_x = (0:n_cols-1) * spacing; % x方向坐标 (m)
data_y = (0:n_rows-1) * spacing; % y方向坐标 (m)

% 将蛇形扫描数据重塑为 n_cols×n_rows×length(data_time)
data_xyt = zeros(n_cols, n_rows, length(data_time));

for col = 1:n_cols
    start_idx = (col-1) * n_rows + 1;
    end_idx = col * n_rows;
    col_data = y(start_idx:end_idx, :);
    
    % 蛇形扫描校正：翻转偶数列
    if mod(col, 2) == 0
        % col_data = flipud(col_data);
    end
    
    % 存储到三维数组中: data_xyt(x, y, t)
    data_xyt(col, :, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d列 × %d行\n', n_cols, n_rows);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  数据形状: %s\n', mat2str(size(data_xyt)));

%% 噪声处理流水线 (新增：时域预处理)
fprintf('\n正在进行去噪预处理 (针对 3D 波场数据)...\n');
% 设置预处理参数
center_freq_viz = 200e3; 
bandwidth_viz = 200e3;
wavelet_name_viz = 'sym4';
wavelet_level_viz = 3;

% 对 data_xyt 进行全局滤波和小波去噪，提升动画清晰度
for i = 1:n_cols
    for j = 1:n_rows
        sig = squeeze(data_xyt(i, j, :));
        % 1. 带通滤波 (4阶)
        sig_f = Filter.apply(sig, fs, center_freq_viz, bandwidth_viz, 4);
        % 2. 小波去噪
        sig_w = Filter.waveletDenoise(sig_f, wavelet_name_viz, wavelet_level_viz, 'soft');
        data_xyt(i, j, :) = sig_w;
    end
end
fprintf('预处理完成，动画信号已净化。\n');

%% 波场动画参数设置
frame_delay = 0.001;  % 每帧显示时间 (秒)，可调整播放速度
frame_skip = 2;      % 跳帧显示，1表示显示所有帧，2表示每2帧显示一次

% 可选：只显示部分帧（节省时间）
% start_frame = 1;
% end_frame = 500;
start_frame = 1;
end_frame = length(data_time);

fprintf('\n波场动画设置:\n');
fprintf('  总帧数: %d\n', length(data_time));
fprintf('  显示帧数: %d (跳帧=%d)\n', ceil((end_frame-start_frame+1)/frame_skip), frame_skip);
fprintf('  每帧延迟: %.3f 秒\n', frame_delay);
fprintf('  预计播放时长: %.2f 秒\n', ceil((end_frame-start_frame+1)/frame_skip)*frame_delay);

%% 创建动画窗口
fig = figure('Name', '波场传播动画', 'Position', [100, 100, 900, 700]);

% 定义中心区域选择逻辑
center_range_x = round(n_cols * 0.25) : round(n_cols * 0.75);
center_range_y = round(n_rows * 0.25) : round(n_rows * 0.75);
[center_X, center_Y] = meshgrid(center_range_x, center_range_y);
center_indices = [center_X(:), center_Y(:)];

% 从中心区域随机选择10个点
num_sample_points = 10;
rng('shuffle');  % 使用当前时间作为随机种子
sample_idx = randperm(size(center_indices, 1), min(num_sample_points, size(center_indices, 1)));
sample_points = center_indices(sample_idx, :);

% 提取这10个点的所有时间序列数据
max_values = zeros(num_sample_points, 1);
min_values = zeros(num_sample_points, 1);

for i = 1:num_sample_points
    point_data = squeeze(data_xyt(sample_points(i,1), sample_points(i,2), :));
    max_values(i) = max(point_data);
    min_values(i) = min(point_data);
end

% 计算平均值作为颜色范围
% color_min = mean(min_values);
% color_max = mean(max_values);

%手动设置颜色范围（可选）
color_min = -4e-12;
color_max = 1e-11;

% 使用对称范围
color_range = max(abs(color_min), abs(color_max));
clim_range = [-color_range, color_range];

% 输出颜色范围信息
fprintf('\n颜色范围计算（基于中心区域%d个随机点）:\n', num_sample_points);
fprintf('  采样点位置: \n');
for i = 1:num_sample_points
    fprintf('    点%d: (%d, %d) -> (%.1f mm, %.1f mm)\n', i, ...
            sample_points(i,1), sample_points(i,2), ...
            data_x(sample_points(i,1))*1e3, data_y(sample_points(i,2))*1e3);
end
fprintf('  最大值平均: %.6e\n', mean(max_values));
fprintf('  最小值平均: %.6e\n', mean(min_values));
fprintf('  使用的颜色范围: [%.6e, %.6e]\n', clim_range(1), clim_range(2));

fprintf('\n开始播放动画...\n');
fprintf('按 Ctrl+C 可以中止播放\n\n');

%% 播放动画
for frame_idx = start_frame:frame_skip:end_frame
    % 检查窗口是否已关闭
    if ~ishandle(fig)
        fprintf('\n动画窗口已关闭，停止播放。\n');
        break;
    end
    
    % 提取当前时刻的波场数据 (n_cols × n_rows)
    wavefield = squeeze(data_xyt(:, :, frame_idx));
    
    % 4倍线性插值
    [X_orig, Y_orig] = meshgrid(data_x, data_y);
    interp_factor = 4;
    x_interp = linspace(data_x(1), data_x(end), n_cols * interp_factor - (interp_factor - 1));
    y_interp = linspace(data_y(1), data_y(end), n_rows * interp_factor - (interp_factor - 1));
    [X_interp, Y_interp] = meshgrid(x_interp, y_interp);
    
    % 注意：wavefield 是 [n_cols, n_rows]，需要转置匹配 meshgrid
    wavefield_interp = interp2(X_orig, Y_orig, wavefield', X_interp, Y_interp, 'linear');
    
    % 显示插值后的波场
    imagesc(x_interp * 1e3, y_interp * 1e3, wavefield_interp);
    axis equal tight;
    colormap('jet');
    colorbar;
    caxis(clim_range);  % 使用固定的颜色范围
    
    % 添加标签和标题
    xlabel('X 位置 (mm)', 'FontSize', 12);
    ylabel('Y 位置 (mm)', 'FontSize', 12);
    title(sprintf('波场传播 | 时间: %.2f μs | 帧: %d/%d', ...
                  data_time(frame_idx)*1e6, frame_idx, length(data_time)), ...
          'FontSize', 14, 'FontWeight', 'bold');
    
    % 添加网格
    grid on;
    set(gca, 'YDir', 'normal');  % 确保Y轴方向正确
    
    % 更新显示
    drawnow;
    
    % 延迟控制帧率
    pause(frame_delay);
end

fprintf('动画播放完成！\n');

%% 可选：保存动画为视频文件
% 取消下面注释可以保存视频
% video_writer = VideoWriter('wavefield_animation.avi');
% video_writer.FrameRate = 1/frame_delay;
% open(video_writer);
% 
% for frame_idx = start_frame:frame_skip:end_frame
%     wavefield = squeeze(data_xyt(:, :, frame_idx));
%     imagesc(data_x * 1e3, data_y * 1e3, wavefield');
%     axis equal tight;
%     colormap('jet');
%     colorbar;
%     caxis(clim_range);
%     xlabel('X 位置 (mm)', 'FontSize', 12);
%     ylabel('Y 位置 (mm)', 'FontSize', 12);
%     title(sprintf('波场传播 | 时间: %.2f μs', data_time(frame_idx)*1e6), ...
%           'FontSize', 14);
%     grid on;
%     set(gca, 'YDir', 'normal');
%     
%     frame = getframe(gcf);
%     writeVideo(video_writer, frame);
% end
% 
% close(video_writer);
% fprintf('视频已保存为 wavefield_animation.avi\n');
