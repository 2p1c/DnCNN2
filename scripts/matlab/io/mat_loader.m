function [data_xyt, data_time, data_x, data_y, fs] = mat_loader(data_file, n_points, spacing)
% MAT_LOADER 从.mat文件加载波场数据并转换为三维数组格式
%
% 输入参数:
%   data_file - 数据文件路径 (字符串)
%   n_points  - 点阵大小:
%               - 标量: 方形点阵 (例如 31 表示 31×31)
%               - 向量: 矩形点阵 [n_x, n_y] (例如 [41, 31] 表示 41×31)
%   spacing   - 点阵物理间距 (米, 例如 1e-3 表示 1mm)
%
% 输出参数:
%   data_xyt  - 三维数组 (n_y × n_x × 时间点数)
%   data_time - 时间向量 (秒)
%   data_x    - X方向坐标向量 (米)
%   data_y    - Y方向坐标向量 (米)
%   fs        - 采样率 (Hz)
%
% 示例:
%   [data_xyt, data_time, data_x, data_y, fs] = mat_loader('data.mat', 31, 1e-3);
%   [data_xyt, data_time, data_x, data_y, fs] = mat_loader('data.mat', [41, 31], 1e-3);
%
% 作者: 自动生成
% 日期: 2026-01-28

    % 解析点阵大小
    if isscalar(n_points)
        n_x = n_points;
        n_y = n_points;
    else
        n_x = n_points(1);
        n_y = n_points(2);
    end

    % 加载原始数据
    fprintf('正在加载数据: %s\n', data_file);
    data_struct = load(data_file, 'x', 'y'); % 加载变量 x (时间) 和 y (扫描数据)
    x = data_struct.x;
    y = data_struct.y;
    
    % 计算采样率和时间向量
    data_time = x;
    fs = 1/(data_time(2)-data_time(1));
    
    % x, y方向坐标
    data_x = (0:n_x-1) * spacing;
    data_y = (0:n_y-1) * spacing;
    
    % 将扫描数据重塑为 n_y×n_x×时间点数 的三维数组
    data_xyt = zeros(n_y, n_x, length(data_time));
    
    for col = 1:n_x
        start_idx = (col-1) * n_y + 1;
        end_idx = col * n_y;
        col_data = y(start_idx:end_idx, :);
        data_xyt(:, col, :) = col_data;
    end
    
    % 显示加载信息
    fprintf('数据加载完成:\n');
    fprintf('  点阵大小: %d × %d (X × Y)\n', n_x, n_y);
    fprintf('  时间点数: %d\n', length(data_time));
    fprintf('  采样率: %.2f MHz\n', fs/1e6);
    fprintf('  时间范围: %.2f - %.2f μs\n', data_time(1)*1e6, data_time(end)*1e6);
    fprintf('  空间范围: %.1f × %.1f mm\n', data_x(end)*1e3, data_y(end)*1e3);
    
end

