%% 单点超声信号分析工具
% 分析processed_data.mat中单个或两个点的时域信号和频谱特征
%
% 使用方法:
%   single_point()                          %- 分析第1个点，不滤波
  single_point(10)                        %- 分析第10个点，不滤波
  single_point(10, [180e3, 220e3])       %- 分析第10个点，带通滤波100-1000kHz
%   single_point(10, [100e3, 500e3], 20)   %- 对比第10和第20个点
%
% 输入参数:
%   point_idx1   - 第一个点的索引号 (默认: 1)
%   filter_range - 带通滤波器频率范围 [低频, 高频] (Hz), 空数组[]表示不滤波 (默认: [])
%   point_idx2   - 第二个点的索引号，用于对比 (默认: [], 不对比)
%
% 输出:
%   显示时域信号和频谱图
%   返回分析结果结构体

function results = single_point(point_idx1, filter_range, point_idx2)
    %% 参数处理
    if nargin < 1 || isempty(point_idx1)
        point_idx1 = 1;
    end
    
    if nargin < 2
        filter_range = [];
    end
    
    if nargin < 3
        point_idx2 = [];
    end
    
    % 判断是否为对比模式
    compare_mode = ~isempty(point_idx2);
    
    %% 加载数据
    fprintf('正在加载数据...\n');
    if ~isfile('processed_data.mat')
        error('未找到 processed_data.mat 文件，请先运行 read.m');
    end
    
    data = load('processed_data.mat');
    x = data.x;              % 时间向量 (秒)
    y = data.y;              % 位移数据 (n_points × n_time_steps)
    data_points = data.data_points;  % 坐标信息
    
    n_points = size(y, 1);
    n_time_steps = size(y, 2);
    
    % 计算采样频率
    dt = x(2) - x(1);
    fs = 1 / dt;
    
    fprintf('数据信息:\n');
    fprintf('  总点数: %d\n', n_points);
    fprintf('  时间步数: %d\n', n_time_steps);
    fprintf('  采样频率: %.2f MHz\n', fs/1e6);
    fprintf('  时间范围: %.3f - %.3f μs\n', x(1)*1e6, x(end)*1e6);
    
    %% 验证索引
    if point_idx1 < 1 || point_idx1 > n_points
        error('点索引1超出范围 (1-%d)', n_points);
    end
    
    if compare_mode && (point_idx2 < 1 || point_idx2 > n_points)
        error('点索引2超出范围 (1-%d)', n_points);
    end
    
    %% 提取信号
    signal1 = y(point_idx1, :);
    coord1 = data_points(point_idx1, :);
    
    if compare_mode
        signal2 = y(point_idx2, :);
        coord2 = data_points(point_idx2, :);
    end
    
    %% 应用滤波器
    apply_filter = ~isempty(filter_range) && length(filter_range) == 2;
    
    if apply_filter
        lowcut = filter_range(1);
        highcut = filter_range(2);
        
        % 验证滤波器参数
        if lowcut >= highcut
            error('低频截止频率必须小于高频截止频率');
        end
        if highcut >= fs/2
            warning('高频截止频率 %.2f MHz 超过奈奎斯特频率 %.2f MHz，将调整为 %.2f MHz', ...
                highcut/1e6, fs/2/1e6, fs/2*0.95/1e6);
            highcut = fs/2 * 0.95;
        end
        
        fprintf('\n应用带通滤波器: %.1f - %.1f kHz\n', lowcut/1e3, highcut/1e3);
        
        % 设计巴特沃斯带通滤波器
        order = 4;
        nyq = fs / 2;
        Wn = [lowcut, highcut] / nyq;
        [b, a] = butter(order, Wn, 'bandpass');
        
        % 应用零相位滤波
        signal1_filtered = filtfilt(b, a, signal1);
        
        if compare_mode
            signal2_filtered = filtfilt(b, a, signal2);
        end
    else
        fprintf('\n未应用滤波器\n');
        signal1_filtered = signal1;
        if compare_mode
            signal2_filtered = signal2;
        end
    end
    
    %% 计算频谱
    nfft = 2^16;  % 使用65536点FFT提高频率分辨率
    
    % 点1的频谱
    fft1_raw = fft(signal1, nfft);
    fft1_filtered = fft(signal1_filtered, nfft);
    freq = (0:nfft-1) * fs / nfft;
    
    % 只保留正频率部分
    pos_idx = 1:nfft/2;
    freq_pos = freq(pos_idx);
    fft1_raw_pos = abs(fft1_raw(pos_idx));
    fft1_filtered_pos = abs(fft1_filtered(pos_idx));
    
    if compare_mode
        fft2_raw = fft(signal2, nfft);
        fft2_filtered = fft(signal2_filtered, nfft);
        fft2_raw_pos = abs(fft2_raw(pos_idx));
        fft2_filtered_pos = abs(fft2_filtered(pos_idx));
    end
    
    %% 计算特征
    features1 = extract_features(signal1_filtered, fs);
    if compare_mode
        features2 = extract_features(signal2_filtered, fs);
    end
    
    %% 可视化
    if ~compare_mode
        % 单点模式: 1行2列
        visualize_single_point(x, signal1, signal1_filtered, freq_pos, ...
            fft1_raw_pos, fft1_filtered_pos, point_idx1, coord1, ...
            features1, apply_filter, filter_range);
    else
        % 对比模式: 2行3列
        visualize_comparison(x, ...
            signal1, signal1_filtered, freq_pos, fft1_raw_pos, fft1_filtered_pos, ...
            signal2, signal2_filtered, fft2_raw_pos, fft2_filtered_pos, ...
            point_idx1, coord1, features1, ...
            point_idx2, coord2, features2, ...
            apply_filter, filter_range);
    end
    
    %% 输出结果
    if nargout > 0
        results.point_idx1 = point_idx1;
        results.coord1 = coord1;
        results.signal1_raw = signal1;
        results.signal1_filtered = signal1_filtered;
        results.features1 = features1;
        
        if compare_mode
            results.point_idx2 = point_idx2;
            results.coord2 = coord2;
            results.signal2_raw = signal2;
            results.signal2_filtered = signal2_filtered;
            results.features2 = features2;
        end
        
        results.time = x;
        results.frequency = freq_pos;
        results.fs = fs;
        results.filter_applied = apply_filter;
        if apply_filter
            results.filter_range = filter_range;
        end
    end
    
    fprintf('\n✓ 分析完成\n');
end

%% 特征提取函数
function features = extract_features(signal, fs)
    % 时域特征
    features.rms = rms(signal);                              % 均方根
    features.peak = max(abs(signal));                        % 峰值
    features.peak_to_peak = max(signal) - min(signal);       % 峰峰值
    features.crest_factor = features.peak / features.rms;    % 波峰因子
    features.mean = mean(signal);                            % 均值
    features.std = std(signal);                              % 标准差
    
    % 频域特征
    nfft = 2^16;  % 使用65536点FFT提高频率分辨率
    fft_signal = fft(signal, nfft);
    freq = (0:nfft/2-1) * fs / nfft;
    psd = abs(fft_signal(1:nfft/2)).^2;
    
    [~, max_idx] = max(psd);
    features.dominant_freq = freq(max_idx);                  % 主频
    features.spectral_centroid = sum(freq .* psd') / sum(psd); % 频谱质心
    
    % 能量
    features.energy = sum(signal.^2);                        % 总能量
end

%% 单点可视化函数
function visualize_single_point(x, signal_raw, signal_filtered, freq, ...
    fft_raw, fft_filtered, point_idx, coord, features, apply_filter, filter_range)
    
    figure('Name', sprintf('单点分析 - 点 #%d', point_idx), ...
        'Position', [100, 100, 1400, 600]);
    
    % 时域信号
    subplot(1, 2, 1);
    hold on;
    if apply_filter
        plot(x*1e6, signal_raw, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1, ...
            'DisplayName', '原始信号');
        plot(x*1e6, signal_filtered, 'b-', 'LineWidth', 1.5, ...
            'DisplayName', '滤波后信号');
        legend('Location', 'best');
    else
        plot(x*1e6, signal_raw, 'b-', 'LineWidth', 1.5);
    end
    hold off;
    xlabel('时间 (μs)', 'FontSize', 11);
    ylabel('位移 (mm)', 'FontSize', 11);
    title(sprintf('点 #%d - 时域信号 (%.2f, %.2f mm)', ...
        point_idx, coord(1), coord(2)), 'FontWeight', 'bold');
    grid on;
    xlim([x(1), x(end)]*1e6);
    
    % 添加特征标注
    text(0.02, 0.98, sprintf('RMS: %.3e mm\n峰值: %.3e mm\n峰峰值: %.3e mm', ...
        features.rms, features.peak, features.peak_to_peak), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 9);
    
    % 频谱
    subplot(1, 2, 2);
    hold on;
    if apply_filter
        plot(freq/1e6, fft_raw, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1, ...
            'DisplayName', '原始频谱');
        plot(freq/1e6, fft_filtered, 'r-', 'LineWidth', 1.5, ...
            'DisplayName', '滤波后频谱');
        
        % 绘制滤波器范围
        ylim_val = ylim;
        patch([filter_range(1), filter_range(2), filter_range(2), filter_range(1)]/1e6, ...
            [ylim_val(1), ylim_val(1), ylim_val(2), ylim_val(2)], ...
            'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '滤波器范围');
        legend('Location', 'best');
    else
        plot(freq/1e6, fft_raw, 'r-', 'LineWidth', 1.5);
    end
    hold off;
    xlabel('频率 (MHz)', 'FontSize', 11);
    ylabel('幅值', 'FontSize', 11);
    title('频谱', 'FontWeight', 'bold');
    grid on;
    if apply_filter
        % 显示滤波器频段范围，稍微扩展10%以便观察
        freq_margin = (filter_range(2) - filter_range(1)) * 0.1;
        xlim([max(0, filter_range(1) - freq_margin)/1e6, (filter_range(2) + freq_margin)/1e6]);
    else
        xlim([0, 1]);
    end
    
    sgtitle(sprintf('单点超声信号分析 - 点索引 #%d', point_idx), ...
        'FontSize', 14, 'FontWeight', 'bold');
end

%% 对比可视化函数
function visualize_comparison(x, ...
    signal1_raw, signal1_filtered, freq, fft1_raw, fft1_filtered, ...
    signal2_raw, signal2_filtered, fft2_raw, fft2_filtered, ...
    point_idx1, coord1, features1, ...
    point_idx2, coord2, features2, ...
    apply_filter, filter_range)
    
    figure('Name', sprintf('双点对比 - 点 #%d vs #%d', point_idx1, point_idx2), ...
        'Position', [50, 50, 1600, 900]);
    
    % === 第一行: 点1 ===
    % 点1 - 时域信号
    subplot(2, 3, 1);
    hold on;
    if apply_filter
        plot(x*1e6, signal1_raw, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
        plot(x*1e6, signal1_filtered, 'b-', 'LineWidth', 1.5);
    else
        plot(x*1e6, signal1_raw, 'b-', 'LineWidth', 1.5);
    end
    hold off;
    xlabel('时间 (μs)', 'FontSize', 10);
    ylabel('位移 (mm)', 'FontSize', 10);
    title(sprintf('点 #%d - 时域信号\n(%.2f, %.2f mm)', ...
        point_idx1, coord1(1), coord1(2)), 'FontWeight', 'bold', 'FontSize', 11);
    grid on;
    xlim([x(1), x(end)]*1e6);
    
    % 点1 - 频谱
    subplot(2, 3, 2);
    hold on;
    if apply_filter
        plot(freq/1e6, fft1_raw, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
        plot(freq/1e6, fft1_filtered, 'r-', 'LineWidth', 1.5);
        
        ylim_val = ylim;
        patch([filter_range(1), filter_range(2), filter_range(2), filter_range(1)]/1e6, ...
            [ylim_val(1), ylim_val(1), ylim_val(2), ylim_val(2)], ...
            'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    else
        plot(freq/1e6, fft1_raw, 'r-', 'LineWidth', 1.5);
    end
    hold off;
    xlabel('频率 (MHz)', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    title('频谱', 'FontWeight', 'bold', 'FontSize', 11);
    grid on;
    if apply_filter
        % 显示滤波器频段范围，稍微扩展10%以便观察
        freq_margin = (filter_range(2) - filter_range(1)) * 0.1;
        xlim([max(0, filter_range(1) - freq_margin)/1e6, (filter_range(2) + freq_margin)/1e6]);
    else
        xlim([0, 1]);
    end
    
    % 点1 - 特征对比
    subplot(2, 3, 3);
    feature_names = {'RMS', '峰值', '峰峰值', '主频\n(kHz)', '频谱质心\n(kHz)'};
    feature_values1 = [features1.rms, features1.peak, features1.peak_to_peak, ...
        features1.dominant_freq/1e3, features1.spectral_centroid/1e3];
    
    bar(1:5, feature_values1, 'FaceColor', [0.2, 0.4, 0.8]);
    set(gca, 'XTick', 1:5, 'XTickLabel', feature_names);
    ylabel('数值', 'FontSize', 10);
    title(sprintf('点 #%d - 特征值', point_idx1), 'FontWeight', 'bold', 'FontSize', 11);
    grid on;
    xtickangle(15);
    
    % === 第二行: 点2 ===
    % 点2 - 时域信号
    subplot(2, 3, 4);
    hold on;
    if apply_filter
        plot(x*1e6, signal2_raw, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
        plot(x*1e6, signal2_filtered, 'b-', 'LineWidth', 1.5);
    else
        plot(x*1e6, signal2_raw, 'b-', 'LineWidth', 1.5);
    end
    hold off;
    xlabel('时间 (μs)', 'FontSize', 10);
    ylabel('位移 (mm)', 'FontSize', 10);
    title(sprintf('点 #%d - 时域信号\n(%.2f, %.2f mm)', ...
        point_idx2, coord2(1), coord2(2)), 'FontWeight', 'bold', 'FontSize', 11);
    grid on;
    xlim([x(1), x(end)]*1e6);
    
    % 点2 - 频谱
    subplot(2, 3, 5);
    hold on;
    if apply_filter
        plot(freq/1e6, fft2_raw, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
        plot(freq/1e6, fft2_filtered, 'r-', 'LineWidth', 1.5);
        
        ylim_val = ylim;
        patch([filter_range(1), filter_range(2), filter_range(2), filter_range(1)]/1e6, ...
            [ylim_val(1), ylim_val(1), ylim_val(2), ylim_val(2)], ...
            'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    else
        plot(freq/1e6, fft2_raw, 'r-', 'LineWidth', 1.5);
    end
    hold off;
    xlabel('频率 (MHz)', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    title('频谱', 'FontWeight', 'bold', 'FontSize', 11);
    grid on;
    if apply_filter
        % 显示滤波器频段范围，稍微扩展10%以便观察
        freq_margin = (filter_range(2) - filter_range(1)) * 0.1;
        xlim([max(0, filter_range(1) - freq_margin)/1e6, (filter_range(2) + freq_margin)/1e6]);
    else
        xlim([0, 1]);
    end
    
    % 点2 - 特征对比
    subplot(2, 3, 6);
    feature_values2 = [features2.rms, features2.peak, features2.peak_to_peak, ...
        features2.dominant_freq/1e3, features2.spectral_centroid/1e3];
    
    bar(1:5, feature_values2, 'FaceColor', [0.8, 0.4, 0.2]);
    set(gca, 'XTick', 1:5, 'XTickLabel', feature_names);
    ylabel('数值', 'FontSize', 10);
    title(sprintf('点 #%d - 特征值', point_idx2), 'FontWeight', 'bold', 'FontSize', 11);
    grid on;
    xtickangle(15);
    
    % 总标题
    if apply_filter
        filter_info = sprintf(' | 滤波器: %.1f-%.1f kHz', ...
            filter_range(1)/1e3, filter_range(2)/1e3);
    else
        filter_info = ' | 无滤波';
    end
    sgtitle(sprintf('双点超声信号对比 - 点 #%d vs 点 #%d%s', ...
        point_idx1, point_idx2, filter_info), ...
        'FontSize', 14, 'FontWeight', 'bold');
    
    % 打印对比信息
    fprintf('\n特征对比:\n');
    fprintf('%-15s | 点 #%-5d | 点 #%-5d | 差异 (%%)\n', '特征', point_idx1, point_idx2);
    fprintf('----------------------------------------------------------\n');
    print_comparison('RMS (mm)', features1.rms, features2.rms);
    print_comparison('峰值 (mm)', features1.peak, features2.peak);
    print_comparison('峰峰值 (mm)', features1.peak_to_peak, features2.peak_to_peak);
    print_comparison('主频 (kHz)', features1.dominant_freq/1e3, features2.dominant_freq/1e3);
    print_comparison('频谱质心 (kHz)', features1.spectral_centroid/1e3, features2.spectral_centroid/1e3);
end

%% 打印对比信息辅助函数
function print_comparison(name, val1, val2)
    diff_percent = abs(val1 - val2) / max(abs(val1), abs(val2)) * 100;
    fprintf('%-15s | %10.4e | %10.4e | %6.2f\n', name, val1, val2, diff_percent);
end
