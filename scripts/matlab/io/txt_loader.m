classdef txt_loader < handle
    % TXT文件加载器 - 统一的TXT信号文件加载模块
    % 
    % 功能:
    %   - 自动检测TXT文件格式
    %   - 智能跳过标题行
    %   - 支持多种分隔符
    %   - 批量加载功能
    %
    % 使用示例:
    %   signal_data = txt_loader.load_single('data.txt');
    %   signal_data = txt_loader.load_batch(folder_path, '*.txt');
    %
    % 版本: 1.0
    % 作者: 重构团队
    % 日期: 2024
    
    methods (Static)
        function signal_data = load_single(file_path, options)
            % 加载单个TXT文件
            %
            % 输入:
            %   file_path - 文件路径
            %   options - 可选参数
            %       .skip_rows - 跳过行数 (默认: 自动检测)
            %       .delimiter - 分隔符 (默认: 自动检测)
            %       .time_col - 时间列索引 (默认: 1)
            %       .signal_col - 信号列索引 (默认: 2)
            %       .remove_nan - 是否移除NaN (默认: true)
            %
            % 输出:
            %   signal_data - 结构体，包含以下字段:
            %       .data - 信号数据
            %       .time - 时间向量
            %       .fs - 采样率
            
            arguments
                file_path (1,:) char
                options.skip_rows (1,1) double = -1  % -1表示自动检测
                options.delimiter (1,:) char = 'auto'
                options.time_col (1,1) double {mustBePositive, mustBeInteger} = 1
                options.signal_col (1,1) double {mustBePositive, mustBeInteger} = 2
                options.remove_nan (1,1) logical = true
            end
            
            % 验证文件存在
            if ~exist(file_path, 'file')
                error('txt_loader:load_single', 'File not found: %s', file_path);
            end
            
            try
                % 自动检测格式
                if options.skip_rows == -1 || strcmp(options.delimiter, 'auto')
                    [skip_rows, delimiter] = txt_loader.detect_format(file_path);
                    if options.skip_rows == -1
                        options.skip_rows = skip_rows;
                    end
                    if strcmp(options.delimiter, 'auto')
                        options.delimiter = delimiter;
                    end
                end
                
                % 读取数据
                [time_data, signal_data_raw, fs] = txt_loader.read_txt_file(...
                    file_path, options.skip_rows, options.delimiter, ...
                    options.time_col, options.signal_col, options.remove_nan);
                
                % 创建结果结构体
                signal_data = struct();
                signal_data.data = signal_data_raw;
                signal_data.time = time_data;
                signal_data.fs = fs;
                signal_data.source = file_path;
                
                fprintf('✓ 成功加载: %s (采样率: %.2f MHz)\n', file_path, fs/1e6);
                
            catch ME
                error('txt_loader:load_single', ...
                      'Failed to load %s: %s', file_path, ME.message);
            end
        end
        
        function [signal_data_array, file_list] = load_batch(folder_path, pattern, options)
            % 批量加载TXT文件
            %
            % 输入:
            %   folder_path - 文件夹路径
            %   pattern - 文件模式 (如 '*.txt', '1.txt;2.txt;3.txt')
            %   options - 同load_single的options
            %
            % 输出:
            %   signal_data_array - 结构体的cell数组，每个结构体包含data/time/fs字段
            %   file_list - 成功加载的文件列表
            
            arguments
                folder_path (1,:) char
                pattern (1,:) char = '*.txt'
                options.skip_rows (1,1) double = -1
                options.delimiter (1,:) char = 'auto'
                options.time_col (1,1) double = 1
                options.signal_col (1,1) double = 2
                options.remove_nan (1,1) logical = true
                options.show_progress (1,1) logical = true
            end
            
            % 验证文件夹存在
            if ~exist(folder_path, 'dir')
                error('txt_loader:load_batch', 'Folder not found: %s', folder_path);
            end
            
            % 获取文件列表
            file_list = txt_loader.get_file_list(folder_path, pattern);
            
            if isempty(file_list)
                warning('txt_loader:load_batch', 'No files found matching pattern: %s', pattern);
                signal_data_array = {};
                return;
            end
            
            % 初始化
            num_files = length(file_list);
            signal_data_array = cell(num_files, 1);
            
            % 进度条
            h_wait = [];
            if options.show_progress
                h_wait = waitbar(0, 'Loading TXT files...');
            end
            
            try
                % 加载所有文件
                for i = 1:num_files
                    try
                        signal_data_array{i} = txt_loader.load_single(...
                            file_list{i}, ...
                            'skip_rows', options.skip_rows, ...
                            'delimiter', options.delimiter, ...
                            'time_col', options.time_col, ...
                            'signal_col', options.signal_col, ...
                            'remove_nan', options.remove_nan);
                        
                        if options.show_progress && ~isempty(h_wait) && ishandle(h_wait)
                            waitbar(i/num_files, h_wait, ...
                                sprintf('Loaded %d/%d files...', i, num_files));
                        end
                        
                    catch ME
                        warning('txt_loader:load_batch', ...
                            'Failed to load %s: %s', file_list{i}, ME.message);
                        signal_data_array{i} = [];
                    end
                end
                
                if options.show_progress && ~isempty(h_wait) && ishandle(h_wait)
                    close(h_wait);
                end
                
                % 移除加载失败的项
                valid_idx = ~cellfun(@isempty, signal_data_array);
                signal_data_array = signal_data_array(valid_idx);
                file_list = file_list(valid_idx);
                
                fprintf('✓ 批量加载完成: %d/%d 文件成功\n', sum(valid_idx), num_files);
                
            catch ME
                if options.show_progress && ~isempty(h_wait) && ishandle(h_wait)
                    close(h_wait);
                end
                rethrow(ME);
            end
        end
        
        function [skip_rows, delimiter] = detect_format(file_path)
            % 自动检测TXT文件格式
            %
            % 输出:
            %   skip_rows - 需要跳过的行数
            %   delimiter - 检测到的分隔符
            
            fid = fopen(file_path, 'r');
            if fid == -1
                error('txt_loader:detect_format', 'Cannot open file: %s', file_path);
            end
            
            try
                % 读取前10行用于分析
                max_check_lines = 10;
                lines = cell(max_check_lines, 1);
                for i = 1:max_check_lines
                    line = fgetl(fid);
                    if ~ischar(line)
                        break;
                    end
                    lines{i} = line;
                end
                fclose(fid);
                
                % 移除空行
                lines = lines(~cellfun(@isempty, lines));
                
                % 检测分隔符
                delimiter = txt_loader.detect_delimiter(lines);
                
                % 检测标题行数
                skip_rows = txt_loader.detect_header_rows(lines, delimiter);
                
            catch ME
                fclose(fid);
                rethrow(ME);
            end
        end
        
        function delimiter = detect_delimiter(lines)
            % 检测分隔符
            %
            % 常见分隔符: tab, space, comma, semicolon
            
            if isempty(lines)
                delimiter = '\t';
                return;
            end
            
            % 测试各种分隔符
            test_delimiters = {'\t', ' ', ',', ';'};
            delimiter_scores = zeros(size(test_delimiters));
            
            for i = 1:length(test_delimiters)
                delim = test_delimiters{i};
                
                % 统计使用该分隔符的列数一致性
                col_counts = zeros(length(lines), 1);
                for j = 1:length(lines)
                    if strcmp(delim, ' ')
                        % 空格需要特殊处理（连续空格算一个）
                        parts = strsplit(strtrim(lines{j}));
                    else
                        parts = strsplit(lines{j}, delim);
                    end
                    col_counts(j) = length(parts);
                end
                
                % 列数一致性越高，分数越高
                if std(col_counts) == 0 && mean(col_counts) >= 2
                    delimiter_scores(i) = mean(col_counts) * 10;
                end
            end
            
            % 选择得分最高的分隔符
            [~, best_idx] = max(delimiter_scores);
            delimiter = test_delimiters{best_idx};
        end
        
        function skip_rows = detect_header_rows(lines, delimiter)
            % 检测需要跳过的标题行数
            %
            % 策略：找到第一行数值数据
            
            skip_rows = 0;
            
            % 修复：确保lines不为空
            if isempty(lines)
                skip_rows = 0;
                return;
            end
            
            for i = 1:length(lines)
                % 修复：确保lines{i}不为空
                if isempty(lines{i})
                    continue;
                end
                
                if strcmp(delimiter, ' ')
                    parts = strsplit(strtrim(lines{i}));
                else
                    parts = strsplit(lines{i}, delimiter);
                end
                
                % 检查是否是数值数据
                if length(parts) >= 2
                    try
                        % 尝试转换前两列为数值
                        val1 = str2double(parts{1});
                        val2 = str2double(parts{2});
                        
                        if ~isnan(val1) && ~isnan(val2)
                            % 找到数值行，返回跳过的行数
                            skip_rows = i - 1;  % 修复：i从1开始，所以是i-1
                            return;
                        end
                    catch
                        % 转换失败，继续下一行
                    end
                end
            end
            
            % 修复：如果都不是数值，默认不跳过（而不是跳过5行）
            % 这样可以避免跳过有效数据
            skip_rows = 0;
        end
        
        function [time_data, signal_data, fs] = read_txt_file(...
                file_path, skip_rows, delimiter, time_col, signal_col, remove_nan)
            % 读取TXT文件数据 - 使用多种策略确保成功读取
            
            % 修复：确保skip_rows至少为0
            if skip_rows < 0
                skip_rows = 0;
            end
            
            % 策略1: 尝试使用readmatrix（MATLAB R2019a+）
            if exist('readmatrix', 'file')
                try
                    fprintf('尝试使用readmatrix读取文件...\n');
                    data_matrix = readmatrix(file_path, 'NumHeaderLines', skip_rows);
                    
                    if ~isempty(data_matrix) && size(data_matrix, 2) >= max(time_col, signal_col)
                        time_data = data_matrix(:, time_col);
                        signal_data = data_matrix(:, signal_col);
                        
                        % 移除NaN值
                        if remove_nan
                            valid_idx = ~isnan(time_data) & ~isnan(signal_data);
                            time_data = time_data(valid_idx);
                            signal_data = signal_data(valid_idx);
                        end
                        
                        % 计算采样率
                        fs = txt_loader.calculate_sampling_rate(time_data);
                        fprintf('readmatrix成功读取 %d 个数据点\n', length(time_data));
                        return;
                    end
                catch ME
                    fprintf('readmatrix失败: %s，尝试其他方法...\n', ME.message);
                end
            end
            
            % 策略2: 使用textscan（带多种编码尝试）
            encodings = {'UTF-8', 'GBK', 'ISO-8859-1', 'native'};
            
            for enc_idx = 1:length(encodings)
                try
                    fprintf('尝试使用textscan (编码: %s)...\n', encodings{enc_idx});
                    
                    % 打开文件
                    if strcmp(encodings{enc_idx}, 'native')
                        fid = fopen(file_path, 'r');
                    else
                        fid = fopen(file_path, 'r', 'n', encodings{enc_idx});
                    end
                    
                    if fid == -1
                        continue;
                    end
                    
                    % 跳过标题行
                    for i = 1:skip_rows
                        line = fgetl(fid);
                        if ~ischar(line)
                            break;
                        end
                    end
                    
                    % 读取数据
                    if strcmp(delimiter, ' ')
                        data_array = textscan(fid, '%f %f', 'Delimiter', ' ', ...
                            'MultipleDelimsAsOne', true, 'CollectOutput', true, ...
                            'EmptyValue', NaN, 'CommentStyle', '#');
                    else
                        data_array = textscan(fid, '%f %f', 'Delimiter', delimiter, ...
                            'CollectOutput', true, 'EmptyValue', NaN, 'CommentStyle', '#');
                    end
                    
                    fclose(fid);
                    
                    % 检查是否读取到数据
                    if ~isempty(data_array) && ~isempty(data_array{1}) && size(data_array{1}, 1) > 0
                        data_matrix = data_array{1};
                        
                        if size(data_matrix, 2) >= max(time_col, signal_col)
                            time_data = data_matrix(:, time_col);
                            signal_data = data_matrix(:, signal_col);
                            
                            % 移除NaN值
                            if remove_nan
                                valid_idx = ~isnan(time_data) & ~isnan(signal_data);
                                time_data = time_data(valid_idx);
                                signal_data = signal_data(valid_idx);
                            end
                            
                            if ~isempty(time_data)
                                fs = txt_loader.calculate_sampling_rate(time_data);
                                fprintf('textscan成功读取 %d 个数据点 (编码: %s)\n', length(time_data), encodings{enc_idx});
                                return;
                            end
                        end
                    end
                    
                catch ME
                    if fid ~= -1
                        fclose(fid);
                    end
                    fprintf('textscan (编码: %s) 失败: %s\n', encodings{enc_idx}, ME.message);
                end
            end
            
            % 策略3: 使用dlmread（旧版MATLAB）
            if exist('dlmread', 'file')
                try
                    fprintf('尝试使用dlmread读取文件...\n');
                    
                    if strcmp(delimiter, '\t')
                        delim_char = '\t';
                    elseif strcmp(delimiter, ' ')
                        delim_char = ' ';
                    else
                        delim_char = delimiter;
                    end
                    
                    data_matrix = dlmread(file_path, delim_char, skip_rows, 0);
                    
                    if ~isempty(data_matrix) && size(data_matrix, 2) >= max(time_col, signal_col)
                        time_data = data_matrix(:, time_col);
                        signal_data = data_matrix(:, signal_col);
                        
                        % 移除NaN值
                        if remove_nan
                            valid_idx = ~isnan(time_data) & ~isnan(signal_data);
                            time_data = time_data(valid_idx);
                            signal_data = signal_data(valid_idx);
                        end
                        
                        fs = txt_loader.calculate_sampling_rate(time_data);
                        fprintf('dlmread成功读取 %d 个数据点\n', length(time_data));
                        return;
                    end
                catch ME
                    fprintf('dlmread失败: %s\n', ME.message);
                end
            end
            
            % 所有方法都失败
            error('txt_loader:read_txt_file', ...
                'No data could be read from file. Check file format.');
        end
        
        function fs = calculate_sampling_rate(time_data)
            % 计算采样率
            if length(time_data) > 1
                % 计算时间间隔
                time_diffs = diff(time_data);
                
                % 移除异常值（如果有）
                valid_diffs = time_diffs(time_diffs > 0);
                
                if ~isempty(valid_diffs)
                    dt = median(valid_diffs);  % 使用中位数更稳健
                    
                    if dt > 0 && dt < 1  % 确保dt在合理范围内（小于1秒）
                        fs = 1 / dt;
                    else
                        fs = 1e6;  % 默认1MHz
                        warning('txt_loader:calculate_sampling_rate', ...
                            'Calculated time step (%.2e) is unreasonable, using default sampling rate 1MHz', dt);
                    end
                else
                    fs = 1e6;
                    warning('txt_loader:calculate_sampling_rate', ...
                        'No valid time differences found, using default sampling rate 1MHz');
                end
            else
                fs = 1e6;
                warning('txt_loader:calculate_sampling_rate', ...
                    'Insufficient data points, using default sampling rate 1MHz');
            end
        end
        
        function file_list = get_file_list(folder_path, pattern)
            % 获取文件列表
            %
            % 支持两种模式:
            % 1. 通配符模式: '*.txt'
            % 2. 编号文件模式: '1.txt;2.txt;...' 或 数字范围
            
            file_list = {};
            
            % 检查是否是编号文件模式
            if contains(pattern, ';')
                % 显式文件列表
                file_names = strsplit(pattern, ';');
                for i = 1:length(file_names)
                    file_path = fullfile(folder_path, strtrim(file_names{i}));
                    if exist(file_path, 'file')
                        file_list{end+1} = file_path;
                    end
                end
            elseif ~contains(pattern, '*')
                % 尝试编号文件模式 (1.txt, 2.txt, ...)
                i = 1;
                while true
                    file_path = fullfile(folder_path, sprintf('%d.txt', i));
                    if exist(file_path, 'file')
                        file_list{end+1} = file_path;
                        i = i + 1;
                    else
                        break;
                    end
                    
                    % 安全限制：最多1000个文件
                    if i > 1000
                        break;
                    end
                end
            else
                % 通配符模式
                files = dir(fullfile(folder_path, pattern));
                for i = 1:length(files)
                    if ~files(i).isdir
                        file_list{end+1} = fullfile(folder_path, files(i).name);
                    end
                end
            end
        end
        
        function config = create_template_config(file_path)
            % 从示例文件创建配置模板
            %
            % 输出:
            %   config - 配置结构体
            
            [skip_rows, delimiter] = txt_loader.detect_format(file_path);
            
            config = struct();
            config.skip_rows = skip_rows;
            config.delimiter = delimiter;
            config.time_col = 1;
            config.signal_col = 2;
            config.remove_nan = true;
            
            fprintf('检测到的配置:\n');
            fprintf('  跳过行数: %d\n', skip_rows);
            fprintf('  分隔符: ''%s''\n', delimiter);
            fprintf('  时间列: %d\n', config.time_col);
            fprintf('  信号列: %d\n', config.signal_col);
        end
    end
end
