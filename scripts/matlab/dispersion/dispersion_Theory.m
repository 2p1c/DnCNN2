%% dispersion_Theory.m
% 多层各向同性板Lamb波频散曲线计算 - 全局矩阵法
% 
% 功能: 计算多层结构中Lamb波的波数-频率频散曲线
% 方法: 全局矩阵法 (Global Matrix Method)
% 边界条件: 顶面和底面自由表面
%
% 作者: Copilot
% 日期: 2026-01-14

clear; clc; close all;

%% ==================== 用户参数设置 ====================
% 层参数定义 (从顶层到底层)
% 每层: [厚度(mm), 密度(kg/m^3), 弹性模量(GPa), 泊松比]

% 示例: 三层结构 (铝-环氧-铝)
layers = [
    2.0,  2700,  70,  0.33;   % 第1层: 铝
    0.4,  1180,  1, 0.4;   % 第2层: 硫化硅橡胶
    % 20.0,  330,  0.04,  0.17;   % 第3层: 防热瓦
];

% 频率范围 (kHz)
f_min = 100;    % 最小频率 kHz
f_max = 1000;   % 最大频率 kHz
f_num = 200;    % 频率点数

% 波数范围 (rad/mm)
k_min = 0.01;   % 最小波数 rad/mm (避免k=0的奇异性)
k_max = 10;     % 最大波数 rad/mm
k_num = 300;    % 波数点数

% 选择要提取模态的界面 (0=顶面, 1=第1-2层界面, 2=第2-3层界面, ..., n=底面)
interface_index = 1;  % 提取第1-2层界面处的模态

% 模态过滤参数
max_modes = 4;        % 只显示前N个主要模态 (设为inf显示全部)
mode_tracking = true; % 是否启用模态追踪算法

%% ==================== 频散曲线计算 ====================
fprintf('正在计算多层板Lamb波频散曲线...\n');
fprintf('层数: %d\n', size(layers, 1));
fprintf('频率范围: %.0f - %.0f kHz\n', f_min, f_max);
fprintf('波数范围: %.2f - %.2f rad/mm\n', k_min, k_max);

% 频率和波数向量
freq_vec = linspace(f_min, f_max, f_num) * 1e3;  % 转换为 Hz
k_vec = linspace(k_min, k_max, k_num) * 1e3;     % 转换为 rad/m

% 预处理层参数
[n_layers, h_vec, rho_vec, cL_vec, cT_vec] = preprocess_layers(layers);

% 计算频散行列式矩阵
det_matrix = compute_dispersion_matrix(freq_vec, k_vec, n_layers, h_vec, rho_vec, cL_vec, cT_vec);

% 提取频散曲线 (寻找行列式过零点)
[k_roots, f_roots] = extract_dispersion_curves(det_matrix, freq_vec, k_vec);

fprintf('找到 %d 个原始频散点\n', length(k_roots));

% 模态追踪与过滤
if mode_tracking && ~isempty(k_roots)
    [modes, n_modes_found] = track_modes(k_roots, f_roots, freq_vec);
    fprintf('识别出 %d 个独立模态\n', n_modes_found);
    
    % 只保留前N个主要模态
    n_modes_to_show = min(max_modes, n_modes_found);
    k_roots_filtered = [];
    f_roots_filtered = [];
    mode_labels = [];  % 用于标记每个点属于哪个模态
    
    for m = 1:n_modes_to_show
        mode_k = modes{m}.k;
        mode_f = modes{m}.f;
        k_roots_filtered = [k_roots_filtered; mode_k(:)];
        f_roots_filtered = [f_roots_filtered; mode_f(:)];
        mode_labels = [mode_labels; m * ones(length(mode_k), 1)];
    end
    
    k_roots = k_roots_filtered;
    f_roots = f_roots_filtered;
    fprintf('显示前 %d 个主要模态, 共 %d 个点\n', n_modes_to_show, length(k_roots));
else
    mode_labels = ones(length(k_roots), 1);  % 全部标记为模态1
    n_modes_to_show = 1;
end

% 转换单位回 kHz 和 rad/mm
k_roots_mm = k_roots / 1e3;
f_roots_kHz = f_roots / 1e3;

%% ==================== 计算界面位移幅值 ====================
fprintf('正在计算界面 %d 处的位移幅值...\n', interface_index);
interface_amps = zeros(size(k_roots));

% 可以在此处开启并行计算
% parfor i = 1:length(k_roots)
for i = 1:length(k_roots)
    interface_amps(i) = calculate_interface_amplitude(k_roots(i), f_roots(i), ...
        n_layers, h_vec, rho_vec, cL_vec, cT_vec, interface_index);
end

% 归一化幅值 (0-1范围)
if ~isempty(interface_amps)
    max_amp = max(interface_amps);
    if max_amp > 0
        interface_amps_norm = interface_amps / max_amp;
    else
        interface_amps_norm = interface_amps;
    end
else
    interface_amps_norm = [];
end

%% ==================== 提取指定界面的模态形状 ====================
fprintf('\n正在提取界面 %d 处的模态形状...\n', interface_index);

% 选择几个代表性的频散点提取模态
n_modes_to_extract = min(10, length(k_roots));
mode_indices = round(linspace(1, length(k_roots), n_modes_to_extract));

mode_shapes = cell(n_modes_to_extract, 1);
for i = 1:n_modes_to_extract
    idx = mode_indices(i);
    [ux, uz, sigma_zz, tau_xz, z_coords] = extract_mode_shape(...
        k_roots(idx), f_roots(idx), n_layers, h_vec, rho_vec, cL_vec, cT_vec, interface_index);
    mode_shapes{i} = struct('ux', ux, 'uz', uz, 'sigma_zz', sigma_zz, ...
                            'tau_xz', tau_xz, 'z', z_coords, ...
                            'k', k_roots_mm(idx), 'f', f_roots_kHz(idx));
end

%% ==================== 可视化 ====================
% 图1: 波数-频率频散曲线
figure('Name', 'Lamb波频散曲线', 'Position', [100, 100, 1000, 500]);

subplot(1,2,1);
hold on;
% 原始绘图代码：
% colors = lines(n_modes_to_show);  % 不同模态用不同颜色
% for m = 1:n_modes_to_show
%     idx = (mode_labels == m);
%     plot(f_roots_kHz(idx), k_roots_mm(idx), '.', 'MarkerSize', 4, ...
%          'Color', colors(m,:), 'DisplayName', sprintf('模态 %d', m));
% end
% legend('Location', 'northwest');

% 新的绘图代码：使用scatter显示界面响应幅值
if ~isempty(k_roots_mm)
    scatter(f_roots_kHz, k_roots_mm, 15, interface_amps_norm, 'filled');
    colormap(jet);
    c = colorbar;
    c.Label.String = sprintf('界面 %d 处垂直位移幅值 |u_z| (归一化)', interface_index);
    c.Label.FontSize = 10;
end

xlabel('频率 (kHz)', 'FontSize', 12);
ylabel('波数 k (rad/mm)', 'FontSize', 12);
title(sprintf('波数-频率频散曲线 (颜色: 界面%d响应)', interface_index), 'FontSize', 14, 'FontWeight', 'bold');
xlim([f_min, f_max]);
ylim([k_min, k_max]);
grid on;
box on;

% 图2: 相速度-频率曲线
subplot(1,2,2);
hold on;
cp_roots = 2*pi*f_roots ./ k_roots;  % 相速度 m/s

% 相速度图也使用幅值颜色
if ~isempty(k_roots_mm)
    scatter(f_roots_kHz, cp_roots/1e3, 15, interface_amps_norm, 'filled');
    colormap(jet);
    c = colorbar;
    c.Label.String = '|u_z|';
end

% 原始相速度绘图代码:
% for m = 1:n_modes_to_show
%     idx = (mode_labels == m);
%     plot(f_roots_kHz(idx), cp_roots(idx)/1e3, '.', 'MarkerSize', 4, ...
%          'Color', colors(m,:), 'DisplayName', sprintf('模态 %d', m));
% end
% legend('Location', 'northeast');

xlabel('频率 (kHz)', 'FontSize', 12);
ylabel('相速度 c_p (km/s)', 'FontSize', 12);
title('相速度-频率频散曲线', 'FontSize', 14, 'FontWeight', 'bold');
xlim([f_min, f_max]);
ylim([0, 15]);
grid on;
box on;

sgtitle(sprintf('多层板Lamb波频散曲线 (%d层结构)', n_layers), 'FontSize', 14, 'FontWeight', 'bold');

% 图3: 模态形状示例
if ~isempty(mode_shapes) && ~isempty(mode_shapes{1}.z)
    figure('Name', '模态形状', 'Position', [150, 150, 800, 600]);
    
    % 选择一个模态进行展示
    mode_idx = ceil(n_modes_to_extract/2);
    mode = mode_shapes{mode_idx};
    
    subplot(2,2,1);
    plot(real(mode.ux), mode.z, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(imag(mode.ux), mode.z, 'b--', 'LineWidth', 1);
    xlabel('u_x', 'FontSize', 11);
    ylabel('z (mm)', 'FontSize', 11);
    title('水平位移 u_x', 'FontSize', 12);
    legend('实部', '虚部', 'Location', 'best');
    grid on;
    
    subplot(2,2,2);
    plot(real(mode.uz), mode.z, 'r-', 'LineWidth', 1.5);
    hold on;
    plot(imag(mode.uz), mode.z, 'r--', 'LineWidth', 1);
    xlabel('u_z', 'FontSize', 11);
    ylabel('z (mm)', 'FontSize', 11);
    title('垂直位移 u_z', 'FontSize', 12);
    legend('实部', '虚部', 'Location', 'best');
    grid on;
    
    subplot(2,2,3);
    plot(real(mode.sigma_zz), mode.z, 'g-', 'LineWidth', 1.5);
    xlabel('\sigma_{zz}', 'FontSize', 11);
    ylabel('z (mm)', 'FontSize', 11);
    title('正应力 \sigma_{zz}', 'FontSize', 12);
    grid on;
    
    subplot(2,2,4);
    plot(real(mode.tau_xz), mode.z, 'm-', 'LineWidth', 1.5);
    xlabel('\tau_{xz}', 'FontSize', 11);
    ylabel('z (mm)', 'FontSize', 11);
    title('剪应力 \tau_{xz}', 'FontSize', 12);
    grid on;
    
    sgtitle(sprintf('模态形状 @ f=%.1f kHz, k=%.2f rad/mm', mode.f, mode.k), ...
            'FontSize', 13, 'FontWeight', 'bold');
end

% 图4: 层结构示意图
figure('Name', '层结构', 'Position', [200, 200, 400, 500]);
plot_layer_structure(layers);

fprintf('\n计算完成!\n');

%% ==================== 函数定义 ====================

function [n_layers, h_vec, rho_vec, cL_vec, cT_vec] = preprocess_layers(layers)
    % PREPROCESS_LAYERS 预处理层参数，计算波速
    %
    % 输入:
    %   layers - 层参数矩阵 [n x 4]: [厚度(mm), 密度(kg/m^3), E(GPa), nu]
    %
    % 输出:
    %   n_layers - 层数
    %   h_vec    - 各层厚度向量 (m)
    %   rho_vec  - 各层密度向量 (kg/m^3)
    %   cL_vec   - 各层纵波速度向量 (m/s)
    %   cT_vec   - 各层剪切波速度向量 (m/s)
    
    n_layers = size(layers, 1);
    
    h_vec = layers(:, 1) * 1e-3;      % mm -> m
    rho_vec = layers(:, 2);            % kg/m^3
    E_vec = layers(:, 3) * 1e9;        % GPa -> Pa
    nu_vec = layers(:, 4);             % 泊松比
    
    % 计算Lamé常数
    lambda_vec = E_vec .* nu_vec ./ ((1 + nu_vec) .* (1 - 2*nu_vec));
    mu_vec = E_vec ./ (2 * (1 + nu_vec));
    
    % 计算波速
    cL_vec = sqrt((lambda_vec + 2*mu_vec) ./ rho_vec);  % 纵波速度
    cT_vec = sqrt(mu_vec ./ rho_vec);                    % 剪切波速度
    
    fprintf('层参数预处理完成:\n');
    for i = 1:n_layers
        fprintf('  第%d层: h=%.2fmm, ρ=%.0fkg/m³, cL=%.0fm/s, cT=%.0fm/s\n', ...
                i, h_vec(i)*1e3, rho_vec(i), cL_vec(i), cT_vec(i));
    end
end

function det_matrix = compute_dispersion_matrix(freq_vec, k_vec, n_layers, h_vec, rho_vec, cL_vec, cT_vec)
    % COMPUTE_DISPERSION_MATRIX 计算频散行列式矩阵
    %
    % 对每个(k, f)点计算全局矩阵的行列式
    
    n_freq = length(freq_vec);
    n_k = length(k_vec);
    det_matrix = zeros(n_freq, n_k);
    
    % 使用并行计算加速 (如果有Parallel Computing Toolbox)
    % parfor i_f = 1:n_freq
    for i_f = 1:n_freq
        omega = 2 * pi * freq_vec(i_f);
        
        for i_k = 1:n_k
            k = k_vec(i_k);
            
            % 构建全局矩阵
            G = build_global_matrix(k, omega, n_layers, h_vec, rho_vec, cL_vec, cT_vec);
            
            % 计算行列式 (使用对数行列式避免数值溢出)
            det_matrix(i_f, i_k) = compute_log_det(G);
        end
        
        % 显示进度
        if mod(i_f, 20) == 0
            fprintf('  计算进度: %.1f%%\n', i_f/n_freq*100);
        end
    end
end

function G = build_global_matrix(k, omega, n_layers, h_vec, rho_vec, cL_vec, cT_vec)
    % BUILD_GLOBAL_MATRIX 构建全局矩阵
    %
    % 对于n层结构:
    %   - 每层有4个未知数 (上行P波, 下行P波, 上行SV波, 下行SV波)
    %   - 总共 4*n 个未知数
    %   - 顶面2个边界条件 + 底面2个边界条件 + (n-1)*4个界面连续性条件
    %   - 总共 4*n 个方程
    
    n_eqns = 4 * n_layers;
    G = zeros(n_eqns, n_eqns);
    
    row = 1;
    
    % 1. 顶面自由边界条件 (z = 0, 第1层顶面)
    [D_top, ~] = layer_matrix(k, omega, 0, rho_vec(1), cL_vec(1), cT_vec(1));
    % sigma_zz = 0, tau_xz = 0
    G(row:row+1, 1:4) = D_top(3:4, :);  % 应力行
    row = row + 2;
    
    % 2. 层间界面连续性条件
    z_interface = 0;
    for n = 1:(n_layers - 1)
        z_interface = z_interface + h_vec(n);
        
        % 第n层底面
        [D_n_bottom, ~] = layer_matrix(k, omega, h_vec(n), rho_vec(n), cL_vec(n), cT_vec(n));
        
        % 第n+1层顶面
        [D_np1_top, ~] = layer_matrix(k, omega, 0, rho_vec(n+1), cL_vec(n+1), cT_vec(n+1));
        
        % 连续性条件: u_x, u_z, sigma_zz, tau_xz 连续
        col_n = (n-1)*4 + 1;
        col_np1 = n*4 + 1;
        
        G(row:row+3, col_n:col_n+3) = D_n_bottom;
        G(row:row+3, col_np1:col_np1+3) = -D_np1_top;
        row = row + 4;
    end
    
    % 3. 底面自由边界条件 (第n层底面)
    [D_bottom, ~] = layer_matrix(k, omega, h_vec(n_layers), rho_vec(n_layers), cL_vec(n_layers), cT_vec(n_layers));
    col_last = (n_layers-1)*4 + 1;
    G(row:row+1, col_last:col_last+3) = D_bottom(3:4, :);  % 应力行
end

function [D, coeffs] = layer_matrix(k, omega, z, rho, cL, cT)
    % LAYER_MATRIX 计算单层的位移-应力矩阵
    %
    % 输入:
    %   k     - 波数 (rad/m)
    %   omega - 角频率 (rad/s)
    %   z     - 层内局部坐标 (m), z=0为层顶面
    %   rho   - 密度 (kg/m^3)
    %   cL    - 纵波速度 (m/s)
    %   cT    - 剪切波速度 (m/s)
    %
    % 输出:
    %   D - 4x4矩阵, 行: [ux, uz, sigma_zz, tau_xz], 列: [A+, A-, B+, B-]
    %       A+/A-: 上行/下行P波系数
    %       B+/B-: 上行/下行SV波系数
    
    % 波数分量
    kL = omega / cL;  % 纵波波数
    kT = omega / cT;  % 剪切波波数
    
    % 垂直波数分量 (注意: 可能是虚数，表示倏逝波)
    alpha_sq = kL^2 - k^2;
    beta_sq = kT^2 - k^2;
    
    % 处理倏逝波情况
    if alpha_sq >= 0
        alpha = sqrt(alpha_sq);
    else
        alpha = 1i * sqrt(-alpha_sq);
    end
    
    if beta_sq >= 0
        beta = sqrt(beta_sq);
    else
        beta = 1i * sqrt(-beta_sq);
    end
    
    % Lamé常数
    mu = rho * cT^2;
    lambda = rho * cL^2 - 2*mu;
    
    % 指数项
    exp_alpha_p = exp(1i * alpha * z);
    exp_alpha_m = exp(-1i * alpha * z);
    exp_beta_p = exp(1i * beta * z);
    exp_beta_m = exp(-1i * beta * z);
    
    % 位移矩阵行 (假设波沿x方向传播, exp(i(kx - omega*t)))
    % u_x = i*k*phi + d(psi)/dz
    % u_z = d(phi)/dz + i*k*psi (注意符号约定)
    
    % 第1行: u_x
    ux_row = [1i*k*exp_alpha_p, 1i*k*exp_alpha_m, 1i*beta*exp_beta_p, -1i*beta*exp_beta_m];
    
    % 第2行: u_z
    uz_row = [1i*alpha*exp_alpha_p, -1i*alpha*exp_alpha_m, -1i*k*exp_beta_p, -1i*k*exp_beta_m];
    
    % 应力矩阵行
    % sigma_zz = lambda*(d^2phi/dx^2 + d^2phi/dz^2) + 2*mu*d^2phi/dz^2 + 2*mu*d^2psi/dxdz
    %          = -lambda*kL^2*phi - 2*mu*alpha^2*phi - 2*mu*i*k*(dpsi/dz)
    % 简化: sigma_zz = -(lambda + 2*mu)*alpha^2*phi - lambda*k^2*phi + 2*mu*i*k*beta*psi
    %                = -rho*cL^2*(alpha^2 + k^2 - kL^2)*phi/something... 
    % 使用标准形式:
    % sigma_zz = (lambda*(-k^2 - alpha^2) + 2*mu*(-alpha^2))*phi + 2*mu*(ik)(dpsi/dz)
    
    % 更简洁的形式 (基于势函数):
    % sigma_zz = -mu*(2*k^2 - kT^2)*phi_coeff - 2*mu*i*k*beta*psi_coeff (with proper signs)
    
    % 使用清晰的公式:
    c1 = lambda * (-k^2) + (lambda + 2*mu) * (-alpha^2);  % = -lambda*k^2 - (lambda+2*mu)*alpha^2
    c2 = 2 * mu * 1i * k;
    
    % sigma_zz for phi: factor * exp(...)
    % sigma_zz for psi: 2*mu*i*k * (d/dz of psi)
    
    % 标准Lamb波矩阵元素:
    % 使用更稳健的公式
    factor_p = 2*k^2 - kT^2;  % = 2k^2 - omega^2/cT^2
    
    % 第3行: sigma_zz
    sigma_zz_row = [-mu*factor_p*exp_alpha_p, -mu*factor_p*exp_alpha_m, ...
                    -2*mu*1i*k*beta*exp_beta_p, 2*mu*1i*k*beta*exp_beta_m];
    
    % 第4行: tau_xz = mu*(du_z/dx + du_x/dz) = mu*(ik*uz + d(ux)/dz)
    % tau_xz = 2*mu*i*k*alpha*phi - mu*(k^2 - beta^2)*psi = 2*mu*i*k*alpha*phi - mu*(2*k^2 - kT^2)*psi
    tau_xz_row = [2*mu*1i*k*alpha*exp_alpha_p, -2*mu*1i*k*alpha*exp_alpha_m, ...
                  mu*factor_p*exp_beta_p, mu*factor_p*exp_beta_m];
    
    % 组装矩阵
    D = [ux_row; uz_row; sigma_zz_row; tau_xz_row];
    
    coeffs = struct('alpha', alpha, 'beta', beta, 'k', k, 'omega', omega, ...
                    'mu', mu, 'lambda', lambda);
end

function log_det = compute_log_det(G)
    % COMPUTE_LOG_DET 计算矩阵行列式的符号化对数值
    %
    % 使用LU分解避免数值溢出/下溢
    
    [~, U, P] = lu(G);
    
    % 行列式 = det(P) * det(L) * det(U) = det(P) * 1 * prod(diag(U))
    diag_U = diag(U);
    
    % 处理零对角元
    if any(abs(diag_U) < 1e-300)
        log_det = -Inf;
        return;
    end
    
    % 计算对数行列式 (保留符号信息)
    log_abs_det = sum(log(abs(diag_U)));
    sign_det = prod(sign(real(diag_U)));
    
    % 考虑置换矩阵的符号
    perm_sign = det(P);  % +1 或 -1
    
    % 返回符号化的"有效行列式"用于过零检测
    % 我们需要检测行列式何时接近零
    log_det = sign_det * perm_sign * exp(log_abs_det - max(log_abs_det, 0));
end

function [k_roots, f_roots] = extract_dispersion_curves(det_matrix, freq_vec, k_vec)
    % EXTRACT_DISPERSION_CURVES 从行列式矩阵中提取频散曲线
    %
    % 通过检测符号变化来找到行列式的零点
    
    k_roots = [];
    f_roots = [];
    
    n_freq = length(freq_vec);
    n_k = length(k_vec);
    
    % 方法: 检测行列式符号变化或局部最小值
    for i_f = 1:n_freq
        det_row = real(det_matrix(i_f, :));
        
        for i_k = 1:(n_k-1)
            % 检测符号变化
            if det_row(i_k) * det_row(i_k+1) < 0
                % 线性插值找零点
                k1 = k_vec(i_k);
                k2 = k_vec(i_k+1);
                d1 = det_row(i_k);
                d2 = det_row(i_k+1);
                
                k_zero = k1 - d1 * (k2 - k1) / (d2 - d1);
                
                k_roots = [k_roots; k_zero];
                f_roots = [f_roots; freq_vec(i_f)];
            end
        end
    end
    
    % 去除异常点 (相速度过小或过大)
    if ~isempty(k_roots)
        cp = 2*pi*f_roots ./ k_roots;
        valid_idx = cp > 100 & cp < 20000;  % 相速度在100-20000 m/s之间
        k_roots = k_roots(valid_idx);
        f_roots = f_roots(valid_idx);
    end
end

function [ux, uz, sigma_zz, tau_xz, z_coords] = extract_mode_shape(k, f, n_layers, h_vec, rho_vec, cL_vec, cT_vec, interface_idx)
    % EXTRACT_MODE_SHAPE 提取指定界面处的模态形状
    %
    % 输入:
    %   k, f - 频散点的波数和频率
    %   interface_idx - 界面索引 (0=顶面, 1=第1-2层界面, ...)
    %
    % 输出:
    %   ux, uz - 位移场
    %   sigma_zz, tau_xz - 应力场
    %   z_coords - z坐标 (mm)
    
    omega = 2 * pi * f;
    
    % 构建全局矩阵并求解模态系数
    G = build_global_matrix(k, omega, n_layers, h_vec, rho_vec, cL_vec, cT_vec);
    
    % 使用SVD求解零空间 (近似零特征值对应的特征向量)
    [~, S, V] = svd(G);
    
    % 最小奇异值对应的向量就是模态系数
    coeff_vec = V(:, end);
    
    % 生成z坐标 (穿过所有层的离散点)
    n_points_per_layer = 50;
    z_coords = [];
    z_offset = 0;
    
    for n = 1:n_layers
        z_local = linspace(0, h_vec(n), n_points_per_layer)';
        z_coords = [z_coords; z_offset + z_local];
        z_offset = z_offset + h_vec(n);
    end
    
    z_coords = z_coords * 1e3;  % 转换为 mm
    
    % 计算各点的位移和应力
    n_total = length(z_coords);
    ux = zeros(n_total, 1);
    uz = zeros(n_total, 1);
    sigma_zz = zeros(n_total, 1);
    tau_xz = zeros(n_total, 1);
    
    z_offset = 0;
    point_idx = 1;
    
    for n = 1:n_layers
        coeff_n = coeff_vec((n-1)*4+1 : n*4);  % 该层的4个系数
        
        z_local_vec = linspace(0, h_vec(n), n_points_per_layer)';
        
        for i = 1:n_points_per_layer
            z_local = z_local_vec(i);
            [D, ~] = layer_matrix(k, omega, z_local, rho_vec(n), cL_vec(n), cT_vec(n));
            
            field = D * coeff_n;
            ux(point_idx) = field(1);
            uz(point_idx) = field(2);
            sigma_zz(point_idx) = field(3);
            tau_xz(point_idx) = field(4);
            
            point_idx = point_idx + 1;
        end
        
        z_offset = z_offset + h_vec(n);
    end
    
    % 归一化
    max_disp = max(abs([ux; uz]));
    if max_disp > 0
        ux = ux / max_disp;
        uz = uz / max_disp;
        sigma_zz = sigma_zz / max(abs(sigma_zz));
        tau_xz = tau_xz / max(abs(tau_xz));
    end
end

function plot_layer_structure(layers)
    % PLOT_LAYER_STRUCTURE 绘制层结构示意图
    
    n_layers = size(layers, 1);
    total_h = sum(layers(:, 1));
    
    colors = lines(n_layers);
    
    z_current = 0;
    for n = 1:n_layers
        h = layers(n, 1);
        
        % 绘制层
        rectangle('Position', [-1, z_current, 2, h], ...
                  'FaceColor', colors(n, :), 'EdgeColor', 'k', 'LineWidth', 1);
        
        % 添加标签
        text(0, z_current + h/2, sprintf('层%d\nh=%.2fmm\nρ=%.0fkg/m³', ...
             n, h, layers(n,2)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'FontSize', 9);
        
        z_current = z_current + h;
    end
    
    xlabel('x', 'FontSize', 11);
    ylabel('z (mm)', 'FontSize', 11);
    title('多层板结构示意图', 'FontSize', 12, 'FontWeight', 'bold');
    xlim([-1.5, 1.5]);
    ylim([-0.1*total_h, 1.1*total_h]);
    set(gca, 'YDir', 'reverse');  % z轴向下
    axis equal;
    box on;
end

function [modes, n_modes] = track_modes(k_roots, f_roots, freq_vec)
    % TRACK_MODES 模态追踪算法 - 将散点连接成连续的模态曲线
    %
    % 算法原理:
    %   1. 按频率分组所有点
    %   2. 从低频开始，使用最近邻算法将相邻频率的点连接起来
    %   3. 根据曲线长度排序，长曲线是主要模态
    %
    % 输入:
    %   k_roots  - 所有频散点的波数 (rad/m)
    %   f_roots  - 所有频散点的频率 (Hz)
    %   freq_vec - 频率向量 (Hz)
    %
    % 输出:
    %   modes   - cell数组，每个元素包含一个模态的k和f值
    %   n_modes - 识别出的模态数量
    
    % 参数设置
    k_tolerance = (max(k_roots) - min(k_roots)) / 20;  % 波数连接容差
    min_mode_length = 10;  % 模态最少点数
    
    % 获取唯一频率值
    unique_freqs = unique(f_roots);
    n_freqs = length(unique_freqs);
    
    % 按频率分组
    k_by_freq = cell(n_freqs, 1);
    for i = 1:n_freqs
        idx = abs(f_roots - unique_freqs(i)) < 1;  % 容差1Hz
        k_by_freq{i} = k_roots(idx);
    end
    
    % 初始化模态追踪
    modes = {};
    used_points = cell(n_freqs, 1);  % 记录已使用的点
    for i = 1:n_freqs
        used_points{i} = false(size(k_by_freq{i}));
    end
    
    % 从低频开始追踪每个模态
    for start_freq_idx = 1:n_freqs
        k_at_freq = k_by_freq{start_freq_idx};
        used = used_points{start_freq_idx};
        
        for start_k_idx = 1:length(k_at_freq)
            if used(start_k_idx)
                continue;  % 该点已被使用
            end
            
            % 开始一条新的模态曲线
            mode_k = k_at_freq(start_k_idx);
            mode_f = unique_freqs(start_freq_idx);
            used_points{start_freq_idx}(start_k_idx) = true;
            
            current_k = mode_k(end);
            
            % 向高频方向追踪
            for freq_idx = (start_freq_idx + 1):n_freqs
                k_candidates = k_by_freq{freq_idx};
                used_candidates = used_points{freq_idx};
                
                % 找最近的未使用点
                available_idx = find(~used_candidates);
                if isempty(available_idx)
                    continue;
                end
                
                k_available = k_candidates(available_idx);
                [min_dist, nearest_idx] = min(abs(k_available - current_k));
                
                if min_dist < k_tolerance
                    % 连接到该点
                    actual_idx = available_idx(nearest_idx);
                    mode_k = [mode_k; k_candidates(actual_idx)];
                    mode_f = [mode_f; unique_freqs(freq_idx)];
                    used_points{freq_idx}(actual_idx) = true;
                    current_k = mode_k(end);
                end
            end
            
            % 保存模态 (如果足够长)
            if length(mode_k) >= min_mode_length
                modes{end+1} = struct('k', mode_k, 'f', mode_f);
            end
        end
    end
    
    n_modes = length(modes);
    
    % 按模态长度排序 (点数多的优先)
    if n_modes > 1
        mode_lengths = cellfun(@(m) length(m.k), modes);
        [~, sort_idx] = sort(mode_lengths, 'descend');
        modes = modes(sort_idx);
    end
    
    % 额外排序: 在相同长度下，按平均波数从小到大排序
    % 这样低阶模态(A0, S0)会排在前面
    if n_modes > 1
        mode_avg_k = cellfun(@(m) mean(m.k), modes);
        mode_lengths = cellfun(@(m) length(m.k), modes);

        % 对于长度接近的模态，按平均波数排序
        % 使用复合排序键: 长度(主要) + 平均波数(次要)
        max_len = max(mode_lengths);
        sort_key = mode_lengths - mode_avg_k / max(mode_avg_k) * 0.1 * max_len;
        [~, sort_idx] = sort(sort_key, 'descend');
        modes = modes(sort_idx);
    end
end

function uz_amp = calculate_interface_amplitude(k, f, n_layers, h_vec, rho_vec, cL_vec, cT_vec, interface_idx)
    % CALCULATE_INTERFACE_AMPLITUDE 计算特定界面处的垂直位移幅值
    %
    % 输入:
    %   k, f - 波数(rad/m)和频率(Hz)
    %   n_layers, h_vec, ... - 层参数
    %   interface_idx - 界面索引 (0=顶面, 1=第一层底面, ...)
    %
    % 输出:
    %   uz_amp - 该位置的垂直位移幅值 (未归一化)

    omega = 2 * pi * f;

    % 1. 构建全局矩阵
    G = build_global_matrix(k, omega, n_layers, h_vec, rho_vec, cL_vec, cT_vec);

    % 2. 使用SVD求解零空间得到模态系数
    [~, ~, V] = svd(G);
    coeff_vec = V(:, end);

    % 3. 确定目标界面的位置 (层索引和局部z坐标)
    target_layer_idx = 0;
    target_z_local = 0;

    if interface_idx == 0
        % 顶面 (第1层, z=0)
        target_layer_idx = 1;
        target_z_local = 0;
    elseif interface_idx >= n_layers
        % 底面 (最后一层, z=h)
        target_layer_idx = n_layers;
        target_z_local = h_vec(n_layers);
    else
        % 中间界面 (第i层底面 或 第i+1层顶面)
        % 选择第i+1层顶面计算方便
        target_layer_idx = interface_idx + 1;
        target_z_local = 0;
    end

    % 4. 计算位移
    % 提取该层的系数
    coeff_n = coeff_vec((target_layer_idx-1)*4+1 : target_layer_idx*4);

    % 计算该层该位置的矩阵
    [D, ~] = layer_matrix(k, omega, target_z_local, rho_vec(target_layer_idx), ...
                          cL_vec(target_layer_idx), cT_vec(target_layer_idx));

    % 计算场量
    field = D * coeff_n;

    % 提取垂直位移幅值 |u_z|
    uz_amp = abs(field(2));
end
