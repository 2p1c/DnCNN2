"""
Acoustic Feature Validation Module

训练后验证：对比输入原始信号、目标信号、预测结果的声学特征，
确认去噪过程中原始信号的声学信息没有丢失。

分析维度：
- 时域特征：信号到达时间、峰值振幅、RMS、峰值因子、过零率
- 频谱特征：功率谱密度(PSD)、频谱质心、频谱带宽、主频、-3dB带宽
- 能量特征：总能量、子频带能量分布
- 波数特征：相速度色散、群速度包络
- 质量指标：互相关、频谱相干性、包络相关性

Usage:
    from acoustic_validation import run_acoustic_validation
    run_acoustic_validation(model, val_loader, device)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from typing import Dict, Tuple, Optional


# ============================================================
# 物理常数 (与 data_utils.py 一致)
# ============================================================
SAMPLING_RATE: float = 6.25e6   # 6.25 MHz
NUM_POINTS: int = 1000          # 每个信号的采样点数
DURATION: float = 160e-6        # 160 μs
CENTER_FREQ: float = 250e3      # 250 kHz

# 频率轴 (用于 FFT 分析)
FREQ_RESOLUTION = SAMPLING_RATE / NUM_POINTS  # ~6.25 kHz per bin

# 子频带边界 (Hz) — 对应超声检测常见频段
SUB_BANDS = [
    (0, 100e3),       # 低频 (环境噪声主导)
    (100e3, 200e3),   # 中低频 (接近中心频率下沿)
    (200e3, 400e3),   # 中心频段 (信号主能量区域)
    (400e3, SAMPLING_RATE / 2),  # 高频 (可能包含高次谐波)
]
SUB_BAND_LABELS = ["0-100k", "100k-200k", "200k-400k", ">400k"]


# ============================================================
# 时域特征提取
# ============================================================
def _signal_arrival_time(sig: np.ndarray, threshold_ratio: float = 0.1) -> float:
    """
    基于阈值的信号到达时间检测.
    
    使用信号包络的最大值 × threshold_ratio 作为阈值，
    找到包络首次超过阈值的时间点。
    
    Args:
        sig: 1D 信号数组 (NUM_POINTS,)
        threshold_ratio: 阈值比例 (相对于包络最大值)
    
    Returns:
        到达时间 (秒), 若未检测到则返回 NaN
    """
    # Hilbert 变换提取包络
    analytic = scipy_signal.hilbert(sig)
    envelope = np.abs(analytic)
    
    threshold = envelope.max() * threshold_ratio
    indices = np.where(envelope > threshold)[0]
    
    if len(indices) == 0:
        return float('nan')
    
    # 将采样索引转换为时间 (秒)
    dt = DURATION / NUM_POINTS
    return indices[0] * dt


def _extract_time_features(sig: np.ndarray) -> Dict[str, float]:
    """
    提取时域特征.
    
    Returns:
        包含以下键的字典:
        - arrival_time_us: 信号到达时间 (μs)
        - peak_amplitude: 峰值振幅
        - rms: 均方根值
        - crest_factor: 峰值因子 (peak / rms)
        - zero_crossing_rate: 过零率
    """
    arrival = _signal_arrival_time(sig)
    peak = np.max(np.abs(sig))
    rms = np.sqrt(np.mean(sig ** 2))
    crest = peak / rms if rms > 1e-10 else 0.0
    
    # 过零率: 符号变化次数 / 总采样点数
    sign_changes = np.sum(np.abs(np.diff(np.sign(sig))) > 0)
    zcr = sign_changes / len(sig)
    
    return {
        'arrival_time_us': arrival * 1e6 if not np.isnan(arrival) else float('nan'),
        'peak_amplitude': float(peak),
        'rms': float(rms),
        'crest_factor': float(crest),
        'zero_crossing_rate': float(zcr),
    }


# ============================================================
# 频谱特征提取
# ============================================================
def _compute_psd(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Welch 方法计算功率谱密度.
    
    参数选择:
    - nperseg=256: 约 40μs 窗长，频率分辨率 ~24 kHz
    - 50% overlap: 标准设置
    
    Returns:
        (frequencies, psd) 数组对
    """
    freqs, psd = scipy_signal.welch(
        sig, fs=SAMPLING_RATE, nperseg=256, noverlap=128
    )
    return freqs, psd


def _extract_freq_features(sig: np.ndarray) -> Dict[str, float]:
    """
    提取频谱特征.
    
    Returns:
        包含以下键的字典:
        - spectral_centroid_khz: 频谱质心 (kHz)
        - spectral_bandwidth_khz: 频谱带宽 (kHz)
        - dominant_freq_khz: 主频 (kHz)
        - bandwidth_3db_khz: -3dB 带宽 (kHz)
    """
    freqs, psd = _compute_psd(sig)
    
    # 避免除零
    total_power = np.sum(psd)
    if total_power < 1e-20:
        return {
            'spectral_centroid_khz': 0.0,
            'spectral_bandwidth_khz': 0.0,
            'dominant_freq_khz': 0.0,
            'bandwidth_3db_khz': 0.0,
        }
    
    # 频谱质心 = Σ(f_i * PSD_i) / Σ(PSD_i)
    centroid = np.sum(freqs * psd) / total_power
    
    # 频谱带宽 = sqrt(Σ((f_i - centroid)² * PSD_i) / Σ(PSD_i))
    bandwidth = np.sqrt(np.sum((freqs - centroid) ** 2 * psd) / total_power)
    
    # 主频 (PSD 最大值对应频率)
    dominant_freq = freqs[np.argmax(psd)]
    
    # -3dB 带宽: PSD 下降到峰值一半处的频率范围
    psd_max = np.max(psd)
    half_power = psd_max / 2.0
    above_half = freqs[psd >= half_power]
    if len(above_half) >= 2:
        bw_3db = above_half[-1] - above_half[0]
    else:
        bw_3db = 0.0
    
    return {
        'spectral_centroid_khz': float(centroid / 1e3),
        'spectral_bandwidth_khz': float(bandwidth / 1e3),
        'dominant_freq_khz': float(dominant_freq / 1e3),
        'bandwidth_3db_khz': float(bw_3db / 1e3),
    }


# ============================================================
# 能量特征提取
# ============================================================
def _extract_energy_features(sig: np.ndarray) -> Dict[str, float]:
    """
    提取能量特征: 总能量 + 子频带能量.
    
    子频带划分 (Hz):
    - [0, 100k]: 低频段
    - [100k, 200k]: 中低频段
    - [200k, 400k]: 中心频段 (信号主能量)
    - [400k, Nyquist]: 高频段
    
    Returns:
        总能量和各子频带能量占比
    """
    total_energy = float(np.sum(sig ** 2))
    
    # FFT 计算频域能量分布
    fft_vals = np.fft.rfft(sig)
    fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / SAMPLING_RATE)
    power_spectrum = np.abs(fft_vals) ** 2
    
    features = {'total_energy': total_energy}
    
    for (low, high), label in zip(SUB_BANDS, SUB_BAND_LABELS):
        mask = (fft_freqs >= low) & (fft_freqs < high)
        band_energy = float(np.sum(power_spectrum[mask]))
        # 存储能量占比 (百分比)
        ratio = (band_energy / np.sum(power_spectrum) * 100) if np.sum(power_spectrum) > 0 else 0
        features[f'energy_ratio_{label}'] = float(ratio)
    
    return features


# ============================================================
# 波数特征提取
# ============================================================
def _extract_wavenumber_features(sig: np.ndarray) -> Dict[str, float]:
    """
    提取波数相关特征.
    
    基于 FFT 相位信息估计：
    - 平均相速度: 通过主频和波长关系估算
    - 群速度: 通过包络传播速度估算
    - 相位线性度: 相位对频率的线性度 (越线性 = 越无色散)
    
    Returns:
        波数特征字典
    """
    fft_vals = np.fft.rfft(sig)
    fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / SAMPLING_RATE)
    
    # 相位谱 (展开)
    phase = np.unwrap(np.angle(fft_vals))
    magnitude = np.abs(fft_vals)
    
    # 在信号有效频段内分析 (100kHz - 500kHz，避免噪声主导低频段)
    valid_mask = (fft_freqs >= 100e3) & (fft_freqs <= 500e3)
    valid_freqs = fft_freqs[valid_mask]
    valid_phase = phase[valid_mask]
    valid_mag = magnitude[valid_mask]
    
    if len(valid_freqs) < 3:
        return {
            'phase_linearity': 0.0,
            'dominant_wavenumber': 0.0,
            'spectral_energy_concentration': 0.0,
        }
    
    # 相位线性度: 用线性回归拟合相位-频率关系，R² 越高 = 色散越小
    # 用幅度加权，让高能量频率成分贡献更大
    weights = valid_mag / (valid_mag.sum() + 1e-10)
    
    # 加权线性拟合
    mean_f = np.sum(weights * valid_freqs)
    mean_p = np.sum(weights * valid_phase)
    cov_fp = np.sum(weights * (valid_freqs - mean_f) * (valid_phase - mean_p))
    var_f = np.sum(weights * (valid_freqs - mean_f) ** 2)
    
    if var_f > 1e-20:
        slope = cov_fp / var_f
        predicted = mean_p + slope * (valid_freqs - mean_f)
        ss_res = np.sum(weights * (valid_phase - predicted) ** 2)
        ss_tot = np.sum(weights * (valid_phase - mean_p) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
    else:
        r_squared = 0.0
    
    # 主波数: k = 2πf/v，这里用 k = f/c 的等效量
    # 简化为 PSD 峰值对应的 "等效波数"
    peak_idx = np.argmax(valid_mag)
    dominant_wavenumber = float(valid_freqs[peak_idx] / 1e3)  # 归一化单位
    
    # 频谱能量集中度: 主频附近 ±50kHz 内能量占总能量比
    peak_freq = valid_freqs[peak_idx]
    conc_mask = (valid_freqs >= peak_freq - 50e3) & (valid_freqs <= peak_freq + 50e3)
    concentration = float(np.sum(valid_mag[conc_mask] ** 2) / (np.sum(valid_mag ** 2) + 1e-10) * 100)
    
    return {
        'phase_linearity': float(r_squared),
        'dominant_wavenumber': dominant_wavenumber,
        'spectral_energy_concentration': concentration,
    }


# ============================================================
# 信号质量指标 (成对比较)
# ============================================================
def _cross_correlation_peak(sig_a: np.ndarray, sig_b: np.ndarray) -> Tuple[float, int]:
    """
    计算归一化互相关的峰值和时延.
    
    互相关峰值越接近 1.0，两个信号越相似。
    时延反映信号之间的时间偏移。
    
    Returns:
        (peak_value, lag_samples)
    """
    # 归一化
    a_norm = sig_a - np.mean(sig_a)
    b_norm = sig_b - np.mean(sig_b)
    
    denom = np.sqrt(np.sum(a_norm ** 2) * np.sum(b_norm ** 2))
    if denom < 1e-10:
        return 0.0, 0
    
    corr = np.correlate(a_norm, b_norm, mode='full') / denom
    peak_idx = np.argmax(corr)
    lag = peak_idx - (len(sig_a) - 1)
    
    return float(corr[peak_idx]), int(lag)


def _spectral_coherence(sig_a: np.ndarray, sig_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算频谱相干性 (MSC - Magnitude Squared Coherence).
    
    Cxy(f) = |Pxy(f)|² / (Pxx(f) * Pyy(f))
    
    值域 [0, 1]，1 = 两信号在该频率完全相干。
    
    Returns:
        (frequencies, coherence) 数组对
    """
    freqs, coh = scipy_signal.coherence(
        sig_a, sig_b, fs=SAMPLING_RATE, nperseg=256, noverlap=128
    )
    return freqs, coh


def _envelope_correlation(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    计算两个信号包络的 Pearson 相关系数.
    
    包络通过 Hilbert 变换提取，反映信号的振幅调制结构。
    
    Returns:
        相关系数 [-1, 1]
    """
    env_a = np.abs(scipy_signal.hilbert(sig_a))
    env_b = np.abs(scipy_signal.hilbert(sig_b))
    
    # Pearson 相关系数
    cov = np.mean((env_a - env_a.mean()) * (env_b - env_b.mean()))
    std_prod = env_a.std() * env_b.std()
    
    if std_prod < 1e-10:
        return 0.0
    return float(cov / std_prod)


# ============================================================
# 特征聚合
# ============================================================
def _extract_all_features(sig: np.ndarray) -> Dict[str, float]:
    """提取单个信号的全部声学特征."""
    features = {}
    features.update(_extract_time_features(sig))
    features.update(_extract_freq_features(sig))
    features.update(_extract_energy_features(sig))
    features.update(_extract_wavenumber_features(sig))
    return features


# ============================================================
# 可视化
# ============================================================
def _plot_validation_figure(
    input_signals: np.ndarray,
    target_signals: np.ndarray,
    predicted_signals: np.ndarray,
    all_features: Dict,
    quality_metrics: Dict,
    save_path: str
) -> None:
    """
    生成综合声学验证图 (7 个面板).
    
    Args:
        input_signals: (N, 1000) 输入信号 (含噪声)
        target_signals: (N, 1000) 目标信号 (干净)
        predicted_signals: (N, 1000) 预测信号 (去噪后)
        all_features: 各信号的特征字典
        quality_metrics: 质量指标字典
        save_path: 图片保存路径
    """
    fig = plt.figure(figsize=(22, 28))
    fig.suptitle(
        'Acoustic Feature Validation Report\n'
        'Verifying acoustic information preservation during denoising',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 时间轴 (μs)
    time_us = np.linspace(0, 160, NUM_POINTS)
    
    # 选择一个代表性样本用于详细展示
    sample_idx = 0
    inp = input_signals[sample_idx]
    tgt = target_signals[sample_idx]
    pred = predicted_signals[sample_idx]
    
    # ---- Panel 1: 波形叠加对比 ----
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(time_us, inp, 'b-', alpha=0.4, linewidth=0.6, label='Input (noisy)')
    ax1.plot(time_us, tgt, 'g-', alpha=0.8, linewidth=0.9, label='Target (clean)')
    ax1.plot(time_us, pred, 'r--', alpha=0.8, linewidth=0.9, label='Predicted (denoised)')
    ax1.set_title('Panel 1: Waveform Overlay', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 160)
    
    # ---- Panel 2: PSD 对比 ----
    ax2 = fig.add_subplot(4, 2, 2)
    for sig, label, color, ls in [
        (inp, 'Input', 'blue', '-'),
        (tgt, 'Target', 'green', '-'),
        (pred, 'Predicted', 'red', '--'),
    ]:
        freqs, psd = _compute_psd(sig)
        ax2.semilogy(freqs / 1e3, psd, color=color, linestyle=ls, alpha=0.8, linewidth=1.0, label=label)
    ax2.set_title('Panel 2: Power Spectral Density (PSD)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('PSD (V²/Hz)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=CENTER_FREQ / 1e3, color='gray', linestyle=':', alpha=0.5, label='Center Freq')
    ax2.set_xlim(0, SAMPLING_RATE / 2 / 1e3)
    
    # ---- Panel 3: 特征保留度条形图 (Predicted vs Target) ----
    ax3 = fig.add_subplot(4, 2, 3)
    
    # 选取关键特征，计算预测值相对目标值的保留度 (%)
    key_features = [
        'peak_amplitude', 'rms', 'crest_factor', 'zero_crossing_rate',
        'spectral_centroid_khz', 'dominant_freq_khz', 'bandwidth_3db_khz',
        'total_energy', 'phase_linearity', 'spectral_energy_concentration'
    ]
    feature_labels = [
        'Peak Amp', 'RMS', 'Crest Factor', 'ZCR',
        'Spectral\nCentroid', 'Dominant\nFreq', '-3dB BW',
        'Total\nEnergy', 'Phase\nLinearity', 'Energy\nConcentration'
    ]
    
    # 计算多样本平均保留度
    preservation_rates = []
    for feat in key_features:
        tgt_vals = [f['target'][feat] for f in all_features if not np.isnan(f['target'].get(feat, float('nan')))]
        pred_vals = [f['predicted'][feat] for f in all_features if not np.isnan(f['predicted'].get(feat, float('nan')))]
        
        if tgt_vals and pred_vals:
            tgt_mean = np.mean(tgt_vals)
            pred_mean = np.mean(pred_vals)
            if abs(tgt_mean) > 1e-10:
                rate = min(pred_mean / tgt_mean, tgt_mean / pred_mean) * 100
            else:
                rate = 100.0 if abs(pred_mean) < 1e-10 else 0.0
        else:
            rate = 0.0
        preservation_rates.append(rate)
    
    colors_bar = ['#4CAF50' if r >= 80 else '#FFC107' if r >= 60 else '#F44336' for r in preservation_rates]
    bars = ax3.bar(range(len(key_features)), preservation_rates, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good (≥80%)')
    ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Warning (≥60%)')
    ax3.set_xticks(range(len(key_features)))
    ax3.set_xticklabels(feature_labels, fontsize=7, rotation=0, ha='center')
    ax3.set_ylabel('Preservation Rate (%)')
    ax3.set_title('Panel 3: Feature Preservation (Predicted vs Target)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 110)
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在每个柱上标注数值
    for bar_item, rate in zip(bars, preservation_rates):
        ax3.text(bar_item.get_x() + bar_item.get_width() / 2, bar_item.get_height() + 1,
                 f'{rate:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # ---- Panel 4: 子频带能量对比 ----
    ax4 = fig.add_subplot(4, 2, 4)
    
    # 多样本平均能量比
    energy_data = {label: {'input': [], 'target': [], 'predicted': []} for label in SUB_BAND_LABELS}
    for f in all_features:
        for label in SUB_BAND_LABELS:
            key = f'energy_ratio_{label}'
            energy_data[label]['input'].append(f['input'].get(key, 0))
            energy_data[label]['target'].append(f['target'].get(key, 0))
            energy_data[label]['predicted'].append(f['predicted'].get(key, 0))
    
    x_pos = np.arange(len(SUB_BAND_LABELS))
    width = 0.25
    
    inp_means = [np.mean(energy_data[l]['input']) for l in SUB_BAND_LABELS]
    tgt_means = [np.mean(energy_data[l]['target']) for l in SUB_BAND_LABELS]
    pred_means = [np.mean(energy_data[l]['predicted']) for l in SUB_BAND_LABELS]
    
    ax4.bar(x_pos - width, inp_means, width, label='Input', color='#2196F3', alpha=0.8)
    ax4.bar(x_pos, tgt_means, width, label='Target', color='#4CAF50', alpha=0.8)
    ax4.bar(x_pos + width, pred_means, width, label='Predicted', color='#F44336', alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{l} Hz' for l in SUB_BAND_LABELS], fontsize=9)
    ax4.set_ylabel('Energy Ratio (%)')
    ax4.set_title('Panel 4: Sub-band Energy Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ---- Panel 5: 互相关函数 ----
    ax5 = fig.add_subplot(4, 2, 5)
    
    # Target vs Predicted 的互相关
    a_norm = tgt - np.mean(tgt)
    b_norm = pred - np.mean(pred)
    denom = np.sqrt(np.sum(a_norm ** 2) * np.sum(b_norm ** 2))
    if denom > 1e-10:
        corr_tp = np.correlate(a_norm, b_norm, mode='full') / denom
    else:
        corr_tp = np.zeros(2 * NUM_POINTS - 1)
    
    lags = np.arange(-(NUM_POINTS - 1), NUM_POINTS)
    lag_us = lags * (DURATION / NUM_POINTS) * 1e6  # 转为 μs
    
    ax5.plot(lag_us, corr_tp, 'r-', linewidth=0.8, label='Target ↔ Predicted')
    ax5.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax5.set_title('Panel 5: Cross-Correlation (Target ↔ Predicted)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Lag (μs)')
    ax5.set_ylabel('Correlation Coefficient')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-20, 20)  # 只展示 ±20μs 范围
    
    peak_corr, peak_lag = _cross_correlation_peak(tgt, pred)
    ax5.annotate(f'Peak: {peak_corr:.4f}\nLag: {peak_lag} samples',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # ---- Panel 6: 频谱相干性 ----
    ax6 = fig.add_subplot(4, 2, 6)
    
    freq_coh, coh_tp = _spectral_coherence(tgt, pred)
    freq_coh_inp, coh_ip = _spectral_coherence(inp, pred)
    
    ax6.plot(freq_coh / 1e3, coh_tp, 'r-', linewidth=1.0, alpha=0.8, label='Target ↔ Predicted')
    ax6.plot(freq_coh_inp / 1e3, coh_ip, 'b-', linewidth=0.8, alpha=0.5, label='Input ↔ Predicted')
    ax6.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
    ax6.axvline(x=CENTER_FREQ / 1e3, color='gray', linestyle=':', alpha=0.5)
    ax6.set_title('Panel 6: Spectral Coherence', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Frequency (kHz)')
    ax6.set_ylabel('Coherence')
    ax6.set_ylim(0, 1.05)
    ax6.set_xlim(0, SAMPLING_RATE / 2 / 1e3)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 计算信号频段内的平均相干性
    signal_band = (freq_coh >= 100e3) & (freq_coh <= 500e3)
    avg_coh = np.mean(coh_tp[signal_band]) if np.any(signal_band) else 0
    ax6.annotate(f'Avg coherence\n(100-500kHz): {avg_coh:.3f}',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 ha='right', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # ---- Panel 7: 综合统计表 ----
    ax7 = fig.add_subplot(4, 2, (7, 8))
    ax7.axis('off')
    
    # 汇总指标
    avg_metrics = quality_metrics['averaged']
    
    # 构建表格数据
    table_data = [
        ['Metric', 'Target ↔ Predicted', 'Assessment'],
        ['Cross-Correlation Peak', f"{avg_metrics['xcorr_target_pred']:.4f}",
         _assess(avg_metrics['xcorr_target_pred'], 0.9, 0.7)],
        ['Envelope Correlation', f"{avg_metrics['envelope_corr_target_pred']:.4f}",
         _assess(avg_metrics['envelope_corr_target_pred'], 0.85, 0.65)],
        ['Avg Spectral Coherence\n(100-500kHz)', f"{avg_metrics['avg_coherence_target_pred']:.4f}",
         _assess(avg_metrics['avg_coherence_target_pred'], 0.8, 0.6)],
        ['Arrival Time Error (μs)', f"{avg_metrics['arrival_time_error_us']:.3f}",
         _assess_lower(avg_metrics['arrival_time_error_us'], 2.0, 5.0)],
        ['Peak Amplitude Preservation', f"{avg_metrics['peak_amplitude_preservation']:.1f}%",
         _assess(avg_metrics['peak_amplitude_preservation'] / 100, 0.8, 0.6)],
        ['RMS Preservation', f"{avg_metrics['rms_preservation']:.1f}%",
         _assess(avg_metrics['rms_preservation'] / 100, 0.8, 0.6)],
        ['Dominant Freq Preservation', f"{avg_metrics['dominant_freq_preservation']:.1f}%",
         _assess(avg_metrics['dominant_freq_preservation'] / 100, 0.85, 0.65)],
        ['Energy Distribution Match', f"{avg_metrics['energy_distribution_match']:.1f}%",
         _assess(avg_metrics['energy_distribution_match'] / 100, 0.8, 0.6)],
    ]
    
    # 绘制表格
    table = ax7.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.25, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # 表头样式
    for j in range(3):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # 根据评估结果给单元格上色
    for i in range(1, len(table_data)):
        assessment = table_data[i][2]
        if '✅' in assessment:
            table[i, 2].set_facecolor('#E8F5E9')
        elif '⚠️' in assessment:
            table[i, 2].set_facecolor('#FFF9C4')
        else:
            table[i, 2].set_facecolor('#FFEBEE')
    
    ax7.set_title('Panel 7: Summary Statistics (Averaged over all samples)',
                   fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _assess(value: float, good_thresh: float, warn_thresh: float) -> str:
    """评估指标: 越高越好."""
    if value >= good_thresh:
        return f'✅ Good'
    elif value >= warn_thresh:
        return f'⚠️ Fair'
    else:
        return f'❌ Poor'


def _assess_lower(value: float, good_thresh: float, warn_thresh: float) -> str:
    """评估指标: 越低越好 (如误差)."""
    if value <= good_thresh:
        return f'✅ Good'
    elif value <= warn_thresh:
        return f'⚠️ Fair'
    else:
        return f'❌ Poor'


# ============================================================
# 文本报告
# ============================================================
def _print_report(quality_metrics: Dict) -> None:
    """打印简洁的验证结果文本报告."""
    avg = quality_metrics['averaged']
    
    print("\n" + "=" * 60)
    print("  ACOUSTIC FEATURE VALIDATION REPORT")
    print("=" * 60)
    
    rows = [
        ("Cross-Correlation Peak",    f"{avg['xcorr_target_pred']:.4f}",          _assess(avg['xcorr_target_pred'], 0.9, 0.7)),
        ("Envelope Correlation",      f"{avg['envelope_corr_target_pred']:.4f}",  _assess(avg['envelope_corr_target_pred'], 0.85, 0.65)),
        ("Spectral Coherence (avg)",  f"{avg['avg_coherence_target_pred']:.4f}",  _assess(avg['avg_coherence_target_pred'], 0.8, 0.6)),
        ("Arrival Time Error",        f"{avg['arrival_time_error_us']:.3f} μs",   _assess_lower(avg['arrival_time_error_us'], 2.0, 5.0)),
        ("Peak Amp Preservation",     f"{avg['peak_amplitude_preservation']:.1f}%", _assess(avg['peak_amplitude_preservation'] / 100, 0.8, 0.6)),
        ("RMS Preservation",          f"{avg['rms_preservation']:.1f}%",          _assess(avg['rms_preservation'] / 100, 0.8, 0.6)),
        ("Dominant Freq Preservation", f"{avg['dominant_freq_preservation']:.1f}%", _assess(avg['dominant_freq_preservation'] / 100, 0.85, 0.65)),
        ("Energy Distribution Match", f"{avg['energy_distribution_match']:.1f}%", _assess(avg['energy_distribution_match'] / 100, 0.8, 0.6)),
    ]
    
    print(f"  {'Metric':<30} {'Value':>12}  {'Status'}")
    print("-" * 60)
    for name, val, status in rows:
        print(f"  {name:<30} {val:>12}  {status}")
    
    print("=" * 60)
    
    # 总体评估
    good_count = sum(1 for _, _, s in rows if '✅' in s)
    total = len(rows)
    print(f"\n  Overall: {good_count}/{total} metrics passed ✅")
    
    if good_count == total:
        print("  → Excellent! Acoustic features well preserved during denoising.")
    elif good_count >= total * 0.7:
        print("  → Good. Most acoustic features preserved. Check ⚠️/❌ items.")
    else:
        print("  → Warning: Significant acoustic information loss detected.")
    
    print("=" * 60 + "\n")


# ============================================================
# 主入口
# ============================================================
def run_acoustic_validation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = "fig_acoustic_validation.png",
    num_samples: int = 20
) -> Dict:
    """
    运行完整的声学特征验证.
    
    在训练结束后调用，对比验证集上的输入(含噪)、
    目标(干净)、预测(去噪)信号的声学特征，
    生成综合可视化图和文本报告。
    
    Args:
        model: 训练好的模型
        dataloader: 验证集 DataLoader
        device: 推理设备
        save_path: 输出图片路径
        num_samples: 用于分析的样本数量
        
    Returns:
        包含所有特征和质量指标的字典
    """
    model.eval()
    
    # ------ Step 1: 收集数据并运行推理 ------
    all_noisy = []
    all_clean = []
    for batch_noisy, batch_clean in dataloader:
        all_noisy.append(batch_noisy)
        all_clean.append(batch_clean)
        if sum(n.shape[0] for n in all_noisy) >= 100:
            break
    
    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)
    
    # 随机选择样本
    total = all_noisy.shape[0]
    n = min(num_samples, total)
    indices = np.random.choice(total, size=n, replace=False)
    
    noisy = all_noisy[indices]
    clean = all_clean[indices]
    
    with torch.no_grad():
        predicted = model(noisy.to(device))
    
    # 转为 NumPy (去掉通道维度)
    inp_np = noisy.cpu().numpy()[:, 0, :]    # (N, 1000)
    tgt_np = clean.cpu().numpy()[:, 0, :]    # (N, 1000)
    pred_np = predicted.cpu().numpy()[:, 0, :]  # (N, 1000)
    
    print(f"[INFO] Analyzing {n} samples for acoustic validation...")
    
    # ------ Step 2: 逐样本提取特征 ------
    all_features = []
    for i in range(n):
        sample_features = {
            'input': _extract_all_features(inp_np[i]),
            'target': _extract_all_features(tgt_np[i]),
            'predicted': _extract_all_features(pred_np[i]),
        }
        all_features.append(sample_features)
    
    # ------ Step 3: 计算质量指标 ------
    xcorr_tp_list = []
    env_corr_tp_list = []
    coherence_tp_list = []
    arrival_errors = []
    peak_pres_list = []
    rms_pres_list = []
    dom_freq_pres_list = []
    energy_match_list = []
    
    for i in range(n):
        tgt_sig = tgt_np[i]
        pred_sig = pred_np[i]
        
        # 互相关
        peak_corr, _ = _cross_correlation_peak(tgt_sig, pred_sig)
        xcorr_tp_list.append(peak_corr)
        
        # 包络相关
        env_corr_tp_list.append(_envelope_correlation(tgt_sig, pred_sig))
        
        # 频谱相干性 (信号频段平均)
        freq_coh, coh = _spectral_coherence(tgt_sig, pred_sig)
        signal_band = (freq_coh >= 100e3) & (freq_coh <= 500e3)
        avg_coh = float(np.mean(coh[signal_band])) if np.any(signal_band) else 0.0
        coherence_tp_list.append(avg_coh)
        
        # 到达时间误差
        fa = all_features[i]
        arr_tgt = fa['target']['arrival_time_us']
        arr_pred = fa['predicted']['arrival_time_us']
        if not np.isnan(arr_tgt) and not np.isnan(arr_pred):
            arrival_errors.append(abs(arr_tgt - arr_pred))
        
        # 峰值振幅保留度
        tgt_peak = fa['target']['peak_amplitude']
        pred_peak = fa['predicted']['peak_amplitude']
        if tgt_peak > 1e-10:
            peak_pres_list.append(min(pred_peak / tgt_peak, tgt_peak / pred_peak) * 100)
        
        # RMS 保留度
        tgt_rms = fa['target']['rms']
        pred_rms = fa['predicted']['rms']
        if tgt_rms > 1e-10:
            rms_pres_list.append(min(pred_rms / tgt_rms, tgt_rms / pred_rms) * 100)
        
        # 主频保留度
        tgt_df = fa['target']['dominant_freq_khz']
        pred_df = fa['predicted']['dominant_freq_khz']
        if tgt_df > 1e-3:
            dom_freq_pres_list.append(min(pred_df / tgt_df, tgt_df / pred_df) * 100)
        
        # 子频带能量分布匹配度 (余弦相似度)
        tgt_energy = np.array([fa['target'].get(f'energy_ratio_{l}', 0) for l in SUB_BAND_LABELS])
        pred_energy = np.array([fa['predicted'].get(f'energy_ratio_{l}', 0) for l in SUB_BAND_LABELS])
        denom_cos = np.linalg.norm(tgt_energy) * np.linalg.norm(pred_energy)
        if denom_cos > 1e-10:
            energy_match_list.append(float(np.dot(tgt_energy, pred_energy) / denom_cos) * 100)
    
    # 汇总
    quality_metrics = {
        'per_sample': {
            'xcorr_target_pred': xcorr_tp_list,
            'envelope_corr_target_pred': env_corr_tp_list,
            'avg_coherence_target_pred': coherence_tp_list,
        },
        'averaged': {
            'xcorr_target_pred': float(np.mean(xcorr_tp_list)) if xcorr_tp_list else 0.0,
            'envelope_corr_target_pred': float(np.mean(env_corr_tp_list)) if env_corr_tp_list else 0.0,
            'avg_coherence_target_pred': float(np.mean(coherence_tp_list)) if coherence_tp_list else 0.0,
            'arrival_time_error_us': float(np.mean(arrival_errors)) if arrival_errors else float('nan'),
            'peak_amplitude_preservation': float(np.mean(peak_pres_list)) if peak_pres_list else 0.0,
            'rms_preservation': float(np.mean(rms_pres_list)) if rms_pres_list else 0.0,
            'dominant_freq_preservation': float(np.mean(dom_freq_pres_list)) if dom_freq_pres_list else 0.0,
            'energy_distribution_match': float(np.mean(energy_match_list)) if energy_match_list else 0.0,
        }
    }
    
    # ------ Step 4: 生成可视化 ------
    _plot_validation_figure(
        inp_np, tgt_np, pred_np,
        all_features, quality_metrics,
        save_path
    )
    print(f"[INFO] Saved acoustic validation figure to {save_path}")
    
    # ------ Step 5: 打印文本报告 ------
    _print_report(quality_metrics)
    
    return {
        'features': all_features,
        'quality_metrics': quality_metrics,
    }


# ============================================================
# 推理模式验证 (无 clean target, 只有 input vs denoised)
# ============================================================
def _plot_inference_validation_figure(
    input_signals: np.ndarray,
    denoised_signals: np.ndarray,
    all_features: Dict,
    quality_metrics: Dict,
    save_path: str
) -> None:
    """
    生成推理模式的声学验证图 (6 个面板).
    
    与训练模式不同，此处没有 clean target，
    因此对比 input (去噪前) vs denoised (去噪后)。
    """
    fig = plt.figure(figsize=(22, 24))
    fig.suptitle(
        'Inference Acoustic Validation Report\n'
        'Comparing acoustic features: Before vs After denoising',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # 时间轴 (μs)
    time_us = np.linspace(0, 160, input_signals.shape[-1])
    
    # 选择代表性样本
    sample_idx = 0
    inp = input_signals[sample_idx]
    den = denoised_signals[sample_idx]
    
    # ---- Panel 1: 波形叠加对比 ----
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(time_us, inp, 'b-', alpha=0.5, linewidth=0.6, label='Input (before)')
    ax1.plot(time_us, den, 'r-', alpha=0.8, linewidth=0.9, label='Denoised (after)')
    ax1.set_title('Panel 1: Waveform Overlay', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 160)
    
    # ---- Panel 2: PSD 对比 ----
    ax2 = fig.add_subplot(3, 2, 2)
    for sig, label, color, ls in [
        (inp, 'Input', 'blue', '-'),
        (den, 'Denoised', 'red', '--'),
    ]:
        freqs, psd = _compute_psd(sig)
        ax2.semilogy(freqs / 1e3, psd, color=color, linestyle=ls,
                     alpha=0.8, linewidth=1.0, label=label)
    ax2.set_title('Panel 2: Power Spectral Density (PSD)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('PSD (V²/Hz)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=CENTER_FREQ / 1e3, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlim(0, SAMPLING_RATE / 2 / 1e3)
    
    # ---- Panel 3: 特征变化条形图 ----
    ax3 = fig.add_subplot(3, 2, 3)
    
    key_features = [
        'peak_amplitude', 'rms', 'crest_factor', 'zero_crossing_rate',
        'spectral_centroid_khz', 'dominant_freq_khz', 'bandwidth_3db_khz',
        'total_energy', 'phase_linearity', 'spectral_energy_concentration'
    ]
    feature_labels = [
        'Peak Amp', 'RMS', 'Crest\nFactor', 'ZCR',
        'Spectral\nCentroid', 'Dominant\nFreq', '-3dB BW',
        'Total\nEnergy', 'Phase\nLinearity', 'Energy\nConc'
    ]
    
    # 计算去噪后相对输入的比率 (%)
    change_rates = []
    for feat in key_features:
        inp_vals = [f['input'][feat] for f in all_features
                    if not np.isnan(f['input'].get(feat, float('nan')))]
        den_vals = [f['denoised'][feat] for f in all_features
                    if not np.isnan(f['denoised'].get(feat, float('nan')))]
        if inp_vals and den_vals:
            inp_mean = np.mean(inp_vals)
            den_mean = np.mean(den_vals)
            if abs(inp_mean) > 1e-10:
                rate = den_mean / inp_mean * 100
            else:
                rate = 100.0
        else:
            rate = 100.0
        change_rates.append(rate)
    
    colors_bar = ['#4CAF50' if 70 <= r <= 130 else '#FFC107' if 50 <= r <= 150 else '#F44336'
                  for r in change_rates]
    bars = ax3.bar(range(len(key_features)), change_rates, color=colors_bar,
                   edgecolor='black', linewidth=0.5)
    ax3.axhline(y=100, color='gray', linestyle='-', alpha=0.5, label='No change (100%)')
    ax3.axhline(y=130, color='orange', linestyle='--', alpha=0.3)
    ax3.axhline(y=70, color='orange', linestyle='--', alpha=0.3)
    ax3.set_xticks(range(len(key_features)))
    ax3.set_xticklabels(feature_labels, fontsize=7, rotation=0, ha='center')
    ax3.set_ylabel('Denoised / Input (%)')
    ax3.set_title('Panel 3: Feature Change After Denoising', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar_item, rate in zip(bars, change_rates):
        ax3.text(bar_item.get_x() + bar_item.get_width() / 2,
                 bar_item.get_height() + 1,
                 f'{rate:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # ---- Panel 4: 子频带能量对比 ----
    ax4 = fig.add_subplot(3, 2, 4)
    
    energy_data = {label: {'input': [], 'denoised': []} for label in SUB_BAND_LABELS}
    for f in all_features:
        for label in SUB_BAND_LABELS:
            key = f'energy_ratio_{label}'
            energy_data[label]['input'].append(f['input'].get(key, 0))
            energy_data[label]['denoised'].append(f['denoised'].get(key, 0))
    
    x_pos = np.arange(len(SUB_BAND_LABELS))
    width = 0.35
    
    inp_means = [np.mean(energy_data[l]['input']) for l in SUB_BAND_LABELS]
    den_means = [np.mean(energy_data[l]['denoised']) for l in SUB_BAND_LABELS]
    
    ax4.bar(x_pos - width / 2, inp_means, width, label='Input', color='#2196F3', alpha=0.8)
    ax4.bar(x_pos + width / 2, den_means, width, label='Denoised', color='#F44336', alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{l} Hz' for l in SUB_BAND_LABELS], fontsize=9)
    ax4.set_ylabel('Energy Ratio (%)')
    ax4.set_title('Panel 4: Sub-band Energy Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ---- Panel 5: 互相关 + 频谱相干性 ----
    ax5 = fig.add_subplot(3, 2, 5)
    
    freq_coh, coh = _spectral_coherence(inp, den)
    ax5.plot(freq_coh / 1e3, coh, 'r-', linewidth=1.0, alpha=0.8, label='Input ↔ Denoised')
    ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
    ax5.axvline(x=CENTER_FREQ / 1e3, color='gray', linestyle=':', alpha=0.5)
    ax5.set_title('Panel 5: Spectral Coherence (Input ↔ Denoised)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Frequency (kHz)')
    ax5.set_ylabel('Coherence')
    ax5.set_ylim(0, 1.05)
    ax5.set_xlim(0, SAMPLING_RATE / 2 / 1e3)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    signal_band = (freq_coh >= 100e3) & (freq_coh <= 500e3)
    avg_coh = np.mean(coh[signal_band]) if np.any(signal_band) else 0
    ax5.annotate(f'Avg coherence\n(100-500kHz): {avg_coh:.3f}',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 ha='right', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # ---- Panel 6: 综合统计表 ----
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    
    avg_metrics = quality_metrics['averaged']
    
    table_data = [
        ['Metric', 'Value', 'Assessment'],
        ['Cross-Correlation\n(Input ↔ Denoised)', f"{avg_metrics['xcorr']:.4f}",
         _assess(avg_metrics['xcorr'], 0.7, 0.4)],
        ['Envelope Correlation', f"{avg_metrics['envelope_corr']:.4f}",
         _assess(avg_metrics['envelope_corr'], 0.7, 0.4)],
        ['Spectral Coherence\n(100-500kHz)', f"{avg_metrics['avg_coherence']:.4f}",
         _assess(avg_metrics['avg_coherence'], 0.6, 0.3)],
        ['Dominant Freq\nPreservation', f"{avg_metrics['dominant_freq_preservation']:.1f}%",
         _assess(avg_metrics['dominant_freq_preservation'] / 100, 0.85, 0.65)],
        ['Energy Distribution\nSimilarity', f"{avg_metrics['energy_distribution_match']:.1f}%",
         _assess(avg_metrics['energy_distribution_match'] / 100, 0.8, 0.6)],
        ['Noise Reduction\n(RMS decrease)', f"{avg_metrics['noise_reduction_pct']:.1f}%",
         _assess(avg_metrics['noise_reduction_pct'] / 100, 0.1, 0.0)],
    ]
    
    table = ax6.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.25, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    for j in range(3):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(table_data)):
        assessment = table_data[i][2]
        if '✅' in assessment:
            table[i, 2].set_facecolor('#E8F5E9')
        elif '⚠️' in assessment:
            table[i, 2].set_facecolor('#FFF9C4')
        else:
            table[i, 2].set_facecolor('#FFEBEE')
    
    ax6.set_title('Panel 6: Summary Statistics',
                   fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _print_inference_report(quality_metrics: Dict) -> None:
    """打印推理模式的验证结果文本报告."""
    avg = quality_metrics['averaged']
    
    print("\n" + "=" * 60)
    print("  INFERENCE ACOUSTIC VALIDATION REPORT")
    print("  (Comparing: Input vs Denoised)")
    print("=" * 60)
    
    rows = [
        ("Cross-Correlation",         f"{avg['xcorr']:.4f}",
         _assess(avg['xcorr'], 0.7, 0.4)),
        ("Envelope Correlation",      f"{avg['envelope_corr']:.4f}",
         _assess(avg['envelope_corr'], 0.7, 0.4)),
        ("Spectral Coherence (avg)",  f"{avg['avg_coherence']:.4f}",
         _assess(avg['avg_coherence'], 0.6, 0.3)),
        ("Dominant Freq Preservation", f"{avg['dominant_freq_preservation']:.1f}%",
         _assess(avg['dominant_freq_preservation'] / 100, 0.85, 0.65)),
        ("Energy Dist Similarity",    f"{avg['energy_distribution_match']:.1f}%",
         _assess(avg['energy_distribution_match'] / 100, 0.8, 0.6)),
        ("Noise Reduction (RMS↓)",    f"{avg['noise_reduction_pct']:.1f}%",
         _assess(avg['noise_reduction_pct'] / 100, 0.1, 0.0)),
    ]
    
    print(f"  {'Metric':<30} {'Value':>12}  {'Status'}")
    print("-" * 60)
    for name, val, status in rows:
        print(f"  {name:<30} {val:>12}  {status}")
    
    print("=" * 60)
    
    good_count = sum(1 for _, _, s in rows if '✅' in s)
    total = len(rows)
    print(f"\n  Overall: {good_count}/{total} metrics passed ✅")
    
    if good_count == total:
        print("  → Excellent! Denoising preserved acoustic features well.")
    elif good_count >= total * 0.5:
        print("  → Reasonable. Denoising effective with some feature changes.")
    else:
        print("  → Caution: Significant acoustic feature changes after denoising.")
    
    print("=" * 60 + "\n")


def run_inference_validation(
    input_signals: np.ndarray,
    denoised_signals: np.ndarray,
    save_path: str = "fig_inference_acoustic_validation.png",
    num_samples: int = 20
) -> Dict:
    """
    推理模式的声学特征验证.
    
    对比去噪前 (input) 和去噪后 (denoised) 的声学特征，
    无需 clean target。适用于 inference.py 处理实验数据后调用。
    
    注意: 推理模式下的评估阈值与训练模式不同——
    因为没有 ground truth，互相关等指标的期望值更低
    (去噪会改变波形，所以完全相同反而说明没去噪)。
    
    Args:
        input_signals: (N, signal_length) 输入信号 (去噪前，已归一化)
        denoised_signals: (N, signal_length) 去噪后信号 (已归一化)
        save_path: 输出图片路径
        num_samples: 分析的样本数量
        
    Returns:
        特征和质量指标字典
    """
    total = input_signals.shape[0]
    n = min(num_samples, total)
    
    # 随机选取样本
    indices = np.random.choice(total, size=n, replace=False)
    inp_np = input_signals[indices]
    den_np = denoised_signals[indices]
    
    print(f"[INFO] Analyzing {n}/{total} signals for inference acoustic validation...")
    
    # ------ Step 1: 逐样本提取特征 ------
    all_features = []
    for i in range(n):
        sample_features = {
            'input': _extract_all_features(inp_np[i]),
            'denoised': _extract_all_features(den_np[i]),
        }
        all_features.append(sample_features)
    
    # ------ Step 2: 计算质量指标 ------
    xcorr_list = []
    env_corr_list = []
    coherence_list = []
    dom_freq_pres_list = []
    energy_match_list = []
    noise_reduction_list = []
    
    for i in range(n):
        inp_sig = inp_np[i]
        den_sig = den_np[i]
        
        # 互相关
        peak_corr, _ = _cross_correlation_peak(inp_sig, den_sig)
        xcorr_list.append(peak_corr)
        
        # 包络相关
        env_corr_list.append(_envelope_correlation(inp_sig, den_sig))
        
        # 频谱相干性
        freq_coh, coh = _spectral_coherence(inp_sig, den_sig)
        signal_band = (freq_coh >= 100e3) & (freq_coh <= 500e3)
        avg_coh = float(np.mean(coh[signal_band])) if np.any(signal_band) else 0.0
        coherence_list.append(avg_coh)
        
        fa = all_features[i]
        
        # 主频保留度
        inp_df = fa['input']['dominant_freq_khz']
        den_df = fa['denoised']['dominant_freq_khz']
        if inp_df > 1e-3 and den_df > 1e-3:
            dom_freq_pres_list.append(min(den_df / inp_df, inp_df / den_df) * 100)
        
        # 子频带能量匹配 (余弦相似度)
        inp_energy = np.array([fa['input'].get(f'energy_ratio_{l}', 0) for l in SUB_BAND_LABELS])
        den_energy = np.array([fa['denoised'].get(f'energy_ratio_{l}', 0) for l in SUB_BAND_LABELS])
        denom_cos = np.linalg.norm(inp_energy) * np.linalg.norm(den_energy)
        if denom_cos > 1e-10:
            energy_match_list.append(float(np.dot(inp_energy, den_energy) / denom_cos) * 100)
        
        # 噪声降低比 (RMS 减少百分比)
        inp_rms = fa['input']['rms']
        den_rms = fa['denoised']['rms']
        if inp_rms > 1e-10:
            noise_reduction_list.append((1.0 - den_rms / inp_rms) * 100)
    
    quality_metrics = {
        'per_sample': {
            'xcorr': xcorr_list,
            'envelope_corr': env_corr_list,
            'avg_coherence': coherence_list,
        },
        'averaged': {
            'xcorr': float(np.mean(xcorr_list)) if xcorr_list else 0.0,
            'envelope_corr': float(np.mean(env_corr_list)) if env_corr_list else 0.0,
            'avg_coherence': float(np.mean(coherence_list)) if coherence_list else 0.0,
            'dominant_freq_preservation': float(np.mean(dom_freq_pres_list)) if dom_freq_pres_list else 0.0,
            'energy_distribution_match': float(np.mean(energy_match_list)) if energy_match_list else 0.0,
            'noise_reduction_pct': float(np.mean(noise_reduction_list)) if noise_reduction_list else 0.0,
        }
    }
    
    # ------ Step 3: 可视化 ------
    _plot_inference_validation_figure(
        inp_np, den_np, all_features, quality_metrics, save_path
    )
    print(f"[INFO] Saved inference acoustic validation figure to {save_path}")
    
    # ------ Step 4: 文本报告 ------
    _print_inference_report(quality_metrics)
    
    return {
        'features': all_features,
        'quality_metrics': quality_metrics,
    }


# ============================================================
# Standalone 测试入口
# ============================================================
if __name__ == "__main__":
    """
    独立测试: 用未训练的模型 + 少量合成数据验证模块功能.
    
    Usage:
        uv run python acoustic_validation.py
    """
    from data_utils import create_dataloaders
    from model import DeepCAE
    
    print("[TEST] Running acoustic validation module standalone test...")
    
    # 创建未训练模型和小数据集
    model = DeepCAE(dropout_rate=0.1)
    device = torch.device('cpu')
    _, val_loader = create_dataloaders(num_val=50, batch_size=16, seed=42)
    
    results = run_acoustic_validation(
        model, val_loader, device,
        save_path='fig_acoustic_validation_test.png',
        num_samples=10
    )
    
    print(f"[TEST] Analyzed {len(results['features'])} samples")
    print("[TEST] ✓ Module test completed successfully!")
