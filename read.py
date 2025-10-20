"""
Interleaved
===========

用于从PCAP文件中高效提取交错CSI样本的方法。

~640k 80MHz采样点每秒。

适用于bcm43455c0和bcm4339芯片。

需要Numpy。

用法
-----

from nexcsi import interleaved

samples = interleaved.read_pcap('path_to_pcap_file')

带宽会自动从pcap文件推断，也可以手动指定：

samples = interleaved.read_pcap('path_to_pcap_file', bandwidth=40)
"""


import datetime
import os
import time
import numpy as np
import sys
import scipy.signal  # 添加导入用于FFT和峰值检测

import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端，避免线程问题
import matplotlib.pyplot as plt

from nexcsi import nulls, pilots
from matplotlib import colors
from scipy.stats import median_abs_deviation
import pywt

from utils import BreathingAnalyzer, HeartbeatAnalyzer


# 设置matplotlib中文字体（适配Windows常见字体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def __find_nsamples_max(pcap_filesize, nsub):
    """
    估算pcap文件中可能包含的最大样本数。

    用pcap文件大小除以每个数据包的大小来计算样本数。
    但有些包有几个字节的填充，所以返回值会略大于实际样本数。
    """

    # PCAP全局头24字节
    # PCAP包头12字节
    # 以太网+IP+UDP头共46字节
    # Nexmon元数据18字节
    # CSI为nsub*4字节
    #
    # 所以每个包为12 + 46 + 18 + nsub*4字节
    nsamples_max = int((pcap_filesize - 24) / (12 + 46 + 18 + (nsub * 4)))

    return nsamples_max

def read_pcap(pcap_filepath, bandwidth=None, nsamples_max=None):

    """
    从pcap文件读取CSI样本，返回Numpy结构化数组。

    带宽和最大样本数默认会自动推断，也可以手动指定。
    """

    pcap_filesize = os.stat(pcap_filepath).st_size

    with open(pcap_filepath, "rb") as pcapfile:
        fc = pcapfile.read()  # ~2.68 s

    # 保留带宽检测输出，但不要退出函数，以便继续解析样本
    print(f"Detected bandwidth: {bandwidth} MHz")
    # Number of OFDM sub-carriers
    nsub = int(bandwidth * 3.2)

    if nsamples_max is None:
        nsamples_max = __find_nsamples_max(pcap_filesize, nsub)

    # Numpy样本数据类型: https://numpy.org/doc/stable/reference/arrays.dtypes.html
    dtype_sample = np.dtype(
        [
            ("ts_sec",  np.uint32),
            ("ts_usec", np.uint32),
            ("saddr", np.dtype(np.uint32).newbyteorder('>')),
            ("daddr", np.dtype(np.uint32).newbyteorder('>')),
            ("sport", np.dtype(np.uint16).newbyteorder('>')),
            ("dport", np.dtype(np.uint16).newbyteorder('>')),
            ("magic", np.uint16),
            ("rssi", np.int8),
            ("fctl", np.uint8),
            ("mac", np.uint8, 6),
            ("seq", np.uint16),
            ("css", np.uint16),
            ("csp", np.uint16),
            ("cvr", np.uint16),
            ("csi", np.int16, nsub * 2),
        ],
    # 这些元数据不会在所有数组操作中保留。
    # 访问这些值时请务必小心。
        metadata={
            'bandwidth': bandwidth,
            'pcap_filepath': pcap_filepath,
            'nulls': nulls[bandwidth],
            'pilots': pilots[bandwidth],
        }
    )

    # 每个样本的字节数
    nbytes_sample = dtype_sample.itemsize

    # 预分配内存以容纳所有样本
    data = bytearray(nsamples_max * nbytes_sample)  # ~ 1.5s

    # 用于记录当前在bytearray `data`中的写入位置
    data_index = 0

    # 文件中的当前位置指针。
    # 比file.tell()更快。
    # =24跳过pcap全局头
    ptr = 24

    nsamples = 0
    while ptr < pcap_filesize:

        frame_len = int.from_bytes(  # ~ 3 s
            fc[ptr + 8: ptr + 12], byteorder="little", signed=False
        )

        # 读取时间戳
        data[data_index: data_index + 8] = fc[ptr: ptr + 8]

         # 读取saddr, daddr, sport, dport
        data[data_index + 8: data_index + 20] = fc[ptr + 42: ptr + 54]

        ptr += 58  # 跳过头部、以太网、IP、UDP

        data[data_index + 20: data_index + nbytes_sample] = fc[
                ptr: ptr + nbytes_sample - 20
        ]  # ~ 5.2 s

        nsamples += 1
        ptr += frame_len - 42
        data_index += nbytes_sample

    samples = np.frombuffer(
        data[:data_index], dtype=dtype_sample, count=nsamples
    )  # ~ 1.8 s

    return samples

def unpack(csi, device, fftshift=True, zero_nulls=False, zero_pilots=False):
    """
    将CSI样本从原始包内格式转换为可用于数学运算的Complex64类型。

    device参数应为raspberry或nexus5。

    如果不关心子载波顺序，可将fftshift设为False以加快速度。
    """
    unpacked = csi.astype(np.float32).view(np.complex64)

    unpacked = np.asmatrix(unpacked)

    if unpacked.shape[1] == 64:
        bandwidth = 20
    elif unpacked.shape[1] == 128:
        bandwidth = 40
    elif unpacked.shape[1] == 256:
        bandwidth = 80
    elif unpacked.shape[1] == 512:
        bandwidth = 160
    else:
        raise ValueError("Couldn't determine bandwidth. Is the packet corrupt? " +
            "Please create a new Issue: https://github.com/nexmonster/nexcsi/issues")
    
    if (zero_nulls or zero_pilots) and not fftshift:
        import warnings
        warnings.warn("FFTshift is automatically enabled when dropping pilots or nulls. Set fftshift to True to silence this warning.")
        fftshift = True

    if fftshift:
        unpacked = np.fft.fftshift(unpacked, axes=(1,))

    if zero_nulls:
        unpacked[:, nulls[bandwidth]] = 0
    
    if zero_pilots:
        unpacked[:, pilots[bandwidth]] = 0

    # 这些元数据不会在所有数组操作中保留。
    # 访问这些值时请务必小心。
    dt = np.dtype(unpacked.dtype, metadata={
        'device': device,
        'nulls': nulls[bandwidth],
        'pilots': pilots[bandwidth],
        'bandwidth': bandwidth,
        'fftshift': fftshift,
        'zero_nulls': zero_pilots,
        'zero_pilots': zero_nulls,
    })

    return unpacked.astype(dt)

def hampel_filter_for_csi(csi_amp, k=5, threshold=3):
    N, S = csi_amp.shape
    filtered = csi_amp.copy()
    outliers = np.zeros_like(csi_amp, dtype=bool)
    
    for s in range(S):  # 对每个子载波
        series = csi_amp[:, s]
        for i in range(k, N - k):
            window = series[i - k : i + k + 1]
            med = np.median(window)
            mad = median_abs_deviation(window)
            if mad == 0:
                mad = 1e-8
            sigma = 1.4826 * mad
            if abs(series[i] - med) > threshold * sigma:
                filtered[i, s] = med
                outliers[i, s] = True
    return filtered, outliers

def dwt_denoise(signal, wavelet='db4', level=2, threshold_type='soft'):
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 对细节系数阈值处理
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 用最高频估计噪声
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    coeffs_thresh = []
    for i in range(1, len(coeffs)):  # 跳过近似系数 A
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=threshold_type))
    coeffs_thresh.insert(0, coeffs[0])  # 保留低频近似
    
    # 重构
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(signal)]  # 避免边界延拓导致长度变化


class ReadCSI(object):

    csi = None

    def __init__(self, filepath='pcap/149-80.pcap', amplitude_spec=(0,1000), subcarrier_spec=(0,60), bandwidth=80):
        self.filepath = filepath
        self.amplitude_spec = amplitude_spec
        self.subcarrier_spec = subcarrier_spec
        self.bandwidth = bandwidth
        self.samples = None

    def read(self, save_image=False, save_path=None):

        pcap_path = self.filepath
        # 默认带宽为80MHz
        samples = read_pcap(pcap_path, bandwidth=self.bandwidth)
        print("样本数:", len(samples))
        
        # 计算并打印采样率
        timestamps = samples['ts_sec'] + samples['ts_usec'] / 1e6
        sampling_rate = 1 / np.mean(np.diff(timestamps))
        print(f"采样率: {sampling_rate:.2f} Hz")
        
        self.samples = samples
        # unpack所有样本的CSI
        csi_raw = samples['csi']
        # 解包得到 complex64 格式的 CSI 矩阵 (n_samples, n_subcarriers)
        csi_unpacked = unpack(csi_raw, device='nexus5', fftshift=True)
        # 转为常规 ndarray（以便后续处理）    
        csi_arr = np.asarray(csi_unpacked).astype(np.complex64)

        # 读取数据后立即应用子载波范围裁剪
        # start = self.subcarrier_spec[0] if self.subcarrier_spec[0] is not None else 0
        # end = self.subcarrier_spec[1] if self.subcarrier_spec[1] is not None else csi_arr.shape[1]
        # csi_arr = csi_arr[:, start:end]

        # if save_image:
        #     # 读取数据后立即展示 CSI 概览图（热图 + 单子载波折线）
        #     self.plot_csi_overview(csi_arr, subcarrier=15, cmap='jet')

        # 计算幅度并应用Hampel滤波去除异常值
        csi_amp = np.abs(csi_arr)
        csi_amp_filtered, outliers = hampel_filter_for_csi(csi_amp)
        print(f"检测到异常值数量: {np.sum(outliers)}")
        # if save_image:
        #     # 使用Hampel滤波后的幅度
        #     self.plot_csi_overview(csi_amp_filtered, subcarrier=15, cmap='jet')

        # 在异常值处理后应用小波去噪
        csi_amp_denoised = np.zeros_like(csi_amp_filtered)
        for s in range(csi_amp_filtered.shape[1]):
            csi_amp_denoised[:, s] = dwt_denoise(csi_amp_filtered[:, s])
        # if save_image:
            # 使用去噪后的幅度（数据已预裁剪）
        if save_image and save_path is not None:
            self.plot_csi_overview(csi_amp_denoised, subcarrier=15, cmap='jet',save_path=save_path)
        
        self.csi = csi_amp_denoised
        return self.csi

    def save(self, save_path='nexmon_csi_read.npy'):
        np.save(save_path, self.csi)
        print(f"CSI数据已保存到 {save_path}")

    def plot_csi_overview(self, csi, subcarrier=None, cmap='jet', figsize=(14, 5), save_path=None):
        """
        绘制 CSI 概览，样式按示例：
        - 上：CSI 幅度热力图（y=子载波，x=包序号，右侧颜色条），子载波从上到下（origin='upper'）
        - 中：指定子载波随时间的幅度折线图
        - 下：各子载波强度方差的线图
        - walking_range: 若提供 (start_idx, end_idx)，在热图下方绘制双向括号并标注 'Walking'
        """
        vmax = self.amplitude_spec[1]
        vmin = self.amplitude_spec[0]
        # 移除子载波切片逻辑，因为数据已预裁剪；保留参数以防需要进一步调整
        arr = np.asarray(csi)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        # 硬裁剪数据到 vmin-vmax
        amp = np.clip(np.abs(arr), vmin, vmax)
        n_samples, N = amp.shape

        if subcarrier is None:
            sub_idx = N // 2
        else:
            # subcarrier 是裁剪后索引
            original_sub = int(subcarrier)
            if original_sub < 0 or original_sub >= N:
                sub_idx = N // 2  # 默认居中
            else:
                sub_idx = original_sub

        # 布局：只保留热力图
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.03])
        ax_heat = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        # extent 显示裁剪后子载波索引，从 0 到 N-1
        im = ax_heat.imshow(amp.T, aspect='auto', origin='lower',
                            extent=[0, n_samples, 0, N-1], cmap=cmap, norm=norm)
        ax_heat.set_ylabel('子载波索引')
        ax_heat.set_xlabel('包序号')
        ax_heat.set_title(f'CSI 幅度热图 (all subcarriers)')
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.set_label('幅度')
        try:
            cb.set_ticks([vmin, vmax])
        except Exception:
            pass
        im.set_clim(vmin, vmax)

        plt.tight_layout()
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"热图已保存到: {save_path}")
        plt.close()  # 关闭图形以释放内存

    def get_breathing_analyzer(self):
        """
        返回BreathingAnalyzer实例，用于呼吸分析。
        """
        if self.csi is None or self.samples is None:
            raise ValueError("请先调用read()方法读取和处理CSI数据。")
        return BreathingAnalyzer(self.csi, self.samples)

    def get_heartbeat_analyzer(self):
        """
        返回HeartbeatAnalyzer实例，用于心跳分析。
        """
        if self.csi is None or self.samples is None:
            raise ValueError("请先调用read()方法读取和处理CSI数据。")
        return HeartbeatAnalyzer(self.csi, self.samples)

