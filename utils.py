
import scipy
import numpy as np
import matplotlib.pyplot as plt

class BreathingAnalyzer(object):
    """
    呼吸分析器类，负责提取和可视化呼吸信号。
    """
    
    def __init__(self, csi, samples):
        self.csi = csi
        self.samples = samples

    def _get_filtered_signal(self, breathing_freq_range=(0.1, 0.5)):
        """
        私有方法：计算滤波后的呼吸信号和相关参数。
        使用SVD提取第一主成分作为信号。
        
        返回:
        - filtered_signal: 滤波后的信号
        - time_axis: 时间轴
        - sampling_rate: 采样率
        """
        # 计算时间轴和采样率
        timestamps = self.samples['ts_sec'] + self.samples['ts_usec'] / 1e6
        time_axis = timestamps - timestamps[0]
        sampling_rate = 1 / np.mean(np.diff(timestamps))
        
        # 使用SVD提取第一主成分
        # self.csi shape: (n_samples, n_subcarriers)
        U, S, Vt = np.linalg.svd(self.csi, full_matrices=False)
        # U shape: (n_samples, n_subcarriers), S shape: (n_subcarriers,)
        # 第一主成分是U的第一列乘以最大奇异值
        principal_component = U[:, 0] * S[0]
        
        # 带通滤波
        low_freq = breathing_freq_range[0]
        high_freq = breathing_freq_range[1]
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        filtered_signal = scipy.signal.filtfilt(b, a, principal_component)
        
        return filtered_signal, time_axis, sampling_rate

    def extract_breathing_rate(self, breathing_freq_range=(0.1, 0.5)):
        """
        从CSI样本中提取呼吸频率。
        
        参数:
        - breathing_freq_range: 呼吸频率范围（Hz），默认0.1-0.5 Hz
        
        返回:
        - breathing_rate: 估计的呼吸频率（次/分钟）
        """
        filtered_signal, _, sampling_rate = self._get_filtered_signal(breathing_freq_range)
        
        # FFT计算频率
        n_samples = len(filtered_signal)
        freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)
        fft_magnitudes = np.abs(np.fft.fft(filtered_signal))
        
        # 只考虑正频率
        positive_freqs = freqs[:n_samples//2]
        positive_magnitudes = fft_magnitudes[:n_samples//2]
        
        # 在呼吸频率范围内找到峰值
        mask = (positive_freqs >= breathing_freq_range[0]) & (positive_freqs <= breathing_freq_range[1])
        if not np.any(mask):
            return 0
        
        peak_idx = np.argmax(positive_magnitudes[mask])
        breathing_freq = positive_freqs[mask][peak_idx]
        
        # 转换为次/分钟
        breathing_rate = breathing_freq * 60
        return breathing_rate

    def extract_breathing_rate_by_voting(self, breathing_freq_range=(0.1, 0.5), vote_threshold=0.05):
        """
        对每个子载波分别提取呼吸频率，通过投票选举最可靠的频率。
        
        参数:
        - breathing_freq_range: 呼吸频率范围（Hz），默认0.1-0.5 Hz
        - vote_threshold: 投票阈值（Hz），频率在此范围内视为同一票
        
        返回:
        - voting_rate: 投票得出的呼吸率（次/分钟）
        - vote_counts: 各频率的投票数
        - all_rates: 所有子载波的呼吸率列表
        """
        timestamps = self.samples['ts_sec'] + self.samples['ts_usec'] / 1e6
        sampling_rate = 1 / np.mean(np.diff(timestamps))
        
        all_rates = []
        
        # 对每个子载波提取呼吸频率
        for s in range(self.csi.shape[1]):
            signal = self.csi[:, s]
            
            # 带通滤波
            low_freq = breathing_freq_range[0]
            high_freq = breathing_freq_range[1]
            nyquist = sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            if low >= 1 or high >= 1:  # 避免无效滤波参数
                continue
            
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered = scipy.signal.filtfilt(b, a, signal)
            
            # FFT计算频率
            n_samples = len(filtered)
            freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)
            fft_magnitudes = np.abs(np.fft.fft(filtered))
            
            positive_freqs = freqs[:n_samples//2]
            positive_magnitudes = fft_magnitudes[:n_samples//2]
            
            # 在呼吸频率范围内找到峰值
            mask = (positive_freqs >= breathing_freq_range[0]) & (positive_freqs <= breathing_freq_range[1])
            if not np.any(mask):
                continue
            
            peak_idx = np.argmax(positive_magnitudes[mask])
            breathing_freq = positive_freqs[mask][peak_idx]
            breathing_rate = breathing_freq * 60
            all_rates.append(breathing_rate)
        
        if not all_rates:
            return 0, {}, []
        
        # 投票机制：将相近频率聚类
        all_rates = np.array(all_rates)
        sorted_rates = np.sort(all_rates)
        
        # 使用简单的聚类投票：找到最密集的频率簇
        vote_counts = {}
        for rate in sorted_rates:
            # 找到与该频率相近的所有速率
            cluster = sorted_rates[np.abs(sorted_rates - rate) <= vote_threshold]
            cluster_center = np.mean(cluster)
            cluster_size = len(cluster)
            
            # 记录簇的投票数
            if cluster_size not in vote_counts:
                vote_counts[cluster_size] = cluster_center
        
        # 选择投票数最多的频率
        if vote_counts:
            max_votes = max(vote_counts.keys())
            voting_rate = vote_counts[max_votes]
        else:
            voting_rate = np.median(all_rates)
        
        print(f"呼吸率提取统计:")
        print(f"  子载波数: {len(all_rates)}")
        print(f"  所有子载波呼吸率范围: {np.min(all_rates):.1f} - {np.max(all_rates):.1f} bpm")
        print(f"  投票结果: {voting_rate:.1f} bpm (投票数: {max(vote_counts.keys())})")
        print(f"  中位数: {np.median(all_rates):.1f} bpm")
        print(f"  平均值: {np.mean(all_rates):.1f} bpm")
        
        # 绘制分布图
        plt.figure(figsize=(8, 6))
        plt.hist(all_rates, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Breathing Rate (bpm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Breathing Rates Across Subcarriers')
        plt.grid(True)
        plt.savefig('breathing_distribution.png', dpi=300)
        plt.show()
        
        return voting_rate, vote_counts, all_rates



class HeartbeatAnalyzer(object):
    """
    心跳分析器类，负责提取和可视化心跳信号。
    """
    
    def __init__(self, csi, samples):
        self.csi = csi
        self.samples = samples

    def _get_filtered_signal(self, heartbeat_freq_range=(0.75, 3.0)):
        """
        私有方法：计算滤波后的心跳信号和相关参数。
        使用SVD提取第一主成分作为信号。
        
        返回:
        - filtered_signal: 滤波后的信号
        - time_axis: 时间轴
        - sampling_rate: 采样率
        """
        # 计算时间轴和采样率
        timestamps = self.samples['ts_sec'] + self.samples['ts_usec'] / 1e6
        time_axis = timestamps - timestamps[0]
        sampling_rate = 1 / np.mean(np.diff(timestamps))
        
        # 使用SVD提取第一主成分
        # self.csi shape: (n_samples, n_subcarriers)
        U, S, Vt = np.linalg.svd(self.csi, full_matrices=False)
        # U shape: (n_samples, n_subcarriers), S shape: (n_subcarriers,)
        # 第一主成分是U的第一列乘以最大奇异值
        principal_component = U[:, 0] * S[0]
        
        # 带通滤波
        low_freq = heartbeat_freq_range[0]
        high_freq = heartbeat_freq_range[1]
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        filtered_signal = scipy.signal.filtfilt(b, a, principal_component)
        
        return filtered_signal, time_axis, sampling_rate

    def extract_heartbeat_rate(self, heartbeat_freq_range=(0.75, 3.0)):
        """
        从CSI样本中提取心跳频率。
        
        参数:
        - heartbeat_freq_range: 心跳频率范围（Hz），默认0.75-3.0 Hz (45-180 bpm)
        
        返回:
        - heartbeat_rate: 估计的心跳频率（次/分钟）
        """
        filtered_signal, _, sampling_rate = self._get_filtered_signal(heartbeat_freq_range)
        
        # FFT计算频率
        n_samples = len(filtered_signal)
        freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)
        fft_magnitudes = np.abs(np.fft.fft(filtered_signal))
        
        # 只考虑正频率
        positive_freqs = freqs[:n_samples//2]
        positive_magnitudes = fft_magnitudes[:n_samples//2]
        
        # 在心跳频率范围内找到峰值
        mask = (positive_freqs >= heartbeat_freq_range[0]) & (positive_freqs <= heartbeat_freq_range[1])
        if not np.any(mask):
            return 0
        
        peak_idx = np.argmax(positive_magnitudes[mask])
        heartbeat_freq = positive_freqs[mask][peak_idx]
        
        # 转换为次/分钟
        heartbeat_rate = heartbeat_freq * 60
        return heartbeat_rate

    def extract_heartbeat_rate_by_voting(self, heartbeat_freq_range=(0.75, 3.0), vote_threshold=0.5):
        """
        对每个子载波分别提取心跳频率，通过投票选举最可靠的频率。
        
        参数:
        - heartbeat_freq_range: 心跳频率范围（Hz），默认0.75-3.0 Hz (45-180 bpm)
        - vote_threshold: 投票阈值（bpm），频率在此范围内视为同一票
        
        返回:
        - voting_rate: 投票得出的心跳率（次/分钟）
        - vote_counts: 各频率的投票数
        - all_rates: 所有子载波的心跳率列表
        """
        timestamps = self.samples['ts_sec'] + self.samples['ts_usec'] / 1e6
        sampling_rate = 1 / np.mean(np.diff(timestamps))
        
        all_rates = []
        
        # 对每个子载波提取心跳频率
        for s in range(self.csi.shape[1]):
            signal = self.csi[:, s]
            
            # 带通滤波
            low_freq = heartbeat_freq_range[0]
            high_freq = heartbeat_freq_range[1]
            nyquist = sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            if low >= 1 or high >= 1:  # 避免无效滤波参数
                continue
            
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered = scipy.signal.filtfilt(b, a, signal)
            
            # FFT计算频率
            n_samples = len(filtered)
            freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)
            fft_magnitudes = np.abs(np.fft.fft(filtered))
            
            positive_freqs = freqs[:n_samples//2]
            positive_magnitudes = fft_magnitudes[:n_samples//2]
            
            # 在心跳频率范围内找到峰值
            mask = (positive_freqs >= heartbeat_freq_range[0]) & (positive_freqs <= heartbeat_freq_range[1])
            if not np.any(mask):
                continue
            
            peak_idx = np.argmax(positive_magnitudes[mask])
            heartbeat_freq = positive_freqs[mask][peak_idx]
            heartbeat_rate = heartbeat_freq * 60
            all_rates.append(heartbeat_rate)
        
        if not all_rates:
            return 0, {}, []
        
        # 投票机制：将相近频率聚类
        all_rates = np.array(all_rates)
        sorted_rates = np.sort(all_rates)
        
        # 使用简单的聚类投票：找到最密集的频率簇
        vote_counts = {}
        for rate in sorted_rates:
            # 找到与该频率相近的所有速率
            cluster = sorted_rates[np.abs(sorted_rates - rate) <= vote_threshold]
            cluster_center = np.mean(cluster)
            cluster_size = len(cluster)
            
            # 记录簇的投票数
            if cluster_size not in vote_counts:
                vote_counts[cluster_size] = cluster_center
        
        # 选择投票数最多的频率
        if vote_counts:
            max_votes = max(vote_counts.keys())
            voting_rate = vote_counts[max_votes]
        else:
            voting_rate = np.median(all_rates)
        
        print(f"心跳率提取统计:")
        print(f"  子载波数: {len(all_rates)}")
        print(f"  所有子载波心跳率范围: {np.min(all_rates):.1f} - {np.max(all_rates):.1f} bpm")
        print(f"  投票结果: {voting_rate:.1f} bpm (投票数: {max(vote_counts.keys())})")
        print(f"  中位数: {np.median(all_rates):.1f} bpm")
        print(f"  平均值: {np.mean(all_rates):.1f} bpm")
        
        # 绘制分布图
        plt.figure(figsize=(8, 6))
        plt.hist(all_rates, bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Heartbeat Rate (bpm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Heartbeat Rates Across Subcarriers')
        plt.grid(True)
        plt.savefig('heartbeat_distribution.png', dpi=300)
        plt.show()
        
        return voting_rate, vote_counts, all_rates
