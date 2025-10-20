"""
实时CSI数据采集与分析脚本
通过adb执行tcpdump采集CSI数据，每次采集指定数量的包，
与之前的包合并，并行进行呼吸估计，直到达到总包数目标。

在开始采集前，会先执行nexutil命令配置CSI参数。

用法示例:
  # 采集1000个包，每批50个
  python csi_realtime_collect.py -t 1000 -b 50
  
  # 采集500个包，每批100个
  python csi_realtime_collect.py -t 500 -b 100
"""

import os
import sys
import time
import argparse
import subprocess
import threading
import numpy as np
from pathlib import Path
from read import ReadCSI


class RealtimeCSICollector:
    """实时CSI数据采集器"""
    
    def __init__(self, batch_size=50, total_packets=1000, 
                 interface="wlan0", dst_port=5500, bandwidth=80,
                 temp_dir="/sdcard", local_dir="./pcap_realtime"):
        """
        初始化采集器
        
        参数:
        - batch_size: 每批次采集的包数量
        - total_packets: 总目标包数量
        - interface: 网络接口名称
        - dst_port: 目标端口
        - bandwidth: 带宽设置（20/40/80/160 MHz）
        - temp_dir: 手机上的临时目录
        - local_dir: 本地保存目录
        """
        self.batch_size = batch_size
        self.total_packets = total_packets
        self.interface = interface
        self.dst_port = dst_port
        self.bandwidth = bandwidth
        self.temp_dir = temp_dir
        self.local_dir = local_dir
        
        # 创建本地目录
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # 状态变量
        self.current_batch = 0
        self.total_collected = 0
        self.merged_pcap_path = None
        self.analysis_thread = None
        self.latest_breathing_rate = None
        
    def execute_adb_command(self, command, timeout=30):
        """
        执行adb命令
        
        参数:
        - command: 要执行的命令
        - timeout: 超时时间（秒）
        
        返回:
        - (returncode, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            print(f"命令执行超时: {command}")
            return -1, "", "Timeout"
        except Exception as e:
            print(f"命令执行失败: {e}")
            return -1, "", str(e)
    
    def capture_batch(self, batch_num):
        """
        采集一批数据包
        
        参数:
        - batch_num: 批次号
        
        返回:
        - success: 是否成功
        - pcap_path: 本地pcap文件路径
        """
        remote_pcap = f"{self.temp_dir}/csi_batch_{batch_num}.pcap"
        local_pcap = os.path.join(self.local_dir, f"csi_batch_{batch_num}.pcap")
        
        # 构建tcpdump命令
        tcpdump_cmd = (
            f"tcpdump -i {self.interface} -v dst port {self.dst_port} "
            f"-w {remote_pcap} -c {self.batch_size}"
        )
        
        # 通过adb su执行tcpdump
        adb_cmd = f'adb shell "su -c \\"{tcpdump_cmd}\\""'
        
        print(f"\n[批次 {batch_num}] 开始采集 {self.batch_size} 个包...")
        print(f"执行命令: {adb_cmd}")
        
        start_time = time.time()
        returncode, stdout, stderr = self.execute_adb_command(adb_cmd, timeout=60)
        elapsed = time.time() - start_time
        
        if returncode != 0:
            print(f"[批次 {batch_num}] tcpdump执行失败")
            print(f"stderr: {stderr}")
            return False, None
        
        print(f"[批次 {batch_num}] 采集完成，耗时 {elapsed:.2f} 秒")
        
        # 拉取文件到本地
        pull_cmd = f"adb pull {remote_pcap} {local_pcap}"
        print(f"[批次 {batch_num}] 拉取文件到本地...")
        returncode, stdout, stderr = self.execute_adb_command(pull_cmd, timeout=30)
        
        if returncode != 0 or not os.path.exists(local_pcap):
            print(f"[批次 {batch_num}] 文件拉取失败")
            return False, None
        
        # 清理手机上的临时文件
        clean_cmd = f'adb shell "su -c \\"rm {remote_pcap}\\""'
        self.execute_adb_command(clean_cmd, timeout=10)
        
        print(f"[批次 {batch_num}] 文件已保存到: {local_pcap}")
        return True, local_pcap
    
    def merge_pcap_files(self, new_pcap):
        """
        合并pcap文件
        
        参数:
        - new_pcap: 新采集的pcap文件路径
        
        返回:
        - merged_path: 合并后的文件路径
        """
        merged_path = os.path.join(self.local_dir, "merged.pcap")
        
        if self.merged_pcap_path is None or not os.path.exists(self.merged_pcap_path):
            # 第一批数据，直接复制
            import shutil
            shutil.copy(new_pcap, merged_path)
            print(f"初始化合并文件: {merged_path}")
        else:
            # 使用mergecap合并文件（如果可用）
            # 否则使用简单的文件追加（仅适用于pcap格式）
            try:
                # 尝试使用mergecap
                merge_cmd = f"mergecap -w {merged_path}.tmp {self.merged_pcap_path} {new_pcap}"
                returncode, stdout, stderr = self.execute_adb_command(merge_cmd, timeout=30)
                
                if returncode == 0:
                    os.replace(f"{merged_path}.tmp", merged_path)
                    print(f"使用mergecap合并文件成功")
                else:
                    # mergecap不可用，使用Python方式合并
                    print("mergecap不可用，使用Python方式合并pcap文件...")
                    self._merge_pcap_python(self.merged_pcap_path, new_pcap, merged_path)
            except:
                # 回退到Python合并
                print("使用Python方式合并pcap文件...")
                self._merge_pcap_python(self.merged_pcap_path, new_pcap, merged_path)
        
        self.merged_pcap_path = merged_path
        return merged_path
    
    def _merge_pcap_python(self, pcap1, pcap2, output):
        """
        使用Python合并两个pcap文件（简单拼接）
        注意：这种方法假设两个文件具有相同的链路层类型
        """
        with open(pcap1, 'rb') as f1:
            data1 = f1.read()
        
        with open(pcap2, 'rb') as f2:
            # 跳过第二个文件的全局头（24字节）
            f2.seek(24)
            data2 = f2.read()
        
        # 写入合并后的文件
        with open(output, 'wb') as fout:
            fout.write(data1)
            fout.write(data2)
        
        print(f"Python合并完成: {output}")
    
    def analyze_breathing(self, pcap_path, batch_num):
        """
        分析呼吸频率（在独立线程中执行）
        
        参数:
        - pcap_path: pcap文件路径
        - batch_num: 批次号
        """
        try:
            print(f"\n[分析-批次{batch_num}] 开始呼吸频率估计...")
            
            # 读取CSI数据
            reader = ReadCSI(
                filepath=pcap_path,
                amplitude_spec=(0, 3000),
                subcarrier_spec=(5, 60),
                bandwidth=self.bandwidth
            )
            
            # 不保存图像，加快处理速度
            csi_data = reader.read(save_image=False)
            
            # 提取呼吸频率（使用投票方法）
            breath_analyzer = reader.get_breathing_analyzer()
            breathing_rate_vote, vote_info, all_rates = breath_analyzer.extract_breathing_rate_by_voting(
                vote_threshold=0.05,
                verbose=False  # 不打印详细信息
            )
            
            self.latest_breathing_rate = breathing_rate_vote
            
            print(f"[分析-批次{batch_num}] 呼吸频率: {breathing_rate_vote:.1f} bpm")
            print(f"[分析-批次{batch_num}] 当前总包数: {self.total_collected}")
            
        except Exception as e:
            print(f"[分析-批次{batch_num}] 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def start_analysis_thread(self, pcap_path, batch_num):
        """
        启动分析线程
        """
        # 等待上一个分析线程完成
        if self.analysis_thread is not None and self.analysis_thread.is_alive():
            print(f"[批次{batch_num}] 等待上一批次分析完成...")
            self.analysis_thread.join()
        
        # 启动新的分析线程
        self.analysis_thread = threading.Thread(
            target=self.analyze_breathing,
            args=(pcap_path, batch_num)
        )
        self.analysis_thread.start()
    
    def run(self):
        """
        运行采集流程
        """
        print("=" * 60)
        print("实时CSI数据采集与分析")
        print("=" * 60)
        print(f"每批次包数: {self.batch_size}")
        print(f"总目标包数: {self.total_packets}")
        print(f"网络接口: {self.interface}")
        print(f"目标端口: {self.dst_port}")
        print(f"带宽: {self.bandwidth} MHz")
        print("=" * 60)
        
        # 检查adb连接
        print("\n检查adb连接...")
        returncode, stdout, stderr = self.execute_adb_command("adb devices")
        if returncode != 0:
            print("错误: adb未安装或无法执行")
            return False
        
        if "device" not in stdout:
            print("错误: 未检测到设备连接")
            print(stdout)
            return False
        
        print("adb连接正常")
        
        # 检查su权限
        print("\n检查root权限...")
        returncode, stdout, stderr = self.execute_adb_command('adb shell "su -c id"')
        if returncode != 0 or "uid=0" not in stdout:
            print("错误: 设备未root或su权限获取失败")
            return False
        
        print("root权限正常")
        
        # 执行nexutil配置CSI
        print("\n配置CSI参数...")
        nexutil_cmd = 'adb shell "su -c \\"nexutil -I{} -s500 -b -l34 -v1 -vm+ABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==\\""'.format(self.interface)
        print(f"执行: nexutil -I{self.interface} -s500 -b -l34 -v1 -vm+ABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==")
        returncode, stdout, stderr = self.execute_adb_command(nexutil_cmd, timeout=10)
        if returncode != 0:
            print(f"警告: nexutil执行失败")
            print(f"stderr: {stderr}")
            print("继续尝试采集...")
        else:
            print("CSI配置成功")
        
        # 开始采集循环
        while self.total_collected < self.total_packets:
            self.current_batch += 1
            
            # 计算本批次应采集的包数
            remaining = self.total_packets - self.total_collected
            current_batch_size = min(self.batch_size, remaining)
            
            # 临时修改batch_size
            original_batch_size = self.batch_size
            self.batch_size = current_batch_size
            
            # 采集一批数据
            success, pcap_path = self.capture_batch(self.current_batch)
            
            # 恢复batch_size
            self.batch_size = original_batch_size
            
            if not success:
                print(f"\n批次 {self.current_batch} 采集失败，跳过")
                continue
            
            # 更新计数
            self.total_collected += current_batch_size
            
            # 合并pcap文件
            merged_path = self.merge_pcap_files(pcap_path)
            
            # 并行启动呼吸分析
            self.start_analysis_thread(merged_path, self.current_batch)
            
            print(f"\n进度: {self.total_collected}/{self.total_packets} 包")
            
            # 短暂延迟，避免过快请求
            if self.total_collected < self.total_packets:
                time.sleep(1)
        
        # 等待最后一个分析线程完成
        if self.analysis_thread is not None and self.analysis_thread.is_alive():
            print("\n等待最后一批次分析完成...")
            self.analysis_thread.join()
        
        print("\n" + "=" * 60)
        print("采集完成！")
        print("=" * 60)
        print(f"总采集包数: {self.total_collected}")
        print(f"合并文件路径: {self.merged_pcap_path}")
        if self.latest_breathing_rate is not None:
            print(f"最新呼吸频率估计: {self.latest_breathing_rate:.1f} bpm")
        print("=" * 60)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='实时CSI数据采集与分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 采集1000个包，每批50个
  python csi_realtime_collect.py -t 1000 -b 50
  
  # 采集500个包，每批100个，指定端口
  python csi_realtime_collect.py -t 500 -b 100 -p 5500
  
  # 自定义所有参数
  python csi_realtime_collect.py -t 1000 -b 50 -i wlan0 -p 5500 -w 80
        """
    )
    
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='每批次采集的包数量 (默认: 50)')
    parser.add_argument('-t', '--total-packets', type=int, default=1000,
                        help='总目标包数量 (默认: 1000)')
    parser.add_argument('-i', '--interface', type=str, default='wlan0',
                        help='网络接口名称 (默认: wlan0)')
    parser.add_argument('-p', '--port', type=int, default=5500,
                        help='目标端口 (默认: 5500)')
    parser.add_argument('-w', '--bandwidth', type=int, default=80,
                        choices=[20, 40, 80, 160],
                        help='带宽设置 (默认: 80 MHz)')
    parser.add_argument('--temp-dir', type=str, default='/sdcard',
                        help='手机上的临时目录 (默认: /sdcard)')
    parser.add_argument('--local-dir', type=str, default='./pcap_realtime',
                        help='本地保存目录 (默认: ./pcap_realtime)')
    
    args = parser.parse_args()
    
    # 创建采集器
    collector = RealtimeCSICollector(
        batch_size=args.batch_size,
        total_packets=args.total_packets,
        interface=args.interface,
        dst_port=args.port,
        bandwidth=args.bandwidth,
        temp_dir=args.temp_dir,
        local_dir=args.local_dir
    )
    
    # 运行采集
    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n\n用户中断采集")
        print(f"已采集 {collector.total_collected} 个包")
        if collector.merged_pcap_path:
            print(f"部分数据已保存到: {collector.merged_pcap_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
