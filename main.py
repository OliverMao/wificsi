from read import ReadCSI 

if __name__ == "__main__":
    name = 'test_heart_beat_0'
    reader = ReadCSI(
        filepath=f'pcap/{name}.pcap',
        amplitude_spec=(0, 3000),
        subcarrier_spec=(5, 60),
        bandwidth=80  # 可以根据需要调整带宽
    )
    csi_data = reader.read(save_image=False, save_path=f'image/{name}.png')
    # 保存csi_data
    reader.save(save_path=f'test/{name}.npy')

    # 呼吸 - 使用投票方法
    breath_ = reader.get_breathing_analyzer()
    breathing_rate_fft = breath_.extract_breathing_rate()
    breathing_rate_vote, vote_info, all_rates = breath_.extract_breathing_rate_by_voting(vote_threshold=0.05)
    print(f'Breathing Rate (FFT单载波): {breathing_rate_fft:.1f} bpm')
    print(f'Breathing Rate (投票): {breathing_rate_vote:.1f} bpm')


    # 心跳 - 使用投票方法
    heartbeat_ = reader.get_heartbeat_analyzer()
    heartbeat_rate_fft = heartbeat_.extract_heartbeat_rate()
    heartbeat_rate_vote, vote_info, all_rates = heartbeat_.extract_heartbeat_rate_by_voting(vote_threshold=0.5)
    print(f'Heartbeat Rate (FFT单载波): {heartbeat_rate_fft:.1f} bpm')
    print(f'Heartbeat Rate (投票): {heartbeat_rate_vote:.1f} bpm')
