
from read import ReadCSI 

if __name__ == "__main__":
    reader = ReadCSI(
        filepath='pcap/csi-149-80.pcap',
        amplitude_spec=(0, 2000),
        subcarrier_spec=(0, 60),
        bandwidth=80  # 可以根据需要调整带宽
    )
    csi_data = reader.read(save_image=True)
    # 保存csi_data

    reader.save(save_path='nexmon_csi_data.npy')