
from read import ReadCSI 

if __name__ == "__main__":
    # for i in range(0, 5):
    #     reader = ReadCSI(
    #         filepath=f'pcap/csi-149-80-{i}.pcap',
    #         amplitude_spec=(0, 2000),
    #         subcarrier_spec=(0, 60),
    #         bandwidth=80  # 可以根据需要调整带宽
    #     )
    #     csi_data = reader.read(save_image=False)
    #     # 保存csi_data

    #     reader.save(save_path=f'data/nexmon_csi_data_{i}.npy')    
    reader = ReadCSI(
        filepath=f'test/csi-149-80-test-no.pcap',
        amplitude_spec=(0, 2000),
        subcarrier_spec=(0, 60),
        bandwidth=80  # 可以根据需要调整带宽
    )
    csi_data = reader.read(save_image=False)
    # 保存csi_data

    reader.save(save_path=f'test/csi-149-80-test-no-man.npy')