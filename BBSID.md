https://blog.csdn.net/weixin_45871537/article/details/132402962

### OP765 
0e:52:b3:54:cf:ac
60信道 20带宽  PNABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==

/home/nexmon/patches/bcm4339/6_37_34_43/nexmon_csi/utils/makecsiparams# ./makecsiparams -c 149/80 -C 1 -N 1 -m 0e:52:b3:54:cf:ac
149 信道 80 带宽
m+ABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==


### NEXUS
ifconfig wlan0 up

nexutil -Iwlan0 -s500 -b -l34 -v1 -vm+ABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==   //启用 CSI 数据通过 UDP 上报
nexutil -Iwlan0 -s108 -v10 -l4 -i   //设置 NDP 发送间隔为 10ms
nexutil -k		//可以查看是否是157信道80带宽
nexutil -Iwlan0 -m1		//设置为monitor模式

```
# 1. 设置监控模式（如果还没设）
ifconfig wlan0 up
nexutil -Iwlan0 -m1

# 2. 设置信道（例如 36，80MHz）
nexutil -Iwlan0 -c36 -b1   # -b1 表示 80MHz 带宽

# 3. 【关键】启用 CSI 报告！
nexutil -Iwlan0 -s500 -b1 -l34 -v1
# 参数说明：
#   -s500 : 使用 UDP 端口 5500 发送 CSI（对应 tcpdump port 5500）
#   -b1   : 带宽 80MHz（必须与信道设置一致）
#   -l34  : 每个 CSI 包包含 34 个子载波（80MHz 下典型值）
#   -v1   : 启用 CSI 报告（version 1 格式）

# 4. 设置 NDP 发送间隔（10ms = 100Hz）
nexutil -k10

```

tcpdump -i wlan0 -v dst port 5500 -w /sdcard/csi-149-80.pcap -c 1000	//采集1000个数据包放在手机的sdcard中



