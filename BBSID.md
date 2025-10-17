

### OP765 
0e:52:b3:54:cf:ac
60信道 20带宽  PNABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==

/home/nexmon/patches/bcm4339/6_37_34_43/nexmon_csi/utils/makecsiparams# ./makecsiparams -c 149/80 -C 1 -N 1 -m 0e:52:b3:54:cf:ac
149 信道 80 带宽
m+ABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==


### NEXUS
nexutil -Iwlan0 -s500 -b -l34 -vm+ABEQAAAQAOUrNUz6wAAAAAAAAAAAAAAAAAAAAAAAAAAA==

nexutil -k		//可以查看是否是157信道80带宽
nexutil -Iwlan0 -m1		//设置为monitor模式

采集数据
tcpdump -i wlan0 -v dst port 5500 -w /sdcard/csi-147-80.pcap -c 1000	//采集1000个数据包放在手机的sdcard中