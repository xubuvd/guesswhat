

#Research on when to stop dialogue, 2019-11-22

#QGen     - VDST
#Guesser  - GST, removed in this work
#Oracle   - xxx

#================Oracle=======================
pw@dell-PowerEdge-T630:~/guesswhat$ nohup bash Train_Oracle.sh > log_oracle_20200312_01.log 2>&1 &
[2] 24083, baseline version, v0.0
GPU0




#===============GST based Guesser model============
pw@dell-PowerEdge-T630:~/guesswhat$ nohup bash Play_Game.sh > p_GST_guesser_20200723_01.log 2>&1 &
[1] 18820, gpuid=0 for ECCV 2020 
>>>-------------- FINAL SCORE ---------------------<<<
INFO:tensorflow:Restoring parameters from out/loop/3de23c468246bee56ce1c0bbf5a4a6ed/params.ckpt
Restoring parameters from out/loop/3de23c468246bee56ce1c0bbf5a4a6ed/params.ckpt
>>>  New Objects  <<<
100%|██████████| 732/732 [06:00<00:00,  2.82it/s]
Accuracy (train - greedy): 0.8323289310595375
ErroRate (train - greedy): 0.16767106894046246
>>> valid set <<<
100%|██████████| 154/154 [01:11<00:00,  3.19it/s]
Accuracy (valid - greedy): 0.8266964648516864
ErroRate (valid - greedy): 0.17330353514831365
>>>  New Games  <<<
100%|██████████| 372/372 [02:57<00:00,  3.68it/s]
Accuracy (test - greedy): 0.815471936094177
ErroRate (test - greedy): 0.184528063905823
>>>------------------------------------------------<<<
Successed at 1 times

pw@dell-PowerEdge-T630:~/guesswhat$ nohup bash Play_Game.sh > p_GST_guesser_20200723_02.log 2>&1 &
[1] 19603, gpuid=0 for ECCV 2020
>>>  New Objects  <<<
100%|██████████| 732/732 [05:28<00:00,  3.99it/s]
Accuracy (train - greedy): 0.8344232166517075
ErroRate (train - greedy): 0.1655767833482925
>>> valid set <<<
100%|██████████| 154/154 [01:03<00:00,  3.57it/s]
Accuracy (valid - greedy): 0.8300487606663958
ErroRate (valid - greedy): 0.16995123933360423
>>>  New Games  <<<
100%|██████████| 372/372 [02:44<00:00,  3.73it/s]
Accuracy (test - greedy): 0.815471936094177
ErroRate (test - greedy): 0.184528063905823
>>>------------------------------------------------<<<
Successed at 1 times

pw@dell-PowerEdge-T630:~/guesswhat$ nohup bash Play_Game.sh > p_GST_guesser_20200723_03.log 2>&1 &
[1] 20228, gpuid=0 for ECCV 2020


