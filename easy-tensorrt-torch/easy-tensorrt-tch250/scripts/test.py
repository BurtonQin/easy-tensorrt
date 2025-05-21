#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import easy_tensorrt_tch250 as easy_trt
import time

def main():
    env = easy_trt.CudaTRTEnvWrapper(0)
    trt_engine = easy_trt.TchTrtEngineWrapper(
        env,
        r"model.engine",
    )
    print(trt_engine.info())
    inputs1 = torch.zeros([2, 3, 512, 512]).cuda()

    outputs = trt_engine.inference([inputs1])
    for i, output in enumerate(outputs):
        print(i, output.shape)
    time1 = time.time()
    for i in range(100):
        outputs = trt_engine.inference([inputs1])
        torch.cuda.synchronize()
    print(f"{(time.time()-time1)/100:.7f}")
if __name__ == '__main__':
    main()