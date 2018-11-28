#!/usr/bin/env bash
nohup python3 stackingcv_feat_sub.py first > nohup_stack_feat_1.out 2>&1&
nohup python3 stackingcv_feat_sub.py second> nohup_stack_feat_2.out 2>&1&
nohup python3 stackingcv_feat_sub.py third > nohup_stack_feat_3.out 2>&1&
nohup python3 deep_learning.py > nohup_dl_v2.out 2>&1&