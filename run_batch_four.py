#!/usr/bin/env bash
nohup python3 stackingcv_real.py second > nohup_stack_real_1.out 2>&1&
nohup python3 stackingcv_real.py third > nohup_stack_real_2.out 2>&1&
nohup python3 stackingcv_real.py fourth > nohup_stack_real_3.out 2>&1&
