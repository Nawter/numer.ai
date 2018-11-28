#!/usr/bin/env bash
nohup python3 stackingcv_random_search.py first > nohup_stack_r_1.out 2>&1&
nohup python3 stackingcv_random_search.py second> nohup_stack_r_2.out 2>&1&
nohup python3 stackingcv_random_search.py third > nohup_stack_r_3.out 2>&1&
nohup python3 stackingcv_random_search.py fourth > nohup_stack_r_4.out 2>&1&