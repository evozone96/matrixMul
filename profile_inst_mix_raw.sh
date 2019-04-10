!# /usr/bin/env bash

nv-nsight-cu-cli -f --metrics sm__inst_executed_pipe_adu.avg,sm__inst_executed_pipe_alu.avg,sm__inst_executed_pipe_cbu.avg,sm__inst_executed_pipe_fma.avg,sm__inst_executed_pipe_fp16.avg,sm__inst_executed_pipe_fp64.avg,sm__inst_executed_pipe_ipa.avg,sm__inst_executed_pipe_lsu.avg,sm__inst_executed_pipe_tensor.avg,sm__inst_executed_pipe_tex.avg,sm__inst_executed_pipe_uniform.avg,sm__inst_executed_pipe_xu.avg,gpu__cycles_elapsed.avg  $1
