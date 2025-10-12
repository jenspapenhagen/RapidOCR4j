package io.github.hzkitty.entity;

public record OrtInferConfig(
    int intraOpNumThreads, // 单线程操作线程数
     int interOpNumThreads, // 多线程操作线程数
     boolean useCuda, // 是否使用 CUDA
     int deviceId, // 显卡编号
     boolean useDml, // 是否使用 DML
     String modelPath, // 模型路径
     boolean useArena
){}
