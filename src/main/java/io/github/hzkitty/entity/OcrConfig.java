package io.github.hzkitty.entity;


// OCR 主配置类

public class OcrConfig {
    public GlobalConfig Global = new GlobalConfig(); // 全局配置
    public DetConfig Det = new DetConfig(); // 检测模块配置
    public ClsConfig Cls = new ClsConfig(); // 分类模块配置
    public RecConfig Rec = new RecConfig(); // 识别模块配置


    // 全局配置类
    public record GlobalConfig(
            float textScore,          // 文本评分阈值
            boolean useDet,           // 是否使用检测模块
            boolean useCls,           // 是否使用分类模块
            boolean useRec,           // 是否使用识别模块
            boolean printVerbose,     // 是否打印详细信息
            int minHeight,            // 最小高度
            float widthHeightRatio,   // 宽高比
            int maxSideLen,           // 最大边长
            int minSideLen,           // 最小边长
            boolean returnWordBox,    // 是否返回单词级别的框
            int intraOpNumThreads,    // 单线程操作线程数
            int interOpNumThreads,    // 多线程操作线程数
            String opencvLibPath      // opencv环境依赖dll或so目录
    ) {
        // You can define a compact constructor to apply default values
        public GlobalConfig() {
            this(
                    0.5f,    // textScore
                    true,    // useDet
                    true,    // useCls
                    true,    // useRec
                    false,   // printVerbose
                    30,      // minHeight
                    8.0f,    // widthHeightRatio
                    2000,    // maxSideLen
                    30,      // minSideLen
                    false,   // returnWordBox
                    -1,      // intraOpNumThreads
                    -1,      // interOpNumThreads
                    null     // opencvLibPath
            );
        }
    }


    // 检测模块配置类
    public record DetConfig(
            int intraOpNumThreads,   // 单线程操作线程数
            int interOpNumThreads,   // 多线程操作线程数
            boolean useCuda,         // 是否使用 CUDA
            int deviceId,            // 显卡编号
            boolean useDml,          // 是否使用 DML
            String modelPath,        // 模型路径
            int limitSideLen,        // 限制边长
            String limitType,        // 限制类型
            float thresh,            // 检测阈值
            float boxThresh,         // 边框阈值
            int maxCandidates,       // 最大候选框数
            float unclipRatio,       // 非极大值抑制后的扩展比例
            boolean useDilation,     // 是否使用膨胀操作
            String scoreMode,        // 评分模式
            boolean useArena         // arena 内存池扩展策略
    ) {
        // Compact constructor providing default values (like the original class)
        public DetConfig() {
            this(
                    -1,                         // intraOpNumThreads
                    -1,                         // interOpNumThreads
                    false,                      // useCuda
                    0,                          // deviceId
                    false,                      // useDml
                    "models/ch_PP-OCRv4_det_infer.onnx", // modelPath
                    736,                        // limitSideLen
                    "min",                      // limitType
                    0.3f,                       // thresh
                    0.5f,                       // boxThresh
                    1000,                       // maxCandidates
                    1.6f,                       // unclipRatio
                    true,                       // useDilation
                    "fast",                     // scoreMode
                    false                       // useArena
            );
        }
    }

    // 分类模块配置类
    public record ClsConfig(
            int intraOpNumThreads,   // 单线程操作线程数
            int interOpNumThreads,   // 多线程操作线程数
            boolean useCuda,         // 是否使用 CUDA
            int deviceId,            // 显卡编号
            boolean useDml,          // 是否使用 DML
            String modelPath,        // 模型路径
            int[] clsImageShape,     // 分类输入图像形状
            int clsBatchNum,         // 分类批量处理数
            float clsThresh,         // 分类阈值
            String[] labelList,      // 分类标签列表
            boolean useArena         // arena 内存池扩展策略
    ) {
        // Compact constructor providing default values (same as original class)
        public ClsConfig() {
            this(
                    -1,                                        // intraOpNumThreads
                    -1,                                        // interOpNumThreads
                    false,                                     // useCuda
                    0,                                         // deviceId
                    false,                                     // useDml
                    "models/ch_ppocr_mobile_v2.0_cls_infer.onnx", // modelPath
                    new int[]{3, 48, 192},                    // clsImageShape
                    1,                                         // clsBatchNum
                    0.9f,                                      // clsThresh
                    new String[]{"0", "180"},                 // labelList
                    false                                      // useArena
            );
        }
    }

    // 识别模块配置类
    public record RecConfig(
            int intraOpNumThreads,   // 单线程操作线程数
            int interOpNumThreads,   // 多线程操作线程数
            boolean useCuda,         // 是否使用 CUDA
            int deviceId,            // 显卡编号
            boolean useDml,          // 是否使用 DML
            String modelPath,        // 模型路径
            int[] recImgShape,       // 识别输入图像形状
            int recBatchNum,         // 识别批量处理数
            boolean useArena,        // arena 内存池扩展策略
            String recKeysPath       // 字典路径（可选）
    ) {
        // Compact constructor providing default values (matching your original class)
        public RecConfig() {
            this(
                    -1,                               // intraOpNumThreads
                    -1,                               // interOpNumThreads
                    false,                            // useCuda
                    0,                                // deviceId
                    false,                            // useDml
                    "models/ch_PP-OCRv4_rec_infer.onnx", // modelPath
                    new int[]{3, 48, 320},           // recImgShape
                    1,                                // recBatchNum
                    false,                            // useArena
                    null                              // recKeysPath
            );
        }
    }
}