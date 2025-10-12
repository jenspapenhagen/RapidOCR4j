package io.github.hzkitty.entity;


public record ParamConfig(
        Float boxThresh, // 边框阈值
        Float unclipRatio, // 非极大值抑制后的扩展比例
        Float textScore, // 文本评分阈值
        Boolean returnWordBox,// 是否返回单词级别的框
        Boolean useDet, // 是否使用检测模块
        Boolean useCls, // 是否使用分类模块
        Boolean useRec // 是否使用识别模块
) {
    //default parameter
    public ParamConfig() {
        this(
                0.5f,
                1.6f,
                0.5f,
                true,
                true,
                true,
                true
                );
    }

}
