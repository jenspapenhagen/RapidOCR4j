package io.github.hzkitty.ch_ppocr_rec;

import io.github.hzkitty.entity.WordBoxInfo; /**
 * 用于存储 decode 后的结果
 */
public record DecodeResult(String text, float confidence, WordBoxInfo wordBoxInfo) {
//    private String text;        // 解码出的文本
//    private float confidence;   // 平均置信度
//
//    // 存储文本坐标/word box 等信息
//    private WordBoxInfo wordBoxInfo;

}
