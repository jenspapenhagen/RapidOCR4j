package io.github.hzkitty.entity;


/**
 * 用于存储文本识别结果 (text, conf)
 */

public record TupleResult(String text, float confidence, WordBoxInfo wordBoxInfo, WordBoxResult wordBoxResult) {


    @Override
    public String toString() {
        return "TupleResult{" +
                ", text='" + text + '\'' +
                ", confidence=" + confidence +
                ", wordBoxInfo=" + wordBoxInfo +
                ", wordBoxResult=" + wordBoxResult +
                '}';
    }
}
