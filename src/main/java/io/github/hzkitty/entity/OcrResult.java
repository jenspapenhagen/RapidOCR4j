package io.github.hzkitty.entity;

import java.util.List;

/**
 * OCR 识别结果
 *
 * @param elapseTime 总耗时
 * @param detTime    检测耗时
 * @param clsTime    分类耗时
 * @param recTime    识别耗时
 */
public record OcrResult(String strRes,
                        List<RecResult> recRes,
                        double elapseTime,
                        double detTime,
                        double clsTime,
                        double recTime) {

    @Override
    public String toString() {
        return "OcrResult{" +
                "strRes='" + strRes + '\'' +
                ", recRes=" + recRes +
                ", elapseTime=" + elapseTime +
                ", detTime=" + detTime +
                ", clsTime=" + clsTime +
                ", recTime=" + recTime +
                '}';
    }
}