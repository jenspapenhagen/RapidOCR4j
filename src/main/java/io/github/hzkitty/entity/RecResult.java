package io.github.hzkitty.entity;

import org.opencv.core.Point;

import java.util.Arrays;


public record RecResult(Point[] dtBoxes, String text, float confidence, WordBoxResult wordBoxResult) {

    @Override
    public String toString() {
        return "RecResult{" +
                "dtBoxes=" + Arrays.toString(dtBoxes) +
                ", text='" + text + '\'' +
                ", confidence=" + confidence +
                ", wordBoxResult=" + wordBoxResult +
                '}';
    }
}
