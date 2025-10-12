package io.github.hzkitty.entity;

import org.opencv.core.Point;

import java.util.Arrays;
import java.util.List;


public record WordBoxResult(List<String> wordBoxContentList, List<Point[]> sortedWordBoxList, List<Float> confList) {

    public WordBoxResult withSortedWordBoxList(List<Point[]> sortedWordBoxList) {
        return new WordBoxResult(wordBoxContentList(), sortedWordBoxList, confList());
    }

    @Override
    public String toString() {
        return "WordBoxResult{" +
                "wordBoxContentList=" + wordBoxContentList +
                ", sortedWordBoxList=" + Arrays.deepToString(sortedWordBoxList.toArray()) +
                ", confList=" + confList +
                '}';
    }
}