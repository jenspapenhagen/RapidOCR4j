package io.github.hzkitty.entity;

import java.util.List;

public record WordBoxInfo(Double textIndexLen, List<List<String>> wordList, List<List<Integer>> wordColList,
                          List<String> stateList, List<Float> confList) {

    public WordBoxInfo withTextIndexLen(Double textIndexLen) {
        return new WordBoxInfo(textIndexLen, wordList(), wordColList(), stateList(), confList());
    }

    @Override
    public String toString() {
        return "WordBoxInfo{" +
                "textIndexLen=" + textIndexLen +
                ", wordList=" + wordList +
                ", wordColList=" + wordColList +
                ", stateList=" + stateList +
                ", confList=" + confList +
                '}';
    }
}