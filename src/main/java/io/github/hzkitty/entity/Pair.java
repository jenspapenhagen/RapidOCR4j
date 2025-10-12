package io.github.hzkitty.entity;

import java.util.Objects;

/**
 * 简单的不可变二元组 (Pair)。
 *
 * @param <L> left 元素类型
 * @param <R> right 元素类型
 */
public record Pair<L, R>(L left, R right) {

    /**
     * 静态工厂方法。
     */
    public static <L, R> Pair<L, R> of(L left, R right) {
        return new Pair<>(left, right);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Pair<?, ?>(Object left1, Object right1))) {
            return false;
        }
        return Objects.equals(left, left1)
                && Objects.equals(right, right1);
    }

    @Override
    public String toString() {
        return "Pair{" + "left=" + left + ", right=" + right + '}';
    }
}
