package io.github.hzkitty.entity;

import java.util.Objects;

/**
 * 简单的不可变三元组 (Triple)。
 *
 * @param <L> left 元素类型
 * @param <M> middle 元素类型
 * @param <R> right 元素类型
 */
public record Triple<L, M, R>(L left, M middle, R right) {

    /**
     * 静态工厂方法。
     */
    public static <L, M, R> Triple<L, M, R> of(L left, M middle, R right) {
        return new Triple<>(left, middle, right);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Triple<?, ?, ?>(Object left1, Object middle1, Object right1))) {
            return false;
        }
        return Objects.equals(left, left1)
                && Objects.equals(middle, middle1)
                && Objects.equals(right, right1);
    }

    @Override
    public String toString() {
        return "Triple{" +
                "left=" + left +
                ", middle=" + middle +
                ", right=" + right +
                '}';
    }
}
