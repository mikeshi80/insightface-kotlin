package me.mikeshi.insightface.utils

import org.apache.mxnet.Context
import org.apache.mxnet.DataDesc
import org.apache.mxnet.DataIter
import org.apache.mxnet.Layout
import org.apache.mxnet.NDArray.zeros
import org.apache.mxnet.io.NDArrayIter
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms.round

operator fun INDArray.plus(other: INDArray): INDArray = add(other)
operator fun INDArray.plus(other: Number): INDArray = add(other)
operator fun INDArray.plusAssign(other: INDArray) {
    addi(other)
}

operator fun INDArray.plusAssign(other: Number) {
    addi(other)
}

operator fun INDArray.minus(other: INDArray): INDArray = sub(other)
operator fun INDArray.minus(other: Number): INDArray = sub(other)
operator fun INDArray.minusAssign(other: INDArray) {
    subi(other)
}

operator fun INDArray.minusAssign(other: Number) {
    subi(other)
}

operator fun INDArray.times(other: INDArray): INDArray = mul(other)
operator fun INDArray.times(other: Number): INDArray = mul(other)
operator fun INDArray.timesAssign(other: INDArray) {
    muli(other)
}

operator fun INDArray.timesAssign(other: Number) {
    muli(other)
}

operator fun INDArray.divAssign(other: INDArray) {
    divi(other)
}

operator fun INDArray.divAssign(other: Number) {
    divi(other)
}


fun INDArray.round(dup: Boolean = false): INDArray = round(this, dup)

fun maximum(left: INDArray, right: INDArray): INDArray {
    val stacked = Nd4j.stack(0, left, right)
    return stacked.max(0)
}

fun maximum(left: INDArray, right: Number): INDArray {
    val bRight = left.like().assign(right)
    val stacked = Nd4j.stack(0, left, bRight)
    return stacked.max(0)
}

fun minimum(left: INDArray, right: INDArray): INDArray {
    val stacked = Nd4j.stack(0, left, right)
    return stacked.min(0)
}

fun minimum(left: INDArray, right: Number): INDArray {
    val bRight = left.like().assign(right)
    val stacked = Nd4j.stack(0, left, bRight)
    return stacked.min(0)
}

fun INDArray.dataIter(confMinSize: Int, ctx: Context? = null, dtype: MXDType = MXDType.Float32): DataIter {
    val size = shape()[0].toInt()
    val label = zeros(ctx, listOf(size).toSeq())
    val data = toMXNet(ctx, dtype)
    return NDArrayIter.Builder().addDataWithDesc(DataDesc("data", data.shape(), data.dtype(), Layout.NCHW()), data)
            .addLabelWithDesc(DataDesc("softmax_label", label.shape(), label.dtype(), Layout.N()), label)
            .setBatchSize(kotlin.math.min(size, confMinSize)).build()
}

operator fun INDArray.set(vararg indices: INDArrayIndex, toSet: INDArray): INDArray = put(indices, toSet)
operator fun INDArray.set(vararg indices: INDArrayIndex, toSet: Number): INDArray = put(indices, toSet)
operator fun INDArray.get(idx: Long): INDArray = getRow(idx)
operator fun INDArray.set(idx: Long, toSet: INDArray): INDArray = getRow(idx).assign(toSet)
operator fun INDArray.set(idx: Long, toSet: Number): INDArray = getRow(idx).assign(toSet)