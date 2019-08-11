package me.mikeshi.insightface.utils

import org.apache.mxnet.Context
import org.apache.mxnet.DType
import org.apache.mxnet.NDArray
import org.apache.mxnet.Shape
import org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB
import org.bytedeco.opencv.opencv_core.Mat
import org.datavec.image.data.ImageWritable
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.transform.ColorConversionTransform
import org.datavec.image.transform.ImageTransform
import org.datavec.image.transform.ResizeImageTransform
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex
import scala.Option
import scala.collection.IndexedSeq
import scala.collection.JavaConverters
import scala.collection.Seq

fun <T> Iterable<T>.toIndexedSeq(): IndexedSeq<T> {
    return JavaConverters.asScalaIteratorConverter(iterator()).asScala().toIndexedSeq()!!
}

fun <T> Iterable<T>.toSeq(): Seq<T> {
    return JavaConverters.asScalaIteratorConverter(iterator()).asScala().toSeq()
}

fun <T> Seq<T>.toJavaList(): List<T> {
    return JavaConverters.seqAsJavaListConverter(this).asJava()
}

fun <T> IndexedSeq<T>.toJavaIterable(): Iterable<T> = JavaConverters.asJavaIterableConverter(this).asJava()

fun <T> none(): Option<T> = Option.apply<T>(null)

enum class MXDType(val value: Int, val dtype: String) {
    Float32(0, "float32"),
    Float64(1, "float64"),
    Float16 (2, "float16"),
    UInt8(3, "uint8"),
    Int32(4, "int32"),
    Unknown(-1, "unknown")
}

fun INDArray.toMXNet(ctx: Context? = null, dtype: MXDType = MXDType.Float32): NDArray {
    val type = DType.apply(dtype.value)
    val shape = shape().map { it.toInt() }
    val array = Nd4j.toFlattened(this).toDoubleVector()

    return NDArray.array(array, Shape(shape.asIterable().toSeq()), ctx).asType(type)
}

fun NDArray.toND4j(): INDArray {
    val shape = shape()
    val array = toArray()

    return Nd4j.create(array, shape.toArray())
}

fun INDArray.toINDArrayIndex(): INDArrayIndex = NDArrayIndex.indices(*toLongVector())

fun INDArray.asMat(loader: NativeImageLoader = NativeImageLoader()): Mat = loader.asMat(this)

fun Mat.asNd4j(loader: NativeImageLoader = NativeImageLoader()): INDArray = loader.asMatrix(this).get(NDArrayIndex.point(0)).permute(1, 2, 0)

fun IntRange.toIntArray(): IntArray {
    if (last < first)
        return IntArray(0)

    val result = IntArray(last - first + 1)
    var index = 0
    for (element in this)
        result[index++] = element
    return result
}

private fun doTransform(image: INDArray, transform: ImageTransform, loader: NativeImageLoader): INDArray {
    val img = ImageWritable(loader.asFrame(image))

    return loader.asMatrix(transform.transform(img))
}

fun resizeImage(image: INDArray, width: Int, height: Int, loader: NativeImageLoader = NativeImageLoader()): INDArray {
    val img = image.permute(2, 0 ,1)
    val transform = ResizeImageTransform(width, height)
    return doTransform(img, transform, loader)[NDArrayIndex.point(0)].permute(1, 2, 0)
}

fun convertColor(image: INDArray, conversionCode: Int = COLOR_BGR2RGB, loader: NativeImageLoader = NativeImageLoader()): INDArray {
    val img = image.permute(2, 0, 1)
    val transform = ColorConversionTransform(conversionCode)
    return doTransform(img, transform, loader)[NDArrayIndex.point(0)].permute(1, 2, 0)
}