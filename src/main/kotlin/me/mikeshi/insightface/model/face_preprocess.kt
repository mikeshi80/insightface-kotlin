package me.mikeshi.insightface.model

import me.mikeshi.insightface.utils.*
import org.bytedeco.opencv.global.opencv_calib3d
import org.bytedeco.opencv.global.opencv_imgcodecs.imread
import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Size
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

fun facePreprocess(img: INDArray, bbox: INDArray?, landmark: INDArray?, width: Int, height: Int): INDArray {

    if (landmark != null) {
        val src = Nd4j.create(intArrayOf(5, 2),
                doubleArrayOf(
                        30.2946, 51.6963,
                        65.5318, 51.5014,
                        48.0252, 71.7366,
                        33.5493, 92.3655,
                        62.7299, 92.2041)
        )

        if (width == 112) {
            src.getColumn(0) += 8.0
        }

        val dst = landmark

        val loader = NativeImageLoader()

        val mat = opencv_calib3d.estimateAffinePartial2D(dst.reshape(1, 5, 2).asMat(loader), src.reshape(1, 5, 2).asMat(loader))

        val warped = Mat()

        opencv_imgproc.warpAffine(img.asMat(loader), warped, mat, Size(width, height))

        return warped.asNd4j(loader)
    } else {
        val det = if (bbox == null) {
            val det = Nd4j.zeros(4)
            det[0] = (img.shape()[1] * 0.0625).toInt()
            det[1] = (img.shape()[0] * 0.0625).toInt()
            det[2] = img.shape()[1] - det.getLong(0)
            det[3] = img.shape()[0] - det.getLong(1)
            det
        } else {
            bbox
        }

        val margin = 44

        val bb = Nd4j.zeros(4)
        bb[0] = maximum(det[0] - margin / 2, 0)
        bb[1] = maximum(det[1] - margin / 2, 0)
        bb[2] = minimum(det[2] + margin / 2, img.shape()[1])
        bb[3] = minimum(det[3] + margin / 2, img.shape()[0])

        return img.get(NDArrayIndex.interval(bb.getInt(1), bb.getInt(3)),
                NDArrayIndex.interval(bb.getInt(0), bb.getInt(2)), NDArrayIndex.all())
    }
}


fun readImage(imgPath: String, loader: NativeImageLoader = NativeImageLoader()): INDArray = imread(imgPath).asNd4j(loader)