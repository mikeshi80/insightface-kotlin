package me.mikeshi.insightface.model

import me.mikeshi.insightface.utils.*
import org.apache.mxnet.Context
import org.apache.mxnet.FeedForward
import org.apache.mxnet.module.Module
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import kotlin.math.min
import kotlin.math.pow

class MtcnnDetector(private val conf: ModelConf, private val ctx: Array<Context>) {
    private val PNet: FeedForward
    private val RNet: FeedForward
    private val ONet: FeedForward
    private val LNet: FeedForward

    init {
        val models = arrayOf("det1", "det2", "det3", "det4").map { "${conf.root}/$it" }
//        PNet = getModel(ctx, models[0], 1, listOf(1, 3, 12, 12))
        PNet = getFeedForward(ctx, models[0], 1)
//        RNet = getModel(ctx, models[1], 1, listOf(57, 3, 24, 24))
        RNet = getFeedForward(ctx, models[1], 1)
//        ONet = getModel(ctx, models[2], 1, listOf(4, 3, 48, 48))
        ONet = getFeedForward(ctx, models[2], 1)
//        LNet = getModel(ctx, models[3], 1, listOf(1, 15, 24, 24))
        LNet = getFeedForward(ctx, models[3], 1)
    }

    /**
    detect face over img
    Parameters:
    ----------
    img: nd4j array, bgr order of shape (1, 3, n, m)
    input image
    Returns:
    -------
    bboxes: nd4j array, n x 5 (x1,y2,x2,y2,score)
    bboxes
    points: nd4j array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
    landmarks
     */
    fun detectFace(img: INDArray?): Pair<INDArray, INDArray>? {
        img ?: return null
        // only works for color image
        if (img.shape().size != 3) {
            return null
        }

        // detect boxes
        val height = img.shape()[0]
        val width = img.shape()[1]

        val MIN_DET_SIZE = 12f


        var minl = min(height, width).toFloat()

        // get all the valid scales
        val scales = mutableListOf<Float>()
        val m = MIN_DET_SIZE / conf.alignment.minSize

        minl *= m
        var factorCount = 0

        while (minl > MIN_DET_SIZE) {
            scales.add(m * conf.alignment.factor.pow(factorCount))
            minl *= conf.alignment.factor
            factorCount += 1
        }

        val threshold = conf.alignment.threshold

        /**
         * first stage
         */

        val totalBoxeList = mutableListOf<INDArray?>()

        for (idx in 0 until scales.size) {
            totalBoxeList.add(detectFirstStage(img, PNet, scales[idx], threshold[0], conf.alignment.batchSize, ctx[0]))
        }

        // remove the nulls
        val totalBoxesListNotNull = totalBoxeList.filterNotNull()

        if (totalBoxesListNotNull.isEmpty()) return null

        var totalBoxes = Nd4j.vstack(totalBoxesListNotNull)

        if (totalBoxes.isEmpty) return null

        // merge the detection from first stage
        var pick = nms(totalBoxes.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 5)), 0.7f, "Union")
        totalBoxes = totalBoxes.getRows(*pick)

        var bbw = totalBoxes.getColumn(2) - totalBoxes.getColumn(0) + 1
        var bbh = totalBoxes.getColumn(3) - totalBoxes.getColumn(1) + 1


        // refine the bboxes
        totalBoxes = Nd4j.vstack(
                totalBoxes.getColumn(0) + totalBoxes.getColumn(5) * bbw,
                totalBoxes.getColumn(1) + totalBoxes.getColumn(6) * bbh,

                totalBoxes.getColumn(2) + totalBoxes.getColumn(7) * bbw,
                totalBoxes.getColumn(3) + totalBoxes.getColumn(8) * bbh,
                totalBoxes.getColumn(4)
        )

        totalBoxes = totalBoxes.transpose()
        totalBoxes = convertToSquare(totalBoxes)


        totalBoxes.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 4)).round(false)


        /**
         * second stage
         */

        var numBox = totalBoxes.shape()[0].toInt()

        // pad the bbox
        val (dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph) = pad(totalBoxes, width, height)

        val inputBuf = Nd4j.zeros(numBox, 3, 24, 24) // need dtype float32

        val loader = NativeImageLoader()

        for (i in 0 until numBox) {
            val tmp = Nd4j.zeros(tmph[i], tmpw[i], 3)
            tmp[NDArrayIndex.interval(dy[i], edy[i] + 1),
                    NDArrayIndex.interval(dx[i], edx[i] + 1),
                    NDArrayIndex.all()] = img[NDArrayIndex.interval(y[i], ey[i] + 1),
                    NDArrayIndex.interval(x[i], ex[i] + 1),
                    NDArrayIndex.all()]

            inputBuf[NDArrayIndex.point(i.toLong()), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()] =
                    adjustInput(resizeImage(img, 24, 24, loader))
        }

        var dataIter = inputBuf.dataIter(conf.alignment.batchSize, ctx[0])
//        var output = RNet.predict(dataIter, -1).map { it.toND4j() }.toTypedArray()
        var output = predict(RNet, dataIter)

        // filter the totalBoxes  with threshold
        var passed = Nd4j.where(output[1].getColumn(1).gt(threshold[1]), null, null)[0].toIntVector()


        totalBoxes = totalBoxes.getRows(*passed)

        if (totalBoxes.isEmpty) return null

        totalBoxes.putColumn(4, output[1].getRows(*passed).getColumn(1).reshape(-1))

        var reg = output[0].getRows(*passed)

        // nms
        pick = nms(totalBoxes, 0.7f, "Union")
        totalBoxes = totalBoxes.getRows(*pick)
        totalBoxes = calibrateBox(totalBoxes, reg.getRows(*pick))
        totalBoxes = convertToSquare(totalBoxes)
        totalBoxes.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 4)).round(false)

        /**
         * third stage
         */
        numBox = totalBoxes.shape()[0].toInt()

        val (dy1, edy1, dx1, edx1, y1, ey1, x1, ex1, tmpw1, tmph1) = pad(totalBoxes, width, height)


        for (i in 0 until numBox) {
            val tmp = Nd4j.zeros(tmph1[i], tmpw1[i], 3)
            tmp.put(arrayOf(
                    NDArrayIndex.interval(dy1[i], edy1[i] + 1),
                    NDArrayIndex.interval(dx1[i], edx1[i] + 1),
                    NDArrayIndex.all()),
                    img.get(NDArrayIndex.interval(y1[i], ey1[i] + 1),
                            NDArrayIndex.interval(x1[i], ex1[i] + 1),
                            NDArrayIndex.all()))

            inputBuf.put(arrayOf(NDArrayIndex.point(i.toLong()), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()),
                    adjustInput(resizeImage(img, 48, 48, loader)))
        }

        dataIter = inputBuf.dataIter(conf.alignment.batchSize, ctx[0])
//        output = ONet.predict(dataIter, -1).map { it.toND4j() }.toTypedArray()
        output = predict(ONet, dataIter)

        // filter the totalBoxes  with threshold
        passed = Nd4j.where(output[2].getColumn(1).gt(threshold[2]), null, null)[0].toIntVector()

        totalBoxes = totalBoxes.getRows(*passed)

        if (totalBoxes.isEmpty) return null

        totalBoxes.putColumn(4, output[2].getRows(*passed).getColumn(1).reshape(-1))

        reg = output[1].getRows(*passed)

        var points = output[0].getRows(*passed)

        // compute landmark points

        bbw = totalBoxes.getColumn(2) - totalBoxes.getColumn(0) + 1
        bbh = totalBoxes.getColumn(3) - totalBoxes.getColumn(1) + 1
        points[NDArrayIndex.all(), NDArrayIndex.interval(0, 5)] =
                Nd4j.expandDims(totalBoxes.getColumn(0), 1) + Nd4j.expandDims(bbw, 1) *
                        points.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 5))
        points[NDArrayIndex.all(), NDArrayIndex.interval(5, 10)] =
                Nd4j.expandDims(totalBoxes.getColumn(1), 1) + Nd4j.expandDims(bbh, 1) *
                        points.get(NDArrayIndex.all(), NDArrayIndex.interval(5, 10))

        // nms
        totalBoxes = calibrateBox(totalBoxes, reg)
        pick = nms(totalBoxes, 0.7f, "Min")
        totalBoxes = totalBoxes.getRows(*pick)
        points = points.getRows(*pick)


        return Pair(totalBoxes, points)

        // not support accurate landmark yet
    }

    /**
    calibrate bboxes

    Parameters:
    ----------
    bbox: nd4j array, shape n x 5
    input bboxes
    reg:  nd4j array, shape n x 4
    bboxex adjustment

    Returns:
    -------
    bboxes after refinement
     */
    private fun calibrateBox(bbox: INDArray, reg: INDArray): INDArray {
        val w = bbox.getColumn(2) - bbox.getColumn(0) + 1
        Nd4j.expandDims(w, 1)
        var h = bbox.getColumn(3) - bbox.getColumn(1) + 1
        h = Nd4j.expandDims(h, 1)
        val regM = Nd4j.hstack(w, h, w, h)
        val aug = regM * reg
        bbox[NDArrayIndex.all(), NDArrayIndex.interval(0, 4)] = bbox[NDArrayIndex.all(), NDArrayIndex.interval(0, 4)] + aug

        return bbox
    }

    data class PadResult(
            val dy: IntArray,
            val edy: IntArray,
            val dx: IntArray,
            val edx: IntArray,
            val y: IntArray,
            val ey: IntArray,
            val x: IntArray,
            val ex: IntArray,
            val tmpW: IntArray,
            val tmpH: IntArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as PadResult

            if (!dy.contentEquals(other.dy)) return false
            if (!edy.contentEquals(other.edy)) return false
            if (!dx.contentEquals(other.dx)) return false
            if (!edx.contentEquals(other.edx)) return false
            if (!y.contentEquals(other.y)) return false
            if (!ey.contentEquals(other.ey)) return false
            if (!x.contentEquals(other.x)) return false
            if (!ex.contentEquals(other.ex)) return false
            if (!tmpW.contentEquals(other.tmpW)) return false
            if (!tmpH.contentEquals(other.tmpH)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = dy.contentHashCode()
            result = 31 * result + edy.contentHashCode()
            result = 31 * result + dx.contentHashCode()
            result = 31 * result + edx.contentHashCode()
            result = 31 * result + y.contentHashCode()
            result = 31 * result + ey.contentHashCode()
            result = 31 * result + x.contentHashCode()
            result = 31 * result + ex.contentHashCode()
            result = 31 * result + tmpW.contentHashCode()
            result = 31 * result + tmpH.contentHashCode()
            return result
        }
    }

    /**
    pad the the bboxes, alse restrict the size of it

    Parameters:
    ----------
    bboxes: nd4j array, n x 5
    input bboxes
    w: float number
    width of the input image
    h: float number
    height of the input image
    Returns :
    -------
    dy, dx : nd4j array, n x 1
    start point of the bbox in target image
    edy, edx : nd4j array, n x 1
    end point of the bbox in target image
    y, x : nd4j array, n x 1
    start point of the bbox in original image
    ex, ex : nd4j array, n x 1
    end point of the bbox in original image
    tmph, tmpw: numpy array, n x 1
    height and width of the bbox
     */
    fun pad(bboxes: INDArray, w: Long, h: Long): PadResult {

        val (tmpw, tmph) = Pair(
                bboxes.getColumn(2) - bboxes.getColumn(0) + 1,
                bboxes.getColumn(3) - bboxes.getColumn(1) + 1
        )

        val numBox = bboxes.rows()

        val (dx, dy, edx, edy) = listOf(
                Nd4j.zeros(numBox), Nd4j.zeros(numBox),
                tmpw.dup() - 1, tmph.dup() - 1
        )

        val (x, y, ex, ey) = listOf(
                bboxes.getColumn(0), bboxes.getColumn(1),
                bboxes.getColumn(2), bboxes.getColumn(3)
        )

        var tmpIndex = Nd4j.where(ex.gt(w - 1), null, null)[0].toINDArrayIndex()
        edx[tmpIndex] = tmpw[tmpIndex] + w - 2 - ex[tmpIndex]
        ex[tmpIndex] = w - 1

        tmpIndex = Nd4j.where(ey.gt(h - 1), null, null)[0].toINDArrayIndex()
        edy[tmpIndex] = tmph[tmpIndex] + h - 2 - ey[tmpIndex]
        ey[tmpIndex] = h - 1

        tmpIndex = Nd4j.where(x.lt(0), null, null)[0].toINDArrayIndex()
        dx[tmpIndex] = dx[tmpIndex].negi()
        x[tmpIndex] = 0

        tmpIndex = Nd4j.where(y.lt(0), null, null)[0].toINDArrayIndex()
        dy[tmpIndex] = dy[tmpIndex].negi()
        y[tmpIndex] = 0

        return PadResult(dy.toIntVector(), edy.toIntVector(), dx.toIntVector(), edx.toIntVector(), y.toIntVector(),
                ey.toIntVector(), x.toIntVector(), ex.toIntVector(), tmpw.toIntVector(), tmph.toIntVector())
    }



    fun convertToSquare(bbox: INDArray): INDArray {
        val ibbox = bbox.dup()
        val squareBBox = bbox.dup()
        val h = ibbox.getColumn(3) - ibbox.getColumn(1) + 1
        val w = ibbox.getColumn(2) - ibbox.getColumn(0) + 1

        val maxSide = maximum(h, w)


        squareBBox.getColumn(0).addi(w * 0.5 - maxSide * 0.5)
        squareBBox.getColumn(1).addi(w * 0.5 - maxSide * 0.5)
        squareBBox.getColumn(2).assign(squareBBox.getColumn(0) + maxSide - 1)
        squareBBox.getColumn(3).assign(squareBBox.getColumn(1) + maxSide - 1)

        return squareBBox
    }

}