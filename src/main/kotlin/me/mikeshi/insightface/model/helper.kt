package me.mikeshi.insightface.model

import me.mikeshi.insightface.utils.*
import org.apache.mxnet.Context
import org.apache.mxnet.FeedForward
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.util.ArrayUtil
import kotlin.math.ceil

/**
adjust the input from (h, w, c) to ( 1, c, h, w) for network input

Parameters:
----------
in_data: ND4j array of shape (h, w, c)
input data
Returns:
-------
out_data: ND4j array of shape (1, c, h, w)
reshaped array
 */

fun adjustInput(inData: INDArray): INDArray {

    var outData = inData.permute(2, 0, 1)
    outData = Nd4j.expandDims(outData, 0)
    outData = (outData - 127.5f) * 0.0078125
    return outData
}

/**
non max suppression

Parameters:
----------
box: numpy array n x 5
input bbox array
overlap_threshold: float number
threshold of overlap
mode: float number
how to compute overlap ratio, 'Union' or 'Min'
Returns:
-------
index array of the selected bbox
 */
fun nms(boxes: INDArray, overlapThreshold: Float, mode: String): IntArray {
    // if there are no boxes, return an empty list
    if (boxes.isEmpty) return intArrayOf()

    // initialize the list of picked indexes
    val pick = mutableListOf<Int>()

    // grab the coordinates of the bounding boxes
    val (x1, y1, x2, y2, score) = (0L until 5L).map { boxes.getColumn(it) }

    val area = (x2 - x1 + 1) * (y2 - y1 + 1)

    val idxs = ArrayUtil.argsort(score.toIntVector()).toMutableList()

    // keep looping while some indexes still remain in the indexes list
    while (idxs.isNotEmpty()) {
        // grab the last index in the indexes list and add the index value to the list of picked indexes
        val last = idxs.size - 1
        val i = idxs[last]
        pick.add(i)

        val idx = i.toLong()
        val rowsIdxs = idxs.subList(0, last).toIntArray()
        val xx1 = maximum(x1[idx], x1.getRows(*rowsIdxs))
        val yy1 = maximum(y1[idx], y1.getRows(*rowsIdxs))
        val xx2 = maximum(x2[idx], x2.getRows(*rowsIdxs))
        val yy2 = maximum(y2[idx], y2.getRows(*rowsIdxs))

        // compute the width and height of the bounding box
        val w = maximum(Nd4j.zerosLike(xx1), xx2 - xx1 + 1)
        val h = maximum(Nd4j.zerosLike(yy1), yy2 - yy1 + 1)

        val inter = w * h

        val overlap = if (mode == "Min") {
            inter / minimum(area.getRow(idx), area.getRows(*rowsIdxs))
        } else {
            inter / (area.getRow(idx) + area.getRows(*rowsIdxs) - inter)
        }

        val toDelete = Nd4j.where(overlap.gt(overlapThreshold), null, null)[0].toIntVector().
                toMutableList().apply { add(last) }

        toDelete.sortDescending()

        for (d in toDelete) {
            idxs.removeAt(d)
        }

    }

    return pick.toIntArray()
}

/**
run PNet for first stage

Parameters:
----------
img: nd4j array, bgr order
input image
scale: float number
how much should the input image scale
net: PNet
worker
Returns:
-------
total_boxes : bboxes
 */
fun detectFirstStage(img: INDArray, net: FeedForward, scale: Float, threshold: Float, batchSize: Int, ctx: Context): INDArray? {

    val (height, width) = Pair(img.shape()[0], img.shape()[1])
    val hs = ceil(height * scale).toInt()
    val ws = ceil(width * scale).toInt()

    val imData = resizeImage(img, ws, hs)

    // adjust for the network input
    val inputBuf = adjustInput(imData)
    val dataIter = inputBuf.dataIter(batchSize, ctx)
//    val output = net.predict(dataIter, -1).map { it.toND4j() }.toTypedArray()
    val output = predict(net, dataIter)

    val idx = arrayOf(
            NDArrayIndex.point(0),
            NDArrayIndex.point(1),
            NDArrayIndex.all(),
            NDArrayIndex.all())
    val boxes = generateBBox(output[1].get(*idx), output[0], scale, threshold)


    if (boxes.isEmpty) return null

    val pick = nms(boxes[NDArrayIndex.all(), NDArrayIndex.interval(0, 5)], 0.5f, "Union")
    return boxes.getRows(*pick)
}

/**
generate bbox from feature map

Parameters:
----------
map: nd4j array , n x m x 1
detect score for each position
reg: nd4j array , n x m x 4
bbox
scale: float number
scale of this detection
threshold: float number
detect threshold
Returns:
-------
bbox array
 */
fun generateBBox(map: INDArray, reg: INDArray, scale: Float, threshold: Float): INDArray {
    val stride = 2
    val cellSize = 12

    val tIndex = Nd4j.where(map.gt(threshold), null, null)

    // find nothing
    if (tIndex[0].isEmpty) return Nd4j.create(floatArrayOf())

    val (dx1, dy1, dx2, dy2) = (0L until 4L).map {
        reg[NDArrayIndex.point(0), NDArrayIndex.point(it),
                tIndex[0].toINDArrayIndex(), tIndex[1].toINDArrayIndex()]
    }

    val reg1 = reg.like()

    reg1[0] = dx1
    reg1[1] = dy1
    reg1[2] = dx2
    reg1[3] = dy2

    val score = map[tIndex[0].toINDArrayIndex(), tIndex[1].toINDArrayIndex()]

    val boundingBox = Nd4j.vstack(
            ((tIndex[1] * stride + 1) / scale).round(false),
            ((tIndex[0] * stride + 1) / scale).round(false),
            ((tIndex[1] * stride + 1 + cellSize) / scale).round(false),
            ((tIndex[0] * stride + 1 + cellSize) / scale).round(false),
            score, reg1)

    return boundingBox.transpose()
}
