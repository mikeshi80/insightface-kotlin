package me.mikeshi.insightface.model

import me.mikeshi.insightface.utils.*
import org.apache.mxnet.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.springframework.stereotype.Service
import javax.annotation.PostConstruct
import org.apache.mxnet.module.Module as MxModule


@Service
class FaceModel(val conf: ModelConf) {

    lateinit var faceModel: MxModule
    lateinit var detector: MtcnnDetector
    lateinit var ctx: Array<Context>

    @PostConstruct
    fun postConstruct() {
        ResourceScope().use {
            ctx = getContexts(conf.gpu)
            faceModel = getModel(ctx, "${conf.root}/${conf.recognition.name}", 0,
                    listOf(1, 3, conf.imageHeight, conf.imageWidth))
            detector = MtcnnDetector(conf, ctx)
        }
    }



    fun getInput(faceImg: INDArray?): INDArray? {
        val ret = detector.detectFace(faceImg)
        ret ?: return null

        faceImg!!

        var (bbox, points) = ret

        if (bbox.shape()[0] == 0L) return null

        bbox = bbox[NDArrayIndex.point(0), NDArrayIndex.interval(0, 4)]
        points = points[0].reshape(2, 5).transpose()

        var nimg: INDArray = facePreprocess(faceImg, bbox, points, conf.imageWidth, conf.imageHeight)
        nimg = convertColor(nimg)
        nimg.permutei(2, 0, 1)
        return nimg
    }

    fun getFeature(aligned: INDArray): INDArray {
        val inputBlob = Nd4j.expandDims(aligned, 0)
        val data = inputBlob.toMXNet(ctx = ctx[0])
        val db = DataBatch.Builder().setData(listOf(data).toSeq()).build()
        faceModel.forward(db, false)
        val embedding = faceModel.outputs.apply(0).toJavaIterable().map { it.toND4j() }.toTypedArray()

        val vsEmbedding = Nd4j.vstack(*embedding)
        return vsEmbedding.norm2(1)
    }


}