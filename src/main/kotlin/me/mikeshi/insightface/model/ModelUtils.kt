package me.mikeshi.insightface.model

import me.mikeshi.insightface.utils.*
import org.apache.mxnet.*
import org.apache.mxnet.module.Module
import org.apache.mxnet.optimizer.SGD
import org.nd4j.linalg.api.ndarray.INDArray
import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties(prefix = "insightface.model")
data class ModelConf(
        val root: String,
        val gpu: Int,
        val imageWidth: Int,
        val imageHeight: Int,
        val recognition: Recognition,
        val alignment: Alignment
) {
    data class Recognition(
            val name: String,
            val threshold: Float
    )

    data class Alignment(
            val name: String,
            val threshold: List<Float>,
            val minSize: Int,
            val factor: Float,
            val batchSize: Int
    )
}

fun getContexts(gpu: Int): Array<Context> {
    return if (gpu < 0) {
        arrayOf(Context.cpu(0))
    } else {
        arrayOf(Context.gpu(gpu))
    }
}

fun getSGD(learningRate: Float = 0.01f, momentum: Float = 0.0f,
           wd: Float = 0.0001f, clipGradient: Float = 0f,
           lrScheduler: LRScheduler? = null): SGD =
        SGD(learningRate, momentum, wd, clipGradient, lrScheduler)


fun loadFeedForward(prefix: String, epoch: Int,
                    ctx: Array<Context> = arrayOf(Context.cpu(0)),
                    numEpoch: Int = -1,
                    epochSize: Int = -1,
                    optimizer: Optimizer = getSGD(),
                    initializer: Initializer = Uniform(0.01f),
                    batchSize: Int = 128,
                    allowExtraParams: Boolean = false): FeedForward {

    return FeedForward.load(prefix, epoch, ctx, numEpoch, epochSize,
            optimizer, initializer, batchSize, allowExtraParams)
}

fun getModel(ctx:Array<Context>, prefix: String, epoch: Int, shape: List<Int>): Module {
    val (sym, argParams, auxParams) = Model.loadCheckpoint(prefix, epoch).let {
        Triple(it._1(), it._2(), it._3())
    }

    val model = Module(sym, listOf("data").toIndexedSeq(), emptyList<String>().toIndexedSeq(), ctx, none(), none())
    model.bind(
            listOf(DataDesc("data", Shape(shape.toSeq()), DType.Float32(), Layout.NCHW())).toIndexedSeq(),
            none(), true, false, false, none(), "write")

    model.setParams(argParams, auxParams, true, true, true)

    return model
}

fun getFeedForward(ctx: Array<Context>, prefix: String, epoch: Int): FeedForward {
    return FeedForward.load(prefix, epoch, ctx, -1, -1,
            SGD(0.01f, 0f, 0.0001f, 0f, null),
            Uniform(0.01f), 128, false)
}

fun predict(net: Module, inputBuf: INDArray): List<INDArray> {
    val db = DataBatch.Builder().setData(listOf(inputBuf.toMXNet()).toSeq()).build()
    net.forward(db, false)
    return net.outputsMerged.toJavaList().map { it.toND4j() }
}

fun predict(net: FeedForward, dataIter: DataIter): List<INDArray> {
    val r = net.predict(dataIter, -1).map { it.toND4j() }
    return r
}