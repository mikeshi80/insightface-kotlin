package me.mikeshi.insightface.utils

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.conditions.GreaterThan


internal class UtilsTest {


    @Test
    fun testMaximum() {
        val left = Nd4j.create(floatArrayOf(1f, 7f, 3f))
        val right = Nd4j.create(floatArrayOf(2f, 4f, 5f))
        val max = maximum(left, right)

        assertEquals(Nd4j.create(floatArrayOf(2f, 7f, 5f)), max)

        val x = 5 / 2
        assertEquals(2, x)
    }

    @Test
    fun testOps() {
        val mtx = Nd4j.create(FloatArray(12) { (it + 1.5).toFloat() }, intArrayOf(3, 4))

        mtx.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3)).round(false)


        assertEquals(7.0f, mtx.getFloat(1, 1))

    }

    @Test
    fun testMatch() {
        val mtx = Nd4j.create(FloatArray(6) { it.toFloat() })
        val match = mtx.match(3, GreaterThan(3))
        val result = Nd4j.where(match, null, null)
        assertEquals(1, result.size)

        assertEquals(4, result[0].getInt(0))
        assertEquals(5, result[0].getInt(1))
    }

    @Test
    fun testAssign() {
        val mtx = Nd4j.zeros(3, 4)
        mtx.getRow(0).assign(1)

        assertEquals(Nd4j.ones(4), mtx.getRow(0))

        mtx.put(arrayOf(NDArrayIndex.interval(1, 3), NDArrayIndex.all()), Nd4j.ones(2, 4))
        assertEquals(Nd4j.ones(3, 4), mtx)

        mtx[NDArrayIndex.indices(0, 2)] = Nd4j.zeros(2, 4)

        assertEquals(Nd4j.zeros(4), mtx.getRow(0))
        assertEquals(Nd4j.ones(4), mtx.getRow(1))
        assertEquals(Nd4j.zeros(4), mtx.getRow(2))

        assertEquals(Nd4j.zeros(2, 4), mtx.getRows(0, 2))

        mtx.getRow(1).assign(0)
        assertEquals(Nd4j.zeros(3, 4), mtx)

        val idx = NDArrayIndex.indices(1, 2)
        val indArray = mtx[idx]
        mtx[idx] = indArray.assign(Nd4j.ones(2, 4))

        assertEquals(Nd4j.ones(2, 4), mtx[idx])

        assertEquals(Nd4j.zeros(4), mtx[0])

        mtx[0] = Nd4j.ones(4)
        assertEquals(Nd4j.ones(3, 4), mtx)

        assertEquals(mtx[0], mtx.get(NDArrayIndex.point(0)))
        assertEquals(mtx.get(NDArrayIndex.indices(0)),
                Nd4j.expandDims(mtx[0], 1))

        assertEquals(mtx.getRows(0, 2), mtx.get(NDArrayIndex.indices(0, 2)))


        val mtx2 = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val gt1 = Nd4j.where(mtx2.gt(3f), null, null)[0].toIntVector()
        val gt2 = Nd4j.where(mtx2.match(3f, GreaterThan(3f)), null, null)[0].toIntVector()
        assertArrayEquals(gt1, gt2)
        assertArrayEquals((3..5).toIntArray(), gt1)
    }

    @Test
    fun testExpandDims() {
        val mtx = Nd4j.create(doubleArrayOf(0.0, 1.0, 2.0))
        val a = Nd4j.expandDims(mtx, 1)

        assertArrayEquals(longArrayOf(3), mtx.shape())
        assertArrayEquals(longArrayOf(3, 1), a.shape())
    }

    @Test
    fun testPermute() {
        val mtx = Nd4j.create(DoubleArray(12) { (it + 1).toDouble() }, intArrayOf(3, 4))
        val permuted = mtx.permute(1, 0)

        assertEquals(mtx.getRow(0), permuted.getColumn(0))

    }

}