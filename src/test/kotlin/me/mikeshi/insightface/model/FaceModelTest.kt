package me.mikeshi.insightface.model

import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.extension.ExtendWith
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.junit.jupiter.SpringExtension
import java.nio.file.Paths

@SpringBootTest
@ExtendWith(SpringExtension::class)
internal class FaceModelTest {

    @Autowired
    lateinit var model: FaceModel

    @Test
    fun getFaceModel() {
        assertNotNull(model.faceModel)
        val path = Paths.get(".").toAbsolutePath().normalize().toString()
        val imgPath = "$path/src/test/resources/images/001.jpg"
        var img = readImage(imgPath)
        img = model.getInput(img)!!
    }


}