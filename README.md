# Brief

Tried to implement the infer part of insightface by Kotlin, and I use Nd4j to implement the operations of numpy.

And I use javacp to substitute for opencv. For the SimilarTransform in skimage, I did not found the same implementation in Java, so I use the estimateAffinePartial2D in javacp.

The model data have been put in the same directory, so that I rename the model-r100-ii to face_model, and gamodel-r50 to ga_model, mtcnn's model stays same as det1~4.

# Some assumpation

I did some modification base on some asumptions listed below:

 * num_worker is 1, so some code has been simplied.
 * I assume det_type is always 0, and some code has been simplied.

And I did not finish the extended stage yet (so you can think the accurate landmark is false)

# Problem

After det1~4 of mtcnn model loaded, and it will be failed when I try to use PNet to predict, error message is std:bad_alloc. I compared the all properties of the parameters with Python (by debugging one statement by one statement), all parameters seem to be same totally. The next statement is to call C API, but Python works well and Java fails.

# How to reproduce

Execute the Unit Test method `FaceModelTest.getFaceModel()`

# Chinese Version

Sorry for my poor English, if you can read Chinese, please refer [这里](README-CN.md)
