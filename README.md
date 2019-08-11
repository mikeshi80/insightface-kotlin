简介
=

尝试将insightface改为kotlin实现，使用了ND4J连实行numpy的操作。

同时用javacp来完成其中opencv的操作。对于skimage中的相似性变换，用javacp中的estimateAffinePartial2D来替代。

模型数据使用insightface的训练模型，由于都放到一个目录下了，所以model-r100-ii的模型名改为face_model，gamodel-r50的改为ga_model。
mtcnn的模型继续使用det1~4的名称。

默认假设
=

有以下几个改动，
 1. 默认num_worker为1，所以代码上没有大于1的对应
 1. 假设det type永远为0
 1. 最后，mtcnn阶段的extended stage还没有写（默认accurate landmark为false)

问题
=

目前加载mtcnn的det1~4（即PNet，RNet等），然后执行到PNet的predict会报错（std:bad_alloc）。已与Python进行了一对一的debug，
传入参数一模一样，python正常，但是scala接口就会报该错。

重现方法，执行UnitTest中的FaceModeTest.getFaceModel()