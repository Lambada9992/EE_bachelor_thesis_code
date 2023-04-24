package com.github.lambada9992.inference

import android.content.Context
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.pytorch.PytorchClassificationModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowClassificationModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowObjectDetectionModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowObjectDetectionOutputIndexes
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoSet
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp

@Module
@InstallIn(SingletonComponent::class)
class InferenceModule {

    @Provides
    fun inferenceService(
        models: Set<@JvmSuppressWildcards InferenceModel>,
        @ApplicationContext applicationContext: Context,
    ): InferenceService {
        return InferenceService(models, applicationContext)
    }

    // Pytorch
    //// Classification
    @Provides @IntoSet
    fun pytorchTest(): InferenceModel {
        return PytorchClassificationModel(
            name = "PytorchTest"
        )
    }

    // TENSORFLOW

    //// CLASSIFICATION
    @Provides @IntoSet
    fun mobilenetV2Classification(): InferenceModel {
        return TensorflowClassificationModel(
            name = "MobileNetV2_CLASSIFICATION",
            path = "lite-model_mobilenet_v2_100_224_fp32_1.tflite",
            inputImageSize = 224,
            numberOfClasses = 1001,
            inputImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(CastOp(DataType.FLOAT32))
                .add(NormalizeOp(127.5f , 127.5f))
            .build()
        )
    }

    //// OD

//    @Provides @IntoSet
//    fun mobilenetV2ObjectDetectionInferenceModel(): InferenceModel{
//        return TensorflowObjectDetectionModel(
//            name = "MobileNetV2_OD_256",
//            path = "lite-model_qat_mobilenet_v2_retinanet_256_1.tflite",
//            inputImageSize = 256,
//            detectionsNum = 100,
//            outputIndexes = TensorflowObjectDetectionOutputIndexes(
//                boxes = 0,
//                detNum = 1,
//                classes = 2,
//                scores = 3,
//            )
//        )
//    }
}
