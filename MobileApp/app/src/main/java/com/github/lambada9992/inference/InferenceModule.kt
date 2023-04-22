package com.github.lambada9992.inference

import android.content.Context
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowClassificationModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowObjectDetectionModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowObjectDetectionOutputIndexes
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoSet

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


    // TENSORFLOW

    //// CLASSIFICATION
    @Provides @IntoSet
    fun mobilenetV2Classification(): InferenceModel {
        return TensorflowClassificationModel(
            name = "MobileNetV2_CLASSIFICATION",
            path = "lite-model_american-sign-language_1.tflite",
            inputImageSize = 224,
            numberOfClasses = 24,
        )
    }

    //// OD

    @Provides @IntoSet
    fun mobilenetV2ObjectDetectionInferenceModel(): InferenceModel{
        return TensorflowObjectDetectionModel(
            name = "MobileNetV2_OD_256",
            path = "lite-model_qat_mobilenet_v2_retinanet_256_1.tflite",
            inputImageSize = 256,
            detectionsNum = 100,
            outputIndexes = TensorflowObjectDetectionOutputIndexes(
                boxes = 0,
                detNum = 1,
                classes = 2,
                scores = 3,
            )
        )
    }
}
