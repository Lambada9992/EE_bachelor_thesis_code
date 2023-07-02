package com.github.lambada9992.inference

import android.content.Context
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.pytorch.PytorchClassificationModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowClassificationModel
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.ElementsIntoSet
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.gpu.GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
import org.tensorflow.lite.nnapi.NnApiDelegate
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
    private val PYTORCH_QUNATIZATIONS = listOf("DYNAMIC_INT8", "FP16", "STATIC_INT8", "NONE")
    private val PYTORCH_SPARSE = listOf("0_0", "0_1", "0_2", "0_3", "0_4", "0_5")

    private fun loadPytorchModels(name: String, imageSize: Int): Set<PytorchClassificationModel> {
        return PYTORCH_QUNATIZATIONS.flatMap { quantization ->
            PYTORCH_SPARSE.map { sparse ->
                val path = "${name}_quantized_${quantization}_sparse_$sparse.ptl"
                PytorchClassificationModel(
                    name = "PT:$name q:${quantization.prepareQuantizationText()} s:${sparse.preparePruningText()}",
                    path = "/storage/0123-4567/pytorch/$path",
                    imageSize = imageSize
                )
            }
        }.toSet()
    }

    // // Classification
    @Provides @ElementsIntoSet fun pt_densenet121(): Set<InferenceModel> {
        return loadPytorchModels("densenet121", 224)
    }
    @Provides @ElementsIntoSet fun pt_densenet169(): Set<InferenceModel> {
        return loadPytorchModels("densenet169", 224)
    }
    @Provides @ElementsIntoSet fun pt_densenet201(): Set<InferenceModel> {
        return loadPytorchModels("densenet201", 224)
    }
    @Provides @ElementsIntoSet fun pt_inception_v3(): Set<InferenceModel> {
        return loadPytorchModels("inception_v3", 299)
    }
    @Provides @ElementsIntoSet fun pt_mobilenetV2(): Set<InferenceModel> {
        return loadPytorchModels("mobilenet_v2", 224)
    }
    @Provides @ElementsIntoSet fun pt_resnet50(): Set<InferenceModel> {
        return loadPytorchModels("resnet50", 224)
    }
    @Provides @ElementsIntoSet fun pt_resnet101(): Set<InferenceModel> {
        return loadPytorchModels("resnet101", 224)
    }
    @Provides @ElementsIntoSet fun pt_resnet152(): Set<InferenceModel> {
        return loadPytorchModels("resnet152", 224)
    }

    // TENSORFLOW
    private val TENSORFLOW_QUANTIZATIONS = listOf("_dint8", "_fint8", "_fp16", "")
    private val TENSORFLOW_SPARSE = listOf("_pruned", "")

    private fun prepareTensorflowModels(name: String, imageSize: Int): Set<TensorflowClassificationModel> {
        val models = TENSORFLOW_QUANTIZATIONS.flatMap { quantization ->
            TENSORFLOW_SPARSE.map { sparse ->
                val path = "${name}${quantization}$sparse.tflite"
                TensorflowClassificationModel(
                    name = "TF:$name q:${quantization.prepareQuantizationText()} s:${sparse.preparePruningText()}",
                    path = "/storage/0123-4567/tensorflow/$path",
                    inputImageProcessor = ImageProcessor.Builder()
                        .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
                        .add(CastOp(DataType.FLOAT32)).add(NormalizeOp(127.5f, 127.5f))
                        .build()
                )
            }
        }.toSet()

        val model_gpu = run {
            val quantization = "_fp16"
            val path = "${name}$quantization.tflite"
            val gpuOptions = GpuDelegateFactory.Options().apply { inferencePreference = INFERENCE_PREFERENCE_SUSTAINED_SPEED }
            TensorflowClassificationModel(
                name = "TF:$name q:${quantization.prepareQuantizationText()} s:${"".preparePruningText()} GPU",
                path = "/storage/0123-4567/tensorflow/$path",
                interpreterOptions = Interpreter.Options().apply { addDelegate(GpuDelegate(gpuOptions)) },
                inputImageProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
                    .add(CastOp(DataType.FLOAT32)).add(NormalizeOp(127.5f, 127.5f))
                    .build()
            )
        }

        val models_nnapi = TENSORFLOW_QUANTIZATIONS.flatMap { quantization ->
            TENSORFLOW_SPARSE.map { sparse ->
                val path = "${name}${quantization}$sparse.tflite"
                TensorflowClassificationModel(
                    name = "TF:$name q:${quantization.prepareQuantizationText()} s:${sparse.preparePruningText()} NNAPI",
                    path = "/storage/0123-4567/tensorflow/$path",
                    interpreterOptions = Interpreter.Options().apply { addDelegate(NnApiDelegate()) },
                    inputImageProcessor = ImageProcessor.Builder()
                        .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
                        .add(CastOp(DataType.FLOAT32)).add(NormalizeOp(127.5f, 127.5f))
                        .build()
                )
            }
        }.toSet()

        return models + setOf(model_gpu) + models_nnapi
    }

    // // CLASSIFICATION
    @Provides @ElementsIntoSet
    fun tf_DenseNet121(): Set<InferenceModel> {
        return prepareTensorflowModels("DenseNet121", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_DenseNet169(): Set<InferenceModel> {
        return prepareTensorflowModels("DenseNet169", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_DenseNet201(): Set<InferenceModel> {
        return prepareTensorflowModels("DenseNet201", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_InceptionResNetV2(): Set<InferenceModel> {
        return prepareTensorflowModels("InceptionResNetV2", 299)
    }

    @Provides @ElementsIntoSet
    fun tf_InceptionV3(): Set<InferenceModel> {
        return prepareTensorflowModels("InceptionV3", 299)
    }

    @Provides @ElementsIntoSet
    fun tf_MobileNet(): Set<InferenceModel> {
        return prepareTensorflowModels("MobileNet", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_mobilenetV2(): Set<InferenceModel> {
        return prepareTensorflowModels("MobileNetV2", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_NASNetMobile(): Set<InferenceModel> {
        return prepareTensorflowModels("NASNetMobile", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_ResNet50(): Set<InferenceModel> {
        return prepareTensorflowModels("ResNet50", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_ResNet101(): Set<InferenceModel> {
        return prepareTensorflowModels("ResNet101", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_ResNet152(): Set<InferenceModel> {
        return prepareTensorflowModels("ResNet152", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_ResNet50V2(): Set<InferenceModel> {
        return prepareTensorflowModels("ResNet50V2", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_ResNet101V2(): Set<InferenceModel> {
        return prepareTensorflowModels("ResNet101V2", 224)
    }

    @Provides @ElementsIntoSet
    fun tf_ResNet152V2(): Set<InferenceModel> {
        return prepareTensorflowModels("ResNet152V2", 224)
    }
}

private fun String.prepareQuantizationText(): String {
    return when (this) {
        "DYNAMIC_INT8", "_dint8" -> "DYNAMIC"
        "FP16", "_fp16" -> "FP16"
        "STATIC_INT8", "_fint8" -> "STATIC"
        "NONE", "" -> "NONE"
        else -> throw IllegalStateException()
    }
}

private fun String.preparePruningText(): String {
    return when (this) {
        "0_0", "" -> "0_0"
        "0_1" -> "0_1"
        "0_2" -> "0_2"
        "0_3" -> "0_3"
        "0_4" -> "0_4"
        "0_5", "_pruned" -> "0_5"
        else -> throw IllegalStateException()
    }
}
