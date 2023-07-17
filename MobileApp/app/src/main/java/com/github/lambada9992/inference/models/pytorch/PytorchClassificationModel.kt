package com.github.lambada9992.inference.models.pytorch

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.github.lambada9992.constants.CLASSES
import com.github.lambada9992.inference.models.ClassificationResult
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.InferenceResult
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class PytorchClassificationModel(
    override val name: String,
    private val path: String,
    private val imageSize: Int,
    private val classes: Array<String> = CLASSES.IMAGE_NET,
) : InferenceModel() {
    private var module: Module? = null
    private val lock = Any()
    private var errorMessage: String? = null
    override fun initialize(context: Context) {
        synchronized(lock) {
            Log.i("PytorchClassificationModel", "Initialing model name: $name")
            if (module == null) {
                try {
                    module = LiteModuleLoader.load(path)
                } catch (e: Exception) {
                    errorMessage = e.message
                }
            }
        }
    }

    override fun close() {
        synchronized(lock) {
            errorMessage?.let { Log.e("PytorchClassificationModel", "Failed to initialize model: $it") }
            Log.i("PytorchClassificationModel", "Closing model name: $name")
            module = module?.let {
                it.destroy()
                null
            }
        }
    }

    override fun runInference(image: Bitmap): InferenceResult? {
        return synchronized(lock) {
            val imageResized = Bitmap.createScaledBitmap(image, imageSize, imageSize, true)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                imageResized,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            inputTensor.shape()

            val outputTensor: Tensor = try {
                module!!.forward(IValue.from(inputTensor)).toTensor()
            } catch (e: Exception) {
                Log.e("PytorchClassificationModel", "Failed to inference")
                return null
            }
            val scores: FloatArray = outputTensor.dataAsFloatArray

            var maxScore = -Float.MAX_VALUE
            var maxScoreIdx = -1
            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }

            try {
                ClassificationResult(name = classes[maxScoreIdx])
            } catch (e: Exception) {
                null
            }
        }
    }
}
