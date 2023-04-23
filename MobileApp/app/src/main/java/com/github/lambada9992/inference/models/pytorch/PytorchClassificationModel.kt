package com.github.lambada9992.inference.models.pytorch

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.github.lambada9992.constants.CLASSES
import com.github.lambada9992.inference.models.ClassificationResult
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.InferenceResult
import org.pytorch.Device
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils


class PytorchClassificationModel(
    override val name: String,
    private val classes: Array<String> = CLASSES.IMAGE_NET,
    private val device: Device = Device.CPU
) : InferenceModel() {
    private var module: Module? = null
    override fun initialize(context: Context) {
        if (module == null) {
            module = LiteModuleLoader.loadModuleFromAsset(context.assets, "model.ptl", device)
        }
    }

    override fun close() {
        module = module?.let {
            it.destroy()
            null
        }
    }

    override fun runInference(image: Bitmap): InferenceResult? {
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            image,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        val outputTensor: Tensor = try {
            module!!.forward(IValue.from(inputTensor)).toTensor()
        } catch (e: Exception){
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

        return try {
            ClassificationResult(name = classes[maxScoreIdx])
        } catch (e: Exception) {
            null
        }
    }
}
