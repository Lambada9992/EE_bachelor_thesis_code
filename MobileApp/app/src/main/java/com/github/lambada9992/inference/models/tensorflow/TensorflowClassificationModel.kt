package com.github.lambada9992.inference.models.tensorflow

import android.content.Context
import android.graphics.Bitmap
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.InferenceResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class TensorflowClassificationModel(
    override val name: String,
    private val path: String,
    private val inputImageSize: Int,
    private val numberOfClasses: Int,
    private val interpreterOptions: Interpreter.Options = Interpreter.Options(),
    private val inputImageProcessor: ImageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
        .build()
): InferenceModel() {
    private var interpreter: Interpreter? = null

    override fun initialize(context: Context) {
        if (interpreter == null) {
            interpreter = Interpreter(FileUtil.loadMappedFile(context, path), interpreterOptions)
        }
    }

    override fun close() {
        interpreter = interpreter?.let {
            it.close()
            null
        }
    }


    override fun runInference(image: Bitmap): InferenceResult? {
        val classification =
            TensorBuffer.createFixedSize(intArrayOf(1, numberOfClasses), DataType.FLOAT32)

        val outputs = mapOf(
            0 to classification.buffer
        )

        val tensorImage = TensorImage.fromBitmap(image).let { inputImageProcessor.process(it) }
        interpreter?.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputs)

        return null
    }

}
