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

class TensorflowObjectDetectionModel(
    override val name: String,
    private val path: String,
    private val inputImageSize: Int,
    private val detectionsNum: Int,
    private val outputIndexes: TensorflowObjectDetectionOutputIndexes,
    private val interpreterOptions: Interpreter.Options = Interpreter.Options(),
    private val inputImageProcessor: ImageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
        .build()
) : InferenceModel() {
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
        val boxes =
            TensorBuffer.createFixedSize(intArrayOf(detectionsNum, 4), DataType.FLOAT32)
        val scores =
            TensorBuffer.createFixedSize(intArrayOf(detectionsNum), DataType.FLOAT32)
        val classes =
            TensorBuffer.createFixedSize(intArrayOf(detectionsNum), DataType.FLOAT32)
        val detectionNum = TensorBuffer.createFixedSize(intArrayOf(detectionsNum), DataType.FLOAT32)

        val outputs = mapOf(
            outputIndexes.boxes to boxes.buffer,
            outputIndexes.classes to classes.buffer,
            outputIndexes.scores to scores.buffer,
            outputIndexes.detNum to detectionNum.buffer
        )

        val tensorImage = TensorImage.fromBitmap(image).let { inputImageProcessor.process(it) }
        interpreter?.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputs)

        return null
    }
}

data class TensorflowObjectDetectionOutputIndexes(
    val boxes: Int,
    val classes: Int,
    val scores: Int,
    val detNum: Int
)
