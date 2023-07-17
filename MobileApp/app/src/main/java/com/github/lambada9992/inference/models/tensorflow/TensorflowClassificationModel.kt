package com.github.lambada9992.inference.models.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.github.lambada9992.constants.CLASSES
import com.github.lambada9992.inference.models.ClassificationResult
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.InferenceResult
import java.io.File
import java.io.FileInputStream
import java.nio.channels.FileChannel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class TensorflowClassificationModel(
    override val name: String,
    private val path: String,
    private val interpreterOptions: Interpreter.Options = Interpreter.Options(),
    private val inputImageProcessor: ImageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .build(),
    private val classes: Array<String> = CLASSES.IMAGE_NET,
) : InferenceModel() {
    private var interpreter: Interpreter? = null
    private val lock = Any()
    private var errorMessage: String? = null

    override fun initialize(context: Context) {
        synchronized(lock) {
            Log.i("TensorflowClassificationModel", "Initialing model name: $name, path: $path")
            if (interpreter == null) {
                try {
                    val file = File(path)
                    val fileIS = FileInputStream(file)
                    val fileBB = fileIS.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
                    interpreter = Interpreter(fileBB, interpreterOptions)
                } catch (e: Exception) {
                    errorMessage = e.message
                }
            }
        }
    }

    override fun close() {
        synchronized(lock) {
            errorMessage?.let { Log.e("TensorflowClassificationModel", "Failed to initialize model: $it") }
            Log.i("TensorflowClassificationModel", "Closing model name: $name")
            interpreter = interpreter?.let {
                it.close()
                null
            }
        }
    }

    override fun runInference(image: Bitmap): InferenceResult? {
        return synchronized(lock) {
            val classification =
                TensorBuffer.createFixedSize(intArrayOf(1, classes.size), DataType.FLOAT32)

            val outputs = mapOf(
                0 to classification.buffer
            )

            val tensorImage = TensorImage.fromBitmap(image).let { inputImageProcessor.process(it) }
            try {
                interpreter?.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputs)
            } catch (e: Exception) {
                return null
            }

            val scores = classification.floatArray

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
