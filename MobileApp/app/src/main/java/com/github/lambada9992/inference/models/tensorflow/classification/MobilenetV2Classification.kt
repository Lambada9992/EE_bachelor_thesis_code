/*
package com.github.lambada9992.inference.models.tensorflow.classification

import android.content.Context
import android.graphics.Bitmap
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.InferenceResult
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class MobilenetV2Classification : InferenceModel() {
    override val name: String = "Classification"
    private val MODEL_PATH = "lite-model_american-sign-language_1.tflite"
    private var detector: ObjectDetector? = null

    override fun initialize(context: Context) {
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(BaseOptions.builder().build())
            .setMaxResults(10)
            .build()

        detector = ObjectDetector.createFromFileAndOptions(context, MODEL_PATH, options)
    }

    override fun close(){
        detector?.close()
        detector = null
    }

    override fun runInference(image: Bitmap): InferenceResult? {
        val result = detector?.detect(TensorImage.fromBitmap(image))
        return null
    }
}
*/
