package com.github.lambada9992.inference.models.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.github.lambada9992.inference.models.InferenceModel
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions

class TensorflowInferenceModel(): InferenceModel(){
    override val name: String = ""
    val MODEL_PATH = "lite-model_object_detection_mobile_object_localizer_v1_1_metadata_2.tflite"
    var detector: ObjectDetector? = null

    fun initialize(context: Context){
        val options = ObjectDetectorOptions.builder()
            .setBaseOptions(BaseOptions.builder().build())
            .setMaxResults(10)
            .build()

        detector = ObjectDetector.createFromFileAndOptions(context, MODEL_PATH, options)
    }

    override fun runInference(image: Bitmap) {
        val result = detector?.detect(TensorImage.fromBitmap(image))
        Log.d("RESULT", result.toString())
    }
}
