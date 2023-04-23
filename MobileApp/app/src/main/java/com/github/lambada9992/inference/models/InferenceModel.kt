package com.github.lambada9992.inference.models

import android.content.Context
import android.graphics.Bitmap

abstract class InferenceModel {
    abstract val name: String

    abstract fun initialize(context: Context)
    abstract fun close()
    abstract fun runInference(image: Bitmap): InferenceResult?
}

abstract class InferenceResult()
data class ObjectDetectionResult(
    val x: String
): InferenceResult()

data class ClassificationResult(
    val name: String
): InferenceResult()
