package com.github.lambada9992.inference.models

import android.graphics.Bitmap

abstract class InferenceModel {
    abstract val name: String

    abstract fun runInference(image: Bitmap)
}

data class InferenceResult(
    val x: String
)
