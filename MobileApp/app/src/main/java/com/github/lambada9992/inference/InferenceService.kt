package com.github.lambada9992.inference

import android.content.Context
import com.github.lambada9992.inference.models.InferenceModel


class InferenceService(
    private val models: Set<InferenceModel>,
    private val applicationContext: Context
) {
    var selectedInferenceModel: InferenceModel? = null


    fun selectModel(modelName: String){
        val model = models.firstOrNull {it.name == modelName}
        selectedInferenceModel?.run { close() }
        selectedInferenceModel = model?.apply { initialize(applicationContext) }
    }

    val getModelsNames = models.map { it.name }
}
