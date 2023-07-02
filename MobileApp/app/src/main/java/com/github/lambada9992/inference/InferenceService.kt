package com.github.lambada9992.inference

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MediatorLiveData
import androidx.lifecycle.MutableLiveData
import com.github.lambada9992.inference.models.ClassificationResult
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.inference.models.InferenceResult
import com.github.lambada9992.statistic.StatisticsService
import com.github.lambada9992.ui.screens.mainscreen.prepareFileName

class InferenceService(
    private val models: Set<InferenceModel>,
    private val applicationContext: Context
) {
    private val result = MutableLiveData<InferenceResult?>(null)
    var selectedInferenceModel: InferenceModel? = null

    val classificationResult: LiveData<String>
        get() = MediatorLiveData<String>().apply {
            addSource(result) { result ->
                value = when (result) {
                    is ClassificationResult -> result.name
                    else -> ""
                }
            }
        }

    fun switchToNextModel(
        context: Context,
        statisticsService: StatisticsService
    ): Boolean {
        var index = selectedInferenceModel?.let { models.indexOf(it) + 1 } ?: 0
        while (index < models.size && statisticsService.statisticsAlreadySaved(
                context, models.elementAt(index).name.prepareFileName() + ".csv"
            )
        ) {
            index++
        }
        Log.i("INDEX", index.toString())
        if (index < models.size) {
            selectModel(models.elementAt(index).name)
        } else {
            selectModel("")
            return false
        }
        return true
    }

    fun setInferenceResult(result: InferenceResult?) {
        this.result.postValue(result)
    }

    fun selectModel(modelName: String) {
        setInferenceResult(null)
        val model = models.firstOrNull { it.name == modelName }
        selectedInferenceModel?.run { close() }
        selectedInferenceModel = model?.apply { initialize(applicationContext) }
    }

    val getModelsNames = models.map { it.name }
}
