package com.github.lambada9992.ui.screens.mainscreen.components

import androidx.compose.foundation.layout.Column
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.github.lambada9992.inference.InferenceService
import com.github.lambada9992.statistic.StatisticsInfo
import com.github.lambada9992.statistic.StatisticsService
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.delay

@Composable
fun InferenceStats(inferenceService: InferenceService, statisticsService: StatisticsService) {
    var statsToDisplay by remember { mutableStateOf<StatisticsInfo?>(null) }

    LaunchedEffect(Unit) {
        while (true) {
            delay(1.seconds)
            val modelName = inferenceService.selectedInferenceModel?.name
            statsToDisplay = modelName?.let {
                statisticsService.getStatistics(it)
            }
        }
    }

    Column() {
        Text("avg: ${statsToDisplay?.avg ?: ""}")
        Text("median: ${statsToDisplay?.median ?: ""}")
        Text("max: ${statsToDisplay?.max ?: ""}")
        Text("min: ${statsToDisplay?.min ?: ""}")
        Text("p99: ${statsToDisplay?.p99 ?: ""}")
        Text("p90: ${statsToDisplay?.p90 ?: ""}")
    }
}
