package com.github.lambada9992

import androidx.lifecycle.ViewModel
import com.github.lambada9992.inference.InferenceService
import com.github.lambada9992.statistic.StatisticsService
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class MainViewModel @Inject constructor(
    val inferenceService: InferenceService,
    val statisticsService: StatisticsService
) : ViewModel()
