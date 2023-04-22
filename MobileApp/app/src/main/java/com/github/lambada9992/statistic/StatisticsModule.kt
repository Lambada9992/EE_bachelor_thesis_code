package com.github.lambada9992.statistic

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

@Module
@InstallIn(SingletonComponent::class)
class StatisticsModule {

    @Provides
    fun statisticsService(): StatisticsService{
        return StatisticsService()
    }
}
