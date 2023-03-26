package com.github.lambada9992.inference

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

@Module
@InstallIn(SingletonComponent::class)
class InferenceModule {
    @Provides
    fun inferenceService(): InferenceService {
        return InferenceService()
    }
}
