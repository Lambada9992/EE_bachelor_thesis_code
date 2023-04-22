package com.github.lambada9992.statistic

import kotlin.math.ceil

class StatisticsService {
    private val data: MutableMap<String, MutableList<Long>> = mutableMapOf()

    fun addStatistic(name: String, value: Long) {
        data[name]?.apply {
            add(value)
        }?: data.put(name, mutableListOf(value))
    }

    fun getStatistics(name: String): StatisticsInfo? {
        val selectedData = data[name]?.toList()
        if(selectedData.isNullOrEmpty()) return null
        return StatisticsInfo(
            avg = selectedData.average(),
            median = med(selectedData),
            max = selectedData.max(),
            min = selectedData.min(),
            p99 = percentile(selectedData, 99),
            p90 = percentile(selectedData, 90)
        )
    }



    private fun med(list: List<Long>) = list.sorted().let {
        if (it.size % 2 == 0)
            (it[it.size / 2] + it[(it.size - 1) / 2]) / 2
        else
            it[it.size / 2]
    }

    private fun percentile(list: List<Long>, percentile: Long) = list.sorted().let {
        val index = ceil(percentile / 100.0 * it.size).toInt()
          it[index - 1]
    }
}

data class StatisticsInfo(
    val avg: Double,
    val median: Long,
    val max: Long,
    val min: Long,
    val p99: Long,
    val p90: Long
)
