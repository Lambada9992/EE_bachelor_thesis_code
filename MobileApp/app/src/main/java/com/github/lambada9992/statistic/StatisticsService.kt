package com.github.lambada9992.statistic

import android.content.Context
import kotlin.math.ceil

class StatisticsService {
    private val data: MutableMap<String, MutableList<Long>> = mutableMapOf()

    fun clear() {
        data.clear()
    }

    fun addStatistic(name: String, value: Long) {
        data[name]?.apply {
            add(value)
        } ?: data.put(name, mutableListOf(value))
    }

    fun getStatistics(name: String): StatisticsInfo? {
        val selectedData = data[name]?.toList()
        if (selectedData.isNullOrEmpty()) return null
        return StatisticsInfo(
            avg = selectedData.average().toLong(),
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

    fun statisticsAlreadySaved(context: Context, fileName: String): Boolean {
        return context.getFileStreamPath(fileName).exists()
    }

    fun saveStatisticsToFile(context: Context, name: String, fileName: String) {
        val statistics = getStatistics(name)
        val header = "avg,median,max,min,p99,p90\n"
        val text = "${statistics?.avg},${statistics?.median},${statistics?.max},${statistics?.min},${statistics?.p99},${statistics?.p90}\n"
        val fileOutputStream = context.openFileOutput(fileName, Context.MODE_PRIVATE)
        fileOutputStream.write(header.toByteArray())
        fileOutputStream.write(text.toByteArray())
        fileOutputStream.close()
    }
}

data class StatisticsInfo(
    val avg: Long,
    val median: Long,
    val max: Long,
    val min: Long,
    val p99: Long,
    val p90: Long
)
