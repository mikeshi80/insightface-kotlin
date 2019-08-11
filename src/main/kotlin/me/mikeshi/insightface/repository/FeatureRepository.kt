package me.mikeshi.insightface.repository

import me.mikeshi.insightface.entity.FeatureEntity
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository

@Repository
interface FeatureRepository: CrudRepository<FeatureEntity, Long> {
    fun findByName(name: String): FeatureEntity
}