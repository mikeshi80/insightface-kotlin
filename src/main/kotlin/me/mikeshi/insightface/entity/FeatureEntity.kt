package me.mikeshi.insightface.entity

import javax.persistence.*

@Entity
@Table(name = "features", indexes = [Index(columnList = "name")])
data class FeatureEntity(
        @Id
        @GeneratedValue(strategy = GenerationType.IDENTITY)
        val id: Long,
        val name: String,
        val photo: String,
        @Column(length = 4096)
        val description: String,
        @Column(length = 4096)
        val features: String
)