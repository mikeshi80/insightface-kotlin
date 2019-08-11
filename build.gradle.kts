import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
	id("org.asciidoctor.convert") version "1.5.3"
	id("org.springframework.boot") version "2.2.0.M4"
	id("io.spring.dependency-management") version "1.0.7.RELEASE"
	kotlin("jvm") version "1.3.41"
	kotlin("plugin.spring") version "1.3.41"
}

group = "com.hyron.ai_platform"
version = "0.0.1-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_1_8

repositories {
	maven { url = uri("http://maven.aliyun.com/nexus/content/groups/public/") }
	mavenCentral()
	maven { url = uri("https://repo.spring.io/milestone") }
}

val snippetsDir = file("build/generated-snippets")

dependencies {
	implementation("org.springframework.boot:spring-boot-starter-data-jpa")
	implementation("org.springframework.boot:spring-boot-starter-web")
	implementation("com.fasterxml.jackson.module:jackson-module-kotlin")
	implementation("org.jetbrains.kotlin:kotlin-reflect")
	implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")

	testImplementation("org.springframework.boot:spring-boot-starter-test") {
		exclude(group = "org.junit.vintage", module = "junit-vintage-engine")
		exclude(group = "junit", module = "junit")
	}
	testImplementation("org.springframework.restdocs:spring-restdocs-mockmvc")

	implementation("org.nd4j:nd4j-native-platform:1.0.0-beta4:linux-x86_64")
	implementation("org.apache.mxnet:mxnet-full_2.11-linux-x86_64-cpu:1.5.0")
	implementation("org.datavec:datavec-data-image:1.0.0-beta4")

	testRuntime("com.h2database:h2:1.4.197")

}

tasks.withType<Test> {
	useJUnitPlatform()
	outputs.dir(snippetsDir)
//	testLogging.showStandardStreams = true
}

tasks.withType<KotlinCompile> {
	kotlinOptions {
		freeCompilerArgs = listOf("-Xjsr305=strict")
		jvmTarget = "1.8"
	}
}

tasks.asciidoctor {
	inputs.dir(snippetsDir)
	dependsOn(tasks.test)
}

tasks.bootJar {
	dependsOn(tasks.asciidoctor)

	from("${tasks.asciidoctor.get().outputDir}/html5") {
		into("static/docs")
	}
}