name := "scalearn"

version := "1.0"

scalaVersion := "2.10.2"

resolvers += "ClouderaRepo" at "https://repository.cloudera.com/content/repositories/releases"

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"



scalacOptions ++= Seq("-unchecked", "-deprecation", "-g:vars")



libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-core" % "0.20.2-cdh3u1",
  "com.lambdaworks" %% "jacks" % "2.2.3"
)
