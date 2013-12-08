package org.scalearn.rbm.mnist

import org.scalearn.rbm._

import scala.io._
import scala.collection.mutable._
import scala.util.Random

import java.io._

import javax.imageio.ImageIO
import java.awt.Graphics2D
import java.awt.image.{ BufferedImage, RenderedImage, WritableRaster }
import java.util.zip.GZIPInputStream

class MnistReader(labelsFile: String, imagesFile: String) {

  val labelSource = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsFile)))
  val imageSource = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesFile)))
  val magic: Int = labelSource.readInt()
  val labelCount: Int = labelSource.readInt()
  println("Labels magic=" + magic + ", count=" + labelCount)
  val rand: Random = new Random(1)
  val magic2: Int = imageSource.readInt()
  val imageCount: Int = imageSource.readInt()

  val rows: Int = imageSource.readInt()
  val cols: Int = imageSource.readInt()

  println("Images magic=" + magic2 + " count=" + imageCount + " rows=" + rows + " cols=" + cols)

  var trainingSet: HashMap[String, ArrayBuffer[MnistInstance]] = HashMap()
  var testSet: HashMap[String, ArrayBuffer[MnistInstance]] = HashMap()

  generateDataSet()

  def generateDataSet() = {
    for (i <- 0 until imageCount) {
      val (l, instance) = nextInstance()
      val inst = MnistInstance(l.toString, instance)
      if (rand.nextDouble > 0.3) {
        var instances = testSet.get(inst.label)
        instances match {
          case None => {
            val l: ArrayBuffer[MnistInstance] = ArrayBuffer(inst)
            testSet.put(inst.label, l)
          }
          case Some(x) => {
            x += inst
          }
        }

      } else {
        var instances = trainingSet.get(inst.label)
        instances match {
          case None => {
            val l: ArrayBuffer[MnistInstance] = ArrayBuffer(inst)
            trainingSet.put(inst.label, l)
          }
          case Some(x) => {
            x += inst
          }
        }
      }
    }
  }

  def getTestItem(): MnistInstance = {
    val instances = testSet.get(rand.nextInt(10).toString).get
    instances(rand.nextInt(instances.size))

  }
  def getTrainingItem(): MnistInstance = {
    val instances = trainingSet.get(rand.nextInt(10).toString).get
    instances(rand.nextInt(instances.size))
  }
  def getTrainingItem(i: Int): MnistInstance = {
    val instances = trainingSet.get(i.toString).get
    return instances(rand.nextInt(instances.size))
  }

  def nextInstance(): (Int, Array[Int]) = {
    //TODO: check to see if > number of images?
    val label = labelSource.readUnsignedByte()
    val results = (for (i <- 0 until rows; j <- 0 until cols) yield imageSource.readUnsignedByte())
    (label, results.toArray[Int])
  }

  def testInstances: HashMap[String, ArrayBuffer[MnistInstance]] = {
    testSet
  }
  def trainingInstances(): HashMap[String, ArrayBuffer[MnistInstance]] = {
    trainingSet
  }

}

object MnistReader {

  def writeToPPM(file: String, features: Array[Int]) = {
    val ppmOut: BufferedWriter = new BufferedWriter(new FileWriter(file))
    val rows: Int = 28
    val cols: Int = 28

    ppmOut.write("P3\n")
    ppmOut.write("" + rows + " " + cols + " 255\n")

    val image = features.grouped(28)toVector

    for (i <- 0 until rows) {
      val triples = (for (j <- 0 until cols) yield Vector(image(i)(j), image(i)(j), image(i)(j)))
      for (a <- triples) {
        val row = a.mkString(" ")
        ppmOut.write(row + " ")
      }
    }
    ppmOut.close
  }
  def writeToJPEG(file: String, cols: Int, rows: Int, features: Array[Int]): Unit = {
    val in: BufferedImage = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB)
    val out: FileOutputStream = new FileOutputStream(file)
    val r: WritableRaster = in.getRaster
    r.setDataElements(0, 0, cols, rows, features)
    ImageIO.write(in, "JPG", out)
    out.close
  }

  def main(args: Array[String]) {

    val labelFile = if (args.size == 3) args(0) else "/home/htosun/dev/dataset/train-labels-idx1-ubyte.gz"
    val featureFile = if (args.size == 3) args(1) else "/home/htosun/dev/dataset/train-images-idx3-ubyte.gz"
    val outputDir = if (args.size == 3) args(2) else "/home/htosun/dev/output"

    val r = new MnistReader(labelFile, featureFile)

  }
}