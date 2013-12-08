package org.scalearn.rbm

import scala.io.Source
import java.io.{ PrintWriter, File }
import com.lambdaworks.jacks._
import scala.util.parsing.json._

class StackedRBM() {
  private var layerSizes: List[Int] = List()
  private var customInputSizes: Array[Int] = null
  private var gaussianFlag: List[Boolean] = List()
  var innerRBMs: List[SimpleRBM] = List()

  def innerRBMList = innerRBMs
  def addLayer(numUnits: Int, gaussian: Boolean): StackedRBM = {
    if (!innerRBMs.isEmpty)
      throw new RuntimeException("The network is already built");
    layerSizes = layerSizes :+ numUnits
    gaussianFlag = gaussianFlag :+ gaussian
    this
  }
  def addCustomInputUnits(numUnits: Int): StackedRBM = {
    customInputSizes = new Array(layerSizes.size)
    for (i <- 0 until layerSizes.size)
      customInputSizes(i) = -1

    customInputSizes(customInputSizes.size - 1) = numUnits
    this
  }

  def build(): StackedRBM = {
    if (!innerRBMs.isEmpty)
      return this //already built

    if (layerSizes.size <= 1)
      throw new IllegalArgumentException("Requires at the leat two layers")

    for (i <- 0 until layerSizes.size - 1) {

      var layer: Int = layerSizes(i)

      println("Input size:" + layer + " customInputSizes.size:" + customInputSizes.size + " i:" + i)
      if (!customInputSizes.isEmpty && customInputSizes.size >= i && customInputSizes(i + 1) != -1)
        layer = customInputSizes(i + 1)

      println("Input size:" + layer + " customInputSizes.size:" + customInputSizes.size + " i:" + i)
      innerRBMs = innerRBMs :+ SimpleRBM(layer, layerSizes(i + 1), gaussianFlag(i))

      println("Added RBM " + layer + " -> " + layerSizes(i + 1));
    }

    this
  }

  def iterator(visible: Layer): Iterator[Tuple] = {

    var input: Layer = visible
    var updatedVisible: Layer = visible

    val stackNum: Int = innerRBMs.size

    for (i <- 0 until stackNum) {
      val iRBM: SimpleRBM = innerRBMs(i)

      if (i == (stackNum - 1)) {
        return iRBM.gibbsSamplerHVH(updatedVisible, TupleFactory(input))
      }
      updatedVisible = iRBM.activateHidden(updatedVisible, null)
    }

    throw new AssertionError("code bug");
  }

  def freeEnergy(): Float = {
    var energy: Float = 0.0f

    for (rbm <- innerRBMs) {
      val e = rbm.freeEnergy()
      println("Free:" + e)
      energy += e
    }

    energy
  }

}

object StackedRBM {

  def apply(model: Map[String, Any]): StackedRBM = {
    val rbms: List[Map[String, Any]] = model("stack").asInstanceOf[List[Map[String, Any]]]
    val rbmList: List[SimpleRBM] = (for (rbm <- rbms) yield SimpleRBM(rbm))
    val rbm: StackedRBM = new StackedRBM()
    rbm.innerRBMs = rbmList
    rbm
  }

  def apply(modelFile: String): StackedRBM = {
    var stackedRBM: StackedRBM = null
    val f: File = new File(modelFile)
    if (f.exists()) {
      try {

        val s: Source = Source.fromFile(modelFile)
        val mStr: String = s.getLines.next
        val obj: Option[Any] = JSON.parseFull(mStr)

        obj match {
          case Some(model: Map[String, Any]) => {
            stackedRBM = StackedRBM(model)
          }
          case None => println("Parsing failed")
          case other => println("Unknown data structure: " + other)
        }

      } catch {
        case e: Exception => e.printStackTrace()

      }
    }
    stackedRBM
  }
  def apply(stack: List[SimpleRBM]): StackedRBM = {
    val rbm: StackedRBM = new StackedRBM()
    rbm.innerRBMs = stack
    rbm
  }

  def toJSON(model: StackedRBM): JSONObject = {
    val stack = for (m <- model.innerRBMs) yield SimpleRBM.toJSON(m)
    new JSONObject(Map("stack" -> new JSONArray(stack)))
  }

  def saveModel(model: StackedRBM, modelFile: String): Unit = {
    try {
      val out: PrintWriter = new PrintWriter(modelFile)
      val json = StackedRBM.toJSON(model)
      out.println(json)
      out.close
    } catch {
      case e: Exception => e.printStackTrace()
    }
  }

}