package org.scalearn.rbm

import scala.io.Source
import scala.util.Random
import java.io.{ PrintWriter, File , DataOutput, DataInput}
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

  def generateSamples(batchSize:Int): List[Pair[Int, Layer]] = {
    var result:List[Pair[Int, Layer]] = List()
    val random = new Random()
    for(m <- 0 until batchSize){
      var current:Int = random.nextInt(10)

      println("Size = "+this.innerRBMs.size)
      val r:SimpleRBM = this.innerRBMList(this.innerRBMList.size - 1)
      var input:Layer = Layer(r.biasVisible.size)
      for (i <- 0 until input.size)
        input.set(i, 0.0f)

      val labelIndex = input.size - 10 + current
      input.set(labelIndex, 100000.0f)
      println("Values for label:"+current+" index:"+(labelIndex)+ " value"+input(labelIndex))
      val it:Iterator[Tuple] = r.forwardGibbsSampler(input)
      for(l<-0 until 2){
        input = it.next.visible
      }
  
      for (i <- (this.innerRBMList.size - 2) to 0 by -1){
        val rbm:SimpleRBM =  this.innerRBMList(i)
        if (input.size > rbm.biasHidden.size) {
          val hidden:Array[Float] = (for(j <- 0 until rbm.biasHidden.size) yield input(i)) toArray
        
          input = Layer(hidden)
         
        }
        input = rbm.activateVisible(input, null)
      }
      result = result :+ (current, input)
    }
    result
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
  
  def save(output:DataOutput)={
    output.write(Layer.MAGIC)
    output.writeInt(innerRBMs.size)
    for (rbm <- innerRBMs) {
      rbm.save(output)
    }
  }
}

object StackedRBM {

  def apply(input:DataInput):StackedRBM  = {
    val magic:Array[Byte] = new Array(4)
    input.readFully(magic)
    val rbms:Int = input.readInt()
    val rbm: StackedRBM = new StackedRBM()
    rbm.innerRBMs = (for(i <- 0 until rbms) yield SimpleRBM(input)).toList
    rbm
  }
  
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