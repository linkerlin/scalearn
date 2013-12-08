package org.scalearn.rbm.mnist

import org.scalearn.rbm._

class MnistBinaryRBM(labels: String, images: String) {
  val dr = new MnistReader(labels, images)
  val rbm = SimpleRBM(dr.cols * dr.rows, 10 * 10, false)
  val trainer = new SimpleRBMTrainer(0.2f, 0.001f, Some(0.2f), 0.1f)
  var count = 0

  def learn(): Array[Float] = {

    var inputBatch: List[Layer] = List()

    for (j <- 0 until 30) {
      val trainItem: MnistInstance = dr.getTrainingItem()
      val input: Layer = Layer(trainItem.data)
      input.toBinary
      inputBatch = inputBatch :+ input
    }

    val error: Double = trainer.learn(rbm, inputBatch, false)

    if (count % 100 == 0)
      println("Error = " + error + ", Energy = " + rbm.freeEnergy())

    return inputBatch(inputBatch.size - 1).toArray

  }

  def evaluate(): Iterator[Tuple] = {
    val test: MnistInstance = dr.getTestItem()
    val input: Layer = Layer(test.data);
    input.toBinary
    return rbm.forwardGibbsSampler(input)
  }
}

object MnistBinaryRBM {

  def start(labels: String, images: String, saveto: String): Unit = {
    val training_epochs: Int = 2000
    val m: MnistBinaryRBM = new MnistBinaryRBM(labels, images)

    // train
    for (epoch <- 0 until training_epochs) {
      println("Epoch:" + epoch)
      m.learn
    }

    val test: MnistInstance = m.dr.getTestItem()
    val input: Layer = Layer(test.data);
    input.toBinary
    val it: Iterator[Tuple] = m.rbm.forwardGibbsSampler(input)
    for (j <- 0 until 2) {
      val r: Tuple = it.next
      val v: Array[Float] = Layer.fromBinary(r.visible)
      val features: Array[Int] = (for (i <- 0 until v.length) yield Math.round(v(i))) toArray

      MnistReader.writeToJPEG("./" + test.label + "_" + j + ".jpg", m.dr.cols, m.dr.rows, features)
    }
  }
}

