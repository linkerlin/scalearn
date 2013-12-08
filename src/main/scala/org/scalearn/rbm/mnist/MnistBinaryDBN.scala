package org.scalearn.rbm.mnist

import org.scalearn.rbm._

/**
 *
 * This is copy of https://github.com/tjake/rbm-dbn-MNIST
 */
class MnistBinaryDBN(labels: String, images: String) {
  val dr = new MnistReader(labels, images)
  var rbm = new StackedRBM()
  val trainer = new StackedRBMTrainer(rbm, 0.5f, 0.001f, Some(0.2f), 0.2f)

  def learn(iterations: Int, addLabels: Boolean, stopAt: Int): Unit = {

    for (p <- 0 until iterations) {

      // Get random input
      var inputBatch: List[Layer] = List()
      var labelBatch: List[Layer] = List()

      for (j <- 0 until 30) {
        val trainItem: MnistInstance = dr.getTrainingItem()
        val input: Layer = Layer(trainItem.data)

        input.toBinary
        inputBatch = inputBatch :+ input

        if (addLabels) {
          var labelInput: Array[Float] = new Array(10)
          labelInput(trainItem.label.toInt) = 1.0f
          labelBatch = labelBatch :+ Layer(labelInput)
        }
      }

      val error: Double = trainer.learn(inputBatch, labelBatch, stopAt)

      if (p % 100 == 0)
        println("Iteration " + p + ", Error = " + error + ", Energy = " + rbm.freeEnergy())
    }

  }

  def evaluate(test: MnistInstance): Iterator[Tuple] = {

    var input: Layer = Layer(test.data)
    input.toBinary

    val stackNum: Int = rbm.innerRBMList.size

    for (i <- 0 until stackNum) {

      val model: SimpleRBM = rbm.innerRBMList(i)

      if (model.biasVisible.size > input.size) {
        input = Layer((for (j <- 0 until model.biasVisible.size) yield if (j < input.size) input(j) else 0.1f) toArray)
      }

      if (i == (stackNum - 1)) {
        return model.forwardGibbsSampler(input)
      }

      input = model.activateHidden(input, null)
    }

    return Iterator()
  }

}

object MnistBinaryDBN {

  def start(labels: String, images: String, saveto: String) = {

    val m: MnistBinaryDBN = new MnistBinaryDBN(labels, images)

    val numIterations: Int = 1000

    var prevStateLoaded: Boolean = false

    //      val s:StackedRBM = StackedRBM(saveto)
    //      if(s != null) {    
    //        m.rbm = s
    //        prevStateLoaded = true
    //      }
    //         
    //    

    if (!prevStateLoaded) {

      m.rbm
        .addLayer(m.dr.rows * m.dr.cols, false)
        .addLayer(500, false)
        .addLayer(500, false)
        .addLayer(2000, false)
        .addCustomInputUnits(510)
        .build()
      println("Iteration:" + numIterations)
      println("Training level 1")
      m.learn(numIterations, false, 1)
      println("Training level 2")
      m.learn(numIterations, false, 2)
      println("Training level 3")
      m.learn(numIterations, true, 3)
      println("Iteration:" + numIterations)

      // StackedRBM.saveModel(m.rbm, saveto)
    }

    var numCorrect: Double = 0
    var numWrong: Double = 0
    var numAlmost: Double = 0.0

    while (true) {
      val testCase: MnistInstance = m.dr.getTestItem()

      val it: Iterator[Tuple] = m.evaluate(testCase)

      var labeld: Array[Float] = new Array(10)

      for (i <- 0 until 2) {
        val t: Tuple = it.next

        var k = 0
        for (j <- (t.visible.size - 10) until t.visible.size if (k < 10)) {
          labeld(k) += t.visible(j)
          k += 1
        }
      }

      var max1: Float = 0.0f
      var max2: Float = 0.0f
      var p1: Int = -1
      var p2: Int = -1

      println("Label is: " + testCase.label)

      for (i <- 0 until labeld.length) {
        labeld(i) /= 2
        if (labeld(i) > max1) {
          max2 = max1
          max1 = labeld(i)

          p2 = p1
          p1 = i
        }
      }

      println(", Winner is " + p1 + "(" + max1 + ") second is " + p2 + "(" + max2 + ")")
      if (p1 == testCase.label.toInt) {
        numCorrect += 1
      } else if (p2 == testCase.label.toInt) {
        numAlmost += 1
      } else {
        numWrong += 1
      }

      println("Error Rate = " + ((numWrong / (numAlmost + numCorrect + numWrong)) * 100))

    }
  }
}