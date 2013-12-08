package org.scalearn.rbm

import scala.math._

//TODO: document each parameters in detail
class SimpleRBMTrainer(momentum: Float, l2: Float, targetSparsity: Option[Float] = None, learningRate: Float) {
  var gw: Array[Layer] = null
  var gv: Layer = null
  var gh: Layer = null

  def resetConfiguration(rbm: SimpleRBM) = {
    //TODO: initialize similar to initial values in python code
    if (gw == null || gw.length != rbm.biasHidden.size || gw(0).size != rbm.biasVisible.size) {
      gw = new Array(rbm.biasHidden.size)
      for (i <- 0 until gw.length)
        gw(i) = Layer(rbm.biasVisible.size)

      gv = Layer(rbm.biasVisible.size)
      gh = Layer(rbm.biasHidden.size)
    } else {
      for (i <- 0 until gw.length)
        gw(i).zero()

      gv.zero()
      gh.zero()
    }
  }

  // Contrastive Divergance of two steps or one step?
  def cd(rbm: SimpleRBM, inputBatch: List[Layer], reverse: Boolean) = {

    for (input <- inputBatch) {
      try {
        val it: Iterator[Tuple] = if (reverse) rbm.reverseGibbsSampler(input) else rbm.forwardGibbsSampler(input)

        val t1: Tuple = it.next
        val t2: Tuple = it.next

        for (i <- 0 until gw.length; j <- 0 until gw(i).size)
          gw(i).+(j, (t1.hidden(i) * t1.visible(j)) - (t2.hidden(i) * t2.visible(j)))

        for (i <- 0 until gv.size)
          gv.+(i, t1.visible(i) - t2.visible(i))

        for (i <- 0 until gh.size)
          gh.+(i, if (!targetSparsity.isDefined) t1.hidden(i) - t2.hidden(i) else targetSparsity.get - t1.hidden(i))

      } catch {
        case ex: Exception => {
          println("Training  exception:" + ex.getStackTraceString)
        }
      }
    }

  }
  def learn(rbm: SimpleRBM, inputBatch: List[Layer], reverse: Boolean): Double = {
    val batchsize: Int = inputBatch.size

    resetConfiguration(rbm)

    // Contrastive Divergance
    cd(rbm, inputBatch, reverse)

    // Average
    for (i <- 0 until gw.length; j <- 0 until gw(i).size) {
      gw(i)./(j, batchsize)
      val dw = gw(i)(j)
      //TODO: verify this formula
      rbm.weights(i).+(j, learningRate * (momentum * (dw - l2 * rbm.weights(i)(j)) + (1 - momentum) * dw))
    }

    var error: Double = 0.0

    for (i <- 0 until gv.size) {
      gv./(i, batchsize)
      val dv = gv(i)
      error += pow(dv, 2)
      //XXHT: verify formula
      rbm.biasVisible.+(i, learningRate * (momentum * dv * rbm.biasVisible(i) + (1 - momentum) * dv))
    }

    if (targetSparsity.isDefined) {
      for (i <- 0 until gh.size) {
        gh./(i, batchsize)
        //XXHT: never tested sparsity
        gh.set(i, targetSparsity.get - gh(i))
      }
    } else {
      for (i <- 0 until gh.size) {
        gh./(i, batchsize)
        val meanHidden = gh(i)
        rbm.biasHidden.+(i, learningRate * (momentum * meanHidden * rbm.biasHidden(i) + (1 - momentum) * meanHidden))
      }
    }

    sqrt(error / gv.size)
  }

}
