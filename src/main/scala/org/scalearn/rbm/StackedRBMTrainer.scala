package org.scalearn.rbm

class StackedRBMTrainer(stackedRBM: StackedRBM, momentum: Float, l2: Float, targetSparsity: Option[Float], learningRate: Float) {
  val inputTrainer: SimpleRBMTrainer = new SimpleRBMTrainer(momentum, l2, targetSparsity, learningRate)

  def setLearningRate(rate: Float) {
    //TODO: make learning rate mutable.
  }

  def learn(bottomBatch: List[Layer], topBatch: List[Layer], stopAt: Int): Double = {
    //TODO: fix condition
    if (!topBatch.isEmpty && topBatch.size != bottomBatch.size)
      throw new IllegalArgumentException("Layers units do not match: TopBatch != BottomBatch")

    if (stopAt < 0 || stopAt > stackedRBM.innerRBMList.size)
      throw new IllegalArgumentException("Invalid stopAt layer")

    var nextInputs: Array[Layer] = bottomBatch.toArray

    for (i <- 0 until stopAt) {
      //At stopping point we need to learn RBM
      if (i == stopAt - 1) {
        return inputTrainer.learn(stackedRBM.innerRBMList(i), nextInputs.toList, false)
      }

      //Use the hidden of this layer as the inputs of the next layer
      for (j <- 0 until nextInputs.size) {
        var next: Layer = stackedRBM.innerRBMList(i).activateHidden(nextInputs(j), null)

        if (!topBatch.isEmpty && i == stopAt - 2) {
          val nextConcat: Array[Float] = next.toArray ++ topBatch(j).toArray
          next = Layer(nextConcat)
        }

        nextInputs(j) = next
      }
    }

    throw new AssertionError("Did not learn any model. No stopAt layer.")
  }

}
